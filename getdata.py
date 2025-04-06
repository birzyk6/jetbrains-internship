import requests
import os
import urllib.request
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import asyncio
import aiohttp
import aiofiles
import sys
import traceback
import logging

# API URL constant
API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("geo_downloader")


# Helper function to read PMIDs from a file
def getPMIDs(path: str) -> list[int]:
    try:
        with open(path, "r") as f:
            pmids = [int(line.strip()) for line in f.readlines()]
        return pmids
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return []
    except ValueError:
        print("Error: Invalid PMID format in file.")
        return []


def getLinkedDatasets(pmids: list[int]) -> list[str]:
    """
    Fetch linked GEO datasets for given PMIDs and return a list of GSE IDs.
    """
    response = requests.get(
        f"{API_URL}elink.fcgi",
        params={
            "dbfrom": "pubmed",
            "db": "gds",
            "linkname": "pubmed_gds",
            "id": ",".join(map(str, pmids)),
            "retmode": "json",
        },
    )

    if response.status_code == 200:
        data = response.json()
        try:
            links = data["linksets"][0]["linksetdbs"][0]["links"]
            return [str(link) for link in links]
        except (KeyError, IndexError):
            return []
    else:
        print(f"Error fetching linked datasets: {response.status_code}")
        return []


def retrieveLinkedData(dataset_ids: list[str]) -> requests.Response | None:
    """
    Retrieve linked data from the GEO database.
    """
    if not dataset_ids:
        print("No dataset IDs provided.")
        return None

    # Join dataset IDs with commas for the API request
    ids_param = ",".join(dataset_ids)

    try:
        response = requests.get(
            f"{API_URL}esummary.fcgi",
            params={
                "db": "gds",
                "id": ids_param,
                "retmode": "xml",  # Ensure XML format
            },
            timeout=30,
        )

        if response.status_code == 200:
            return response
        else:
            print(f"Error fetching detailed data: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None


def get_ftp_links(dataset_ids: list[str]) -> dict:
    """
    Get FTP links for all dataset IDs in a single request.
    Returns a dictionary mapping dataset IDs to their FTP links.
    """
    result = {}

    if not dataset_ids:
        print("No dataset IDs provided.")
        return result

    try:
        # Combine all dataset IDs into a comma-separated string
        gdids_param = ",".join(dataset_ids)

        print(
            f"Fetching FTP links for {len(dataset_ids)} datasets in a single request..."
        )

        # Make a single request for all datasets
        url = f"{API_URL}esummary.fcgi"
        response = requests.get(
            url,
            params={"db": "gds", "id": gdids_param, "retmode": "xml"},
            timeout=60,  # Longer timeout for bulk request
        )

        if response.status_code != 200:
            print(f"Failed to fetch summaries: Status code {response.status_code}")
            return result

        # Parse the XML response
        root = ET.fromstring(response.content)

        # Find all Docsum elements
        docsums = root.findall(".//DocSum")
        print(f"Found {len(docsums)} DocSum elements in response")

        for docsum in docsums:
            # Get the dataset ID from the Id element
            id_elem = docsum.find("Id")
            if id_elem is None or not id_elem.text:
                continue

            gdid = id_elem.text

            # Find the FTPLink item directly
            ftp_link = None

            # Look for the Item with Name="FTPLink"
            for item in docsum.findall(".//Item[@Name='FTPLink']"):
                if item.text and item.text.startswith("ftp://"):
                    ftp_link = item.text
                    break

            # If not found, try TargetFTPLink
            if not ftp_link:
                for item in docsum.findall(".//Item[@Name='TargetFTPLink']"):
                    if item.text and item.text.startswith("ftp://"):
                        ftp_link = item.text
                        break

            # If neither direct link is found, look in the summary text
            if not ftp_link:
                for item in docsum.findall(".//Item[@Name='summary']"):
                    if item.text:
                        match = re.search(
                            r"ftp://ftp\.ncbi\.nlm\.nih\.gov/geo/series/\S+", item.text
                        )
                        if match:
                            ftp_link = match.group(0)
                            break

            if ftp_link:
                # Ensure link ends with a trailing slash for consistency
                if not ftp_link.endswith("/"):
                    ftp_link += "/"
                result[gdid] = ftp_link
                print(f"Found FTP link for dataset {gdid}: {ftp_link}")

    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        # Try to determine what part of the XML is malformed
        print("Response content preview:")
        print(response.content[:200])

    except Exception as e:
        print(f"Error getting FTP links: {str(e)}")

    print(f"Found FTP links for {len(result)} out of {len(dataset_ids)} datasets")
    return result


def extract_gse_id_from_ftp(ftp_link: str) -> str:
    """
    Extract the complete GSE ID from an FTP link.
    """
    # Try extracting from the path structure
    parts = ftp_link.rstrip("/").split("/")
    if parts:
        last_part = parts[-1]
        if last_part.startswith("GSE"):
            return last_part

    # If all else fails, return unknown
    return "Unknown"


async def download_file_async(url, local_path):
    """
    Download a single file asynchronously without retries.
    Returns True if successful, False otherwise.
    """
    # Convert FTP URLs to HTTP URLs if needed - NCBI supports both protocols
    if url.startswith("ftp://"):
        http_url = url.replace("ftp://", "https://")
    else:
        http_url = http_url

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

        # Use aiohttp for asynchronous HTTP requests
        timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(http_url) as response:
                if response.status == 200:
                    # Stream the response to file to avoid memory issues
                    async with aiofiles.open(local_path, "wb") as f:
                        # Read and write in chunks
                        chunk_size = 8192  # 8KB chunks
                        while True:
                            chunk = await response.content.read(chunk_size)
                            if not chunk:
                                break
                            await f.write(chunk)
                    return True
                else:
                    print(f"Error downloading {http_url}: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"Error downloading {http_url}: {str(e)}")
        return False


def worker_task(urls_batch):
    """
    Worker function that processes a batch of URLs asynchronously.
    Each worker handles multiple downloads asynchronously.

    Args:
        urls_batch: List of tuples (url, local_path, gdid)

    Returns:
        List of successful downloads as (gdid, local_path) tuples
    """
    # Create a new event loop for this thread safely
    loop = None
    try:
        # Create and set the event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process_batch():
            tasks = []
            for url, path, gdid in urls_batch:
                tasks.append(download_and_track(url, path, gdid))
            return await asyncio.gather(*tasks)

        async def download_and_track(url, path, gdid):
            success = await download_file_async(url, path)
            return (gdid, path, success)

        # Run the async tasks in this thread's event loop
        results = loop.run_until_complete(process_batch())
        return [(gdid, path) for gdid, path, success in results if success]
    except Exception as e:
        print(f"Error in worker task: {str(e)}")
        # If async fails, try synchronous download as fallback
        return download_batch_sync(urls_batch)
    finally:
        # Always properly close the loop to prevent resource leaks
        if loop:
            try:
                # Cancel all running tasks
                pending = asyncio.all_tasks(loop=loop)
                for task in pending:
                    task.cancel()

                # Run the event loop until all tasks are cancelled
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

                # Close the loop
                loop.close()
            except Exception as e:
                print(f"Error closing event loop: {str(e)}")


def download_file_simple(url, local_path):
    """
    Download a single file using sync requests.
    Returns True if successful, False otherwise.
    """
    # Convert FTP URLs to HTTP URLs if needed
    if url.startswith("ftp://"):
        http_url = url.replace("ftp://", "https://")
    else:
        http_url = url

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

        # Use regular requests for sync download
        response = requests.get(http_url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        else:
            print(f"Error downloading {http_url}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {http_url}: {str(e)}")
        return False


def download_batch_sync(urls_batch):
    """
    Synchronous fallback function to download a batch of files.
    Used when asyncio causes thread/loop errors.
    """
    successful = []
    for url, path, gdid in urls_batch:
        if download_file_simple(url, path):
            successful.append((gdid, path))
    return successful


def download_dataset_files(
    ftp_links: dict,
    output_dir="datasets",
    max_workers=5,  # Reduced from 10 to avoid thread issues
    batch_size=3,  # Smaller batches
    progress_callback=None,
) -> dict:
    """
    Download files using thread-based workers, each handling async downloads.
    Falls back to synchronous methods if asyncio fails.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    download_tasks = []

    # Prepare all download tasks
    for gdid, ftp_link in ftp_links.items():
        try:
            # Create a directory for this dataset
            dataset_dir = os.path.join(output_dir, f"dataset_{gdid}")
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            # Extract complete GSE ID from the FTP link
            gse_id = extract_gse_id_from_ftp(ftp_link)
            print(f"Extracted GSE ID: {gse_id} from link: {ftp_link}")

            # Paths to SOFT format files instead of MINiML
            common_files = [
                f"/soft/{gse_id}_family.soft.gz",  # Use SOFT format instead of XML
            ]

            # Create tasks for each file to download
            for file_path in common_files:
                # Construct the full URL properly
                if file_path.startswith("/"):
                    full_url = ftp_link + file_path[1:]
                else:
                    full_url = ftp_link + file_path

                filename = os.path.basename(file_path)
                local_path = os.path.join(dataset_dir, filename)

                # Add to task list (url, path, gdid)
                download_tasks.append((full_url, local_path, gdid))
        except Exception as e:
            print(f"Error processing dataset {gdid}: {str(e)}")

    print(
        f"Starting download of {len(download_tasks)} files with {max_workers} workers"
    )

    # Track successful and failed downloads
    successful_downloads = 0
    failed_downloads = 0
    download_results = {}

    # Split tasks into batches for workers
    worker_batches = []
    for i in range(0, len(download_tasks), batch_size):
        worker_batches.append(download_tasks[i : i + batch_size])

    print(f"Created {len(worker_batches)} batches of {batch_size} files each")

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create progress bar
        with tqdm(total=len(download_tasks), desc="Downloading files") as progress:
            # Submit batches to worker pool
            futures = {
                executor.submit(worker_task, batch): i
                for i, batch in enumerate(worker_batches)
            }

            # Process completed batches
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    batch_id = futures[future]

                    # Update successful downloads
                    for gdid, path in batch_results:
                        if gdid not in download_results:
                            download_results[gdid] = []
                        download_results[gdid].append(path)
                        successful_downloads += 1

                    # Update failed downloads count
                    batch_size_actual = len(worker_batches[batch_id])
                    failed_in_batch = batch_size_actual - len(batch_results)
                    failed_downloads += failed_in_batch

                    # Update progress bar
                    progress.update(batch_size_actual)
                    progress.set_postfix(
                        successful=successful_downloads, failed=failed_downloads
                    )
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    failed_downloads += len(worker_batches[futures[future]])
                    progress.update(len(worker_batches[futures[future]]))
                    progress.set_postfix(
                        successful=successful_downloads, failed=failed_downloads
                    )

    print(
        f"Download completed: {successful_downloads} successful, {failed_downloads} failed"
    )
    return download_results


def download_and_save_datasets(
    pmids_file="PMIDs_list.txt", output_dir="datasets", progress_callback=None
):
    """
    Download GEO datasets based on PMIDs.

    Args:
        pmids_file: Path to the file containing PMIDs
        output_dir: Directory to save downloaded dataset files
        progress_callback: Optional callback function for progress updates

    Returns:
        Tuple of (links_file, download_results)
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read PMIDs from file
        with open(pmids_file, "r") as f:
            pmid_lines = [line.strip() for line in f.readlines()]
            # Handle both comma-separated and line-by-line formats
            pmids = []
            for line in pmid_lines:
                for pmid in line.split(","):
                    if pmid.strip() and pmid.strip().isdigit():
                        pmids.append(pmid.strip())

        logger.info(f"Found {len(pmids)} PMIDs in {pmids_file}")
        if not pmids:
            logger.warning("No PMIDs found in the input file")
            return None, []

        # Convert PMIDs to integers for the API
        pmid_ints = [int(pmid) for pmid in pmids]

        # Step 1: Get linked dataset IDs from PMIDs
        logger.info(f"Fetching linked datasets for {len(pmids)} PMIDs")
        dataset_ids = getLinkedDatasets(pmid_ints)

        if not dataset_ids:
            logger.warning("No linked datasets found for the provided PMIDs")
            return None, []

        logger.info(f"Found {len(dataset_ids)} linked datasets")

        # Step 2: Get FTP links for datasets
        logger.info("Retrieving FTP links for datasets")
        ftp_links = get_ftp_links(dataset_ids)

        if not ftp_links:
            logger.warning("No FTP links found for the datasets")
            return None, []

        # Step 3: Save FTP links to a file for reference
        links_file = save_ftp_links_to_file(ftp_links, os.path.dirname(output_dir))

        # Step 4: Download dataset files with proper error handling
        logger.info(f"Downloading files for {len(ftp_links)} datasets")
        try:
            # Bypass asyncio if it's causing issues
            try:
                if hasattr(asyncio, "get_running_loop"):
                    # Check if there's a running loop - if so, use sync methods
                    try:
                        asyncio.get_running_loop()
                        logger.warning(
                            "Event loop already running, using synchronous downloads"
                        )
                        use_sync = True
                    except RuntimeError:
                        # No running loop, safe to use async
                        use_sync = False
                else:
                    # Older Python versions
                    use_sync = asyncio.get_event_loop().is_running()
            except Exception:
                # If any error checking the loop, default to sync
                use_sync = True

            if use_sync:
                # Use synchronous downloads
                download_results = {}
                for gdid, ftp_link in ftp_links.items():
                    result = download_dataset_sync(
                        gdid, ftp_link, output_dir, progress_callback
                    )
                    if result:
                        download_results[gdid] = result
                        # Update progress if we have a callback
                        if progress_callback and len(download_results) % 2 == 0:
                            total_files = sum(
                                len(files) for files in download_results.values()
                            )
                            progress_callback(total_files, 0, len(download_results))
            else:
                # Use the normal async/threaded approach
                download_results = download_dataset_files(
                    ftp_links, output_dir, progress_callback=progress_callback
                )
        except Exception as e:
            logger.error(f"Error in download process: {str(e)}")
            # Final fallback - simple synchronous download
            download_results = {}
            for gdid, ftp_link in ftp_links.items():
                result = download_dataset_sync(
                    gdid, ftp_link, output_dir, progress_callback
                )
                if result:
                    download_results[gdid] = result

        # Now we can safely log the results since download_results is defined
        logger.info(f"Download completed. Found {len(download_results)} datasets.")

        return links_file, download_results

    except Exception as e:
        logger.error(f"Error downloading datasets: {str(e)}")
        traceback.print_exc()
        return None, []


def save_ftp_links_to_file(ftp_links, output_dir):
    """
    Save FTP links to a file for reference.

    Args:
        ftp_links: Dictionary mapping dataset IDs to FTP links
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"dataset_ftp_links_{timestamp}.txt")

    with open(filename, "w") as f:
        for gdid, link in ftp_links.items():
            f.write(f"{gdid}\t{link}\n")

    return filename


def download_dataset_sync(gdid, ftp_link, output_dir, progress_callback=None):
    """
    Download a single dataset synchronously.
    Used as a fallback when asyncio causes thread/loop errors.
    """
    try:
        # Create a directory for this dataset
        dataset_dir = os.path.join(output_dir, f"dataset_{gdid}")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Extract GSE ID
        gse_id = extract_gse_id_from_ftp(ftp_link)
        print(f"Synchronously downloading GSE ID: {gse_id}")

        # File path for SOFT format
        file_path = f"/soft/{gse_id}_family.soft.gz"

        # Construct the full URL
        if file_path.startswith("/"):
            full_url = ftp_link + file_path[1:]
        else:
            full_url = ftp_link + file_path

        filename = os.path.basename(file_path)
        local_path = os.path.join(dataset_dir, filename)

        # Download the file
        success = download_file_simple(full_url, local_path)

        if success:
            if progress_callback:
                # Call with successful=1, failed=0, total_datasets=1
                progress_callback(1, 0, 1)
            return [local_path]
        else:
            if progress_callback:
                # Call with successful=0, failed=1, total_datasets=0
                progress_callback(0, 1, 0)
            return []

    except Exception as e:
        print(f"Error downloading dataset {gdid}: {str(e)}")
        if progress_callback:
            progress_callback(0, 1, 0)
        return []


if __name__ == "__main__":
    # This allows this file to be run directly for testing
    print("Starting dataset download process...")
    links_file, download_results = download_and_save_datasets()

    if links_file:
        print(f"Process completed successfully. Links saved to {links_file}")
        print(f"Downloaded {len(download_results)} datasets")
    else:
        print("Process failed.")
