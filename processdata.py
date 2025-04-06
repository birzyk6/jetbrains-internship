import os
import gzip
import pandas as pd
import re
import glob
from tqdm import tqdm
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional


def extract_archive(archive_path: str, extract_dir: Optional[str] = None) -> str:
    """
    Extract a gzip archive to a directory.

    Args:
        archive_path: Path to the .gz file
        extract_dir: Directory to extract to. If None, extracts to the same directory as the archive.

    Returns:
        Path to the extracted file
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # If no extract directory specified, use the archive's directory
    if extract_dir is None:
        extract_dir = os.path.dirname(archive_path)

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    try:
        # Get the output filename (remove .gz extension)
        output_file = os.path.join(extract_dir, os.path.basename(archive_path)[:-3])

        # Extract .gz file
        with gzip.open(archive_path, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                f_out.write(f_in.read())

        print(f"Successfully extracted {archive_path} to {output_file}")
        return output_file

    except Exception as e:
        print(f"Error extracting {archive_path}: {str(e)}")
        return ""


def find_soft_files(directory: str) -> List[str]:
    """
    Find all SOFT files in a directory and its subdirectories.

    Args:
        directory: Directory to search

    Returns:
        List of paths to SOFT files
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []

    soft_files = []

    # Look for SOFT files
    patterns = [
        os.path.join(directory, "**", "*.soft"),
        os.path.join(directory, "**", "*.SOFT"),
        os.path.join(directory, "**", "*_family.soft"),
    ]

    for pattern in patterns:
        found_files = glob.glob(pattern, recursive=True)
        soft_files.extend(found_files)

    # Remove duplicates
    soft_files = list(set(soft_files))

    if not soft_files:
        # Debug: list directory contents if no SOFT files found
        print(f"No SOFT files found in {directory}. Directory contents:")
        for root, dirs, files in os.walk(directory):
            for file in files:
                print(f"  {os.path.join(root, file)}")

    return soft_files


def parse_geo_soft(soft_file: str) -> Dict[str, Any]:
    """
    Parse a GEO SOFT file and extract series information including PMIDs.
    """
    try:
        # Initialize data dictionary
        data = {
            "series_id": "",
            "title": "",
            "summary": "",
            "overall_design": "",
            "types": [],  # Changed to list to store multiple types
            "sample_organisms": [],
            "pubmed_ids": [],  # Added to store PubMed IDs
        }

        # Extract filename for reference
        filename = os.path.basename(soft_file)

        # Extract GSE ID from filename if possible
        match = re.search(r"(GSE\d+)", filename)
        if match:
            data["series_id"] = match.group(1)

        # Flag to track if we've found sample organisms
        found_organism = False

        # Open and read the SOFT file
        current_section = None
        with open(soft_file, "r", encoding="utf-8", errors="replace") as f:
            # First pass - look for Series section and extract key fields
            for line in f:
                line = line.strip()

                # Check if we're in the ^SERIES section
                if line.startswith("^SERIES"):
                    current_section = "SERIES"
                    # Try to extract series ID if not already found
                    if not data["series_id"]:
                        parts = line.split(" = ")
                        if len(parts) > 1:
                            data["series_id"] = parts[1]
                    continue

                # If we found another section, stop processing if we've already processed SERIES
                elif line.startswith("^") and current_section == "SERIES":
                    current_section = None

                # Process series fields
                if current_section == "SERIES":
                    if line.startswith("!Series_title"):
                        parts = line.split(" = ", 1)
                        if len(parts) > 1:
                            data["title"] = parts[1]

                    elif line.startswith("!Series_summary"):
                        parts = line.split(" = ", 1)
                        if len(parts) > 1:
                            data["summary"] = parts[1]

                    elif line.startswith("!Series_overall_design"):
                        parts = line.split(" = ", 1)
                        if len(parts) > 1:
                            data["overall_design"] = parts[1]

                    elif line.startswith("!Series_type"):
                        parts = line.split(" = ", 1)
                        if len(parts) > 1:
                            data["types"].append(parts[1])

                    # Handle all variations of the organism field name
                    elif (
                        line.startswith("!Series_sample_organism")
                        or line.startswith("!Series_organism")
                        or line.startswith("!Series_organism_ch")
                        or "organism" in line.lower()
                        and line.startswith("!Series_")
                    ):
                        parts = line.split(" = ", 1)
                        if len(parts) > 1 and parts[1].strip():
                            data["sample_organisms"].append(parts[1].strip())
                            found_organism = True
                            print(f"Found organism: {parts[1].strip()}")

                    # Extract PubMed IDs
                    elif line.startswith("!Series_pubmed_id"):
                        parts = line.split(" = ", 1)
                        if len(parts) > 1 and parts[1].strip():
                            data["pubmed_ids"].append(parts[1].strip())

        # If we didn't find organism in Series section, look in Sample sections
        if not found_organism:
            print("No organisms found in Series section, checking Sample sections...")
            # Reopen file and check Sample sections
            with open(soft_file, "r", encoding="utf-8", errors="replace") as f:
                current_section = None
                for line in f:
                    line = line.strip()

                    # Check for Sample section
                    if line.startswith("^SAMPLE"):
                        current_section = "SAMPLE"
                        continue
                    elif line.startswith("^") and current_section == "SAMPLE":
                        current_section = None

                    # Look for organism in Sample section
                    if current_section == "SAMPLE" and (
                        line.startswith("!Sample_organism")
                        or line.startswith("!Sample_organism_ch")
                    ):
                        parts = line.split(" = ", 1)
                        if len(parts) > 1 and parts[1].strip():
                            # Add to sample_organisms if not already there
                            organism = parts[1].strip()
                            if organism not in data["sample_organisms"]:
                                data["sample_organisms"].append(organism)
                                print(f"Found organism in Sample section: {organism}")

        # Print summary of what we found
        print(f"Parsed SOFT file: {filename}")
        print(f"  Series ID: {data['series_id']}")
        print(
            f"  Title: {data['title'][:50]}..."
            if len(data["title"]) > 50
            else f"  Title: {data['title']}"
        )
        print(f"  Types found: {len(data['types'])}")
        print(f"  Organisms found: {len(data['sample_organisms'])}")
        print(f"  PubMed IDs found: {len(data['pubmed_ids'])}")
        if data["pubmed_ids"]:
            print(f"  PubMed IDs: {data['pubmed_ids']}")

        return data

    except Exception as e:
        print(f"Error processing SOFT file {soft_file}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {}


def process_dataset_archives(dataset_dir: str) -> dict:
    """
    Process all SOFT archives in the dataset directory.

    Args:
        dataset_dir: Directory containing dataset directories

    Returns:
        Dictionary mapping file IDs to their extracted data
    """
    # Find all .soft.gz files
    archive_pattern = os.path.join(dataset_dir, "**", "*.soft.gz")
    archives = glob.glob(archive_pattern, recursive=True)

    if not archives:
        print(f"No SOFT archives found in {dataset_dir}")
        return {}

    print(f"Found {len(archives)} SOFT archives to process")

    # Extract archives first
    extracted_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}

        # Submit extraction tasks
        for archive in archives:
            extract_dir = os.path.dirname(archive)  # Extract to the same directory
            futures[executor.submit(extract_archive, archive, extract_dir)] = archive

        # Process results
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Extracting archives",
        ):
            archive = futures[future]
            try:
                extracted_file = future.result()
                if extracted_file:
                    extracted_files.append(extracted_file)
            except Exception as e:
                print(f"Error extracting {archive}: {str(e)}")

    # Find additional SOFT files that might already be extracted
    more_soft_files = find_soft_files(dataset_dir)
    for soft_file in more_soft_files:
        if soft_file not in extracted_files:
            extracted_files.append(soft_file)

    print(f"Found {len(extracted_files)} SOFT files to process")

    if not extracted_files:
        print("No SOFT files found.")
        return {}

    # Process all SOFT files
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {}

        for soft_file in extracted_files:
            future = executor.submit(parse_geo_soft, soft_file)
            future_to_file[future] = soft_file

        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=len(future_to_file),
            desc="Parsing SOFT files",
        ):
            soft_file = future_to_file[future]
            try:
                data = future.result()
                if data and data.get("series_id"):
                    results[data["series_id"]] = data
            except Exception as e:
                print(f"Error processing {soft_file}: {str(e)}")

    print(f"Successfully extracted data from {len(results)} SOFT files")
    return results


def save_dataset_data(results_dict, output_dir="output"):
    """
    Save the extracted series data to CSV files.

    Args:
        results_dict: Dictionary mapping series IDs to data
        output_dir: Directory to save output files

    Returns:
        Path to the summary CSV file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Convert results to a DataFrame
    series_data = []
    for series_id, data in results_dict.items():
        row = {
            "series_id": series_id,
            "title": data.get("title", ""),
            "summary": data.get("summary", ""),
            "overall_design": data.get("overall_design", ""),
            "types": ", ".join(data.get("types", [])),  # Join multiple types
            "sample_organisms": ", ".join(data.get("sample_organisms", [])),
            "pubmed_ids": ", ".join(data.get("pubmed_ids", [])),  # Added PubMed IDs
        }
        series_data.append(row)

    if not series_data:
        print("No series data to save.")
        return ""

    # Create Series DataFrame
    series_df = pd.DataFrame(series_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"series_data_{timestamp}.csv")
    series_df.to_csv(csv_path, index=False)
    print(f"Saved data for {len(series_df)} series to {csv_path}")

    # Try to save to Excel format as well
    try:
        excel_path = os.path.join(output_dir, f"series_data_{timestamp}.xlsx")
        series_df.to_excel(excel_path, index=False)
        print(f"Saved data to Excel: {excel_path}")
    except Exception as e:
        print(f"Could not save to Excel: {str(e)}")

    # Create a summary file
    summary_data = [
        {"field": "Series ID", "count": len(series_df)},
        {"field": "Title", "count": series_df["title"].notna().sum()},
        {"field": "Summary", "count": series_df["summary"].notna().sum()},
        {"field": "Overall Design", "count": series_df["overall_design"].notna().sum()},
        {
            "field": "Types",
            "count": series_df["types"].str.len().gt(0).sum(),
        },  # Changed field name
        {
            "field": "Sample Organisms",
            "count": series_df["sample_organisms"].str.len().gt(0).sum(),
        },
        {
            "field": "PubMed IDs",
            "count": series_df["pubmed_ids"].str.len().gt(0).sum(),
        },  # Added PubMed IDs
    ]

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"series_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved summary to {summary_path}")
    return csv_path


def process_datasets(dataset_dir: str = "datasets", output_dir: str = "output") -> str:
    """
    Process all datasets and extract series information from SOFT files.

    Args:
        dataset_dir: Directory containing the downloaded dataset files
        output_dir: Directory to save the output files

    Returns:
        Path to the CSV file
    """
    try:
        print(f"Processing datasets from {dataset_dir}...")

        # Process SOFT archives and extract series data
        results_dict = process_dataset_archives(dataset_dir)

        if not results_dict:
            print("No series information found in SOFT files.")
            return ""

        print(f"Successfully extracted data for {len(results_dict)} series.")

        # Save data to CSV files
        csv_path = save_dataset_data(results_dict, output_dir)

        return csv_path

    except Exception as e:
        print(f"Error processing datasets: {str(e)}")
        import traceback

        traceback.print_exc()
        return ""


if __name__ == "__main__":
    # This allows this file to be run directly for testing
    print("Starting dataset processing...")
    result = process_datasets()

    if result:
        print(f"Processing completed successfully. Data saved to {result}")
    else:
        print("Processing failed.")
