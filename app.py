import os
import sys
import time
from datetime import datetime

# Import functions from our modules
from getdata import download_and_save_datasets
from processdata import process_datasets
from analysis import analyze_dataset


def run_pipeline(
    pmids_file="PMIDs_list.txt",
    output_dir="output",
    dataset_dir="datasets",
    analysis_dir="analysis",
):
    """
    Run the complete data pipeline:
    1. Download dataset files based on PMIDs
    2. Process the downloaded files into a structured dataset
    3. Analyze the dataset using TF-IDF and clustering

    Args:
        pmids_file: Path to the file containing PMIDs
        output_dir: Directory to save output files
        dataset_dir: Directory to save downloaded dataset files
        analysis_dir: Directory to save analysis output files

    Returns:
        Tuple of (links_file, csv_file, visualization_files) with paths to the generated files
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print(f"Starting data pipeline at {timestamp}")
    print("=" * 80)

    # Create the analysis directory if it doesn't exist
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Step 1: Download datasets
    print("\n[STEP 1] DOWNLOADING DATASETS")
    print("-" * 80)
    links_file, download_results = download_and_save_datasets(
        pmids_file=pmids_file, output_dir=dataset_dir
    )

    if not download_results:
        print("No datasets were downloaded. Pipeline cannot continue.")
        return links_file, None

    # Step 2: Process datasets
    print("\n[STEP 2] PROCESSING DATASETS")
    print("-" * 80)
    csv_file = process_datasets(dataset_dir=dataset_dir, output_dir=output_dir)

    # Step 3: Analyze datasets
    print("\n[STEP 3] ANALYZING DATASETS")
    print("-" * 80)
    visualization_files = None
    if csv_file:
        visualization_files = analyze_dataset(csv_file, analysis_dir)
    else:
        print("No dataset file available for analysis.")

    # Calculate execution time
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print(f"Pipeline completed in {duration:.2f} seconds")
    print(f"- Links file: {links_file}")
    print(f"- Data file: {csv_file}")
    if visualization_files:
        print(f"- Cluster visualization: {visualization_files[0]}")
        print(f"- Network visualization: {visualization_files[1]}")
    print("=" * 80)

    return links_file, csv_file, visualization_files


if __name__ == "__main__":
    # Check if PMIDs file exists
    pmids_file = "PMIDs_list.txt"
    if not os.path.exists(pmids_file):
        print(
            f"Error: {pmids_file} not found. Please create this file with a list of PMIDs."
        )
        sys.exit(1)

    # Setup output directories
    output_dir = "output"
    dataset_dir = "datasets"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Setup analysis directory
    analysis_dir = "analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Run the pipeline
    links_file, csv_file, visualization_files = run_pipeline(
        pmids_file=pmids_file,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        analysis_dir=analysis_dir,
    )

    # Provide final status
    if csv_file:
        print(f"Process completed successfully. Data saved to {csv_file}")
        if visualization_files:
            print(f"Analysis saved to {analysis_dir}")
    else:
        print("Process failed or no data was processed.")
