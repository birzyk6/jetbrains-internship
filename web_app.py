import os
import uuid
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
import threading
from datetime import datetime
import time

# Import functions from our modules
from getdata import download_and_save_datasets
from processdata import process_datasets
from analysis import analyze_dataset

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "output"
app.config["DATASET_FOLDER"] = "datasets"
app.config["ANALYSIS_FOLDER"] = "analysis"
app.config["VISUALIZATION_FOLDER"] = "static/visualizations"

# Create necessary directories
for folder in [
    app.config["UPLOAD_FOLDER"],
    app.config["OUTPUT_FOLDER"],
    app.config["DATASET_FOLDER"],
    app.config["ANALYSIS_FOLDER"],
    app.config["VISUALIZATION_FOLDER"],
]:
    os.makedirs(folder, exist_ok=True)

# Dictionary to track job status for each session
job_status = {}


@app.route("/")
def index():
    """
    Display the home page with a form for entering PMIDs.
    """
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit_pmids():
    """
    Handle form submission with PMIDs.
    """
    if "pmids" not in request.form:
        flash("No PMIDs provided", "error")
        return redirect(url_for("index"))

    pmids = request.form["pmids"].strip()

    # Check if PMIDs are provided
    if not pmids:
        flash("Please enter at least one PMID", "error")
        return redirect(url_for("index"))

    # Generate a unique session ID for tracking this job
    session_id = str(uuid.uuid4())

    # Save PMIDs to a temporary file
    temp_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    os.makedirs(temp_dir, exist_ok=True)

    pmids_file = os.path.join(temp_dir, "PMIDs_list.txt")
    with open(pmids_file, "w") as f:
        f.write(pmids)

    # Initialize job status
    job_status[session_id] = {
        "status": "started",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Job initiated",
        "progress": 0,
        "results": None,
    }

    # Start the data processing pipeline in a background thread
    thread = threading.Thread(target=run_pipeline, args=(session_id, pmids_file))
    thread.daemon = True
    thread.start()

    # Redirect to the status page
    return redirect(url_for("job_status_page", session_id=session_id))


@app.route("/status/<session_id>")
def job_status_page(session_id):
    """
    Display the job status page.
    """
    if session_id not in job_status:
        flash("Invalid session ID or job has expired", "error")
        return redirect(url_for("index"))

    return render_template(
        "status.html", session_id=session_id, status=job_status[session_id]
    )


@app.route("/api/status/<session_id>")
def get_job_status(session_id):
    """
    API endpoint to get job status updates.
    """
    if session_id not in job_status:
        return {"error": "Invalid session ID"}, 404

    return job_status[session_id]


@app.route("/results/<session_id>")
def show_results(session_id):
    """
    Display the results page with visualizations.
    """
    if session_id not in job_status:
        flash("Invalid session ID or job has expired", "error")
        return redirect(url_for("index"))

    job_data = job_status[session_id]

    if job_data["status"] != "completed":
        return redirect(url_for("job_status_page", session_id=session_id))

    results = job_data["results"]

    # Check if we have valid results
    if results is None or not all(results):
        flash("No valid results were generated", "error")
        return redirect(url_for("index"))

    # Extract the file paths - results now has just filenames, not full paths
    links_file, csv_file, viz_filenames = results

    # Pass just the filenames directly to the template
    return render_template(
        "results.html",
        session_id="",  # No need to prefix in template
        csv_file=os.path.basename(csv_file),
        cluster_viz=viz_filenames[0],  # Already just the filename
        network_viz=viz_filenames[1],  # Already just the filename
    )


@app.route("/visualizations/<path:filename>")
def serve_visualization(filename):
    """
    Serve visualization files.
    """
    return send_from_directory(app.config["VISUALIZATION_FOLDER"], filename)


@app.route("/debug/visualizations")
def list_visualizations():
    """
    Debug route to list all visualization files available in the visualization folder.
    """
    try:
        files = os.listdir(app.config["VISUALIZATION_FOLDER"])
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}, 500


def update_job_status(session_id, status, message, progress, results=None, stats=None):
    """
    Update the status of a job.

    Args:
        session_id: Unique identifier for the job
        status: Status string (started, validating, downloading, etc.)
        message: Status message to display
        progress: Progress percentage (0-100)
        results: Optional results tuple
        stats: Optional dictionary with additional statistics
    """
    if session_id in job_status:
        job_status[session_id].update(
            {
                "status": status,
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        if stats is not None:
            if "stats" not in job_status[session_id]:
                job_status[session_id]["stats"] = {}
            job_status[session_id]["stats"].update(stats)

        if results is not None:
            job_status[session_id]["results"] = results


def run_pipeline(session_id, pmids_file):
    """
    Run the data processing pipeline and update job status.
    """
    try:
        # Make session-specific directories
        session_dataset_dir = os.path.join(app.config["DATASET_FOLDER"], session_id)
        session_output_dir = os.path.join(app.config["OUTPUT_FOLDER"], session_id)
        session_analysis_dir = os.path.join(app.config["ANALYSIS_FOLDER"], session_id)

        for directory in [
            session_dataset_dir,
            session_output_dir,
            session_analysis_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Parse and validate PMIDs before downloading
        update_job_status(session_id, "validating", "Validating PubMed IDs...", 10)

        # Read and clean PMIDs from file
        with open(pmids_file, "r") as f:
            content = f.read().strip()

        # Process PMIDs - handle both newline and comma-separated formats
        pmids = []
        for item in content.replace(",", "\n").split("\n"):
            item = item.strip()
            if item and item.isdigit():  # Ensure it's a valid PMID (numbers only)
                pmids.append(item)

        # Check if we have valid PMIDs after cleaning
        if not pmids:
            update_job_status(
                session_id,
                "failed",
                "No valid PMIDs found. Please enter numeric PubMed IDs.",
                10,
                None,
            )
            return

        # Write cleaned PMIDs back to file
        with open(pmids_file, "w") as f:
            f.write("\n".join(pmids))

        # Log the PMIDs being processed
        print(
            f"Processing {len(pmids)} PMIDs: {', '.join(pmids[:10])}{' ...' if len(pmids) > 10 else ''}"
        )

        # Step 1: Download datasets with more detailed error handling and progress tracking
        update_job_status(
            session_id,
            "downloading",
            f"Downloading datasets for {len(pmids)} PMIDs...",
            25,
            stats={
                "total_pmids": len(pmids),
                "downloaded_files": 0,
                "failed_files": 0,
                "found_datasets": 0,
            },
        )

        # Create a custom download callback to update progress - modified to round percentage
        def download_progress_callback(successful, failed, total_datasets):
            # Calculate progress and round to 2 decimal places
            progress_value = min(
                25 + (successful / max(1, successful + failed) * 25), 49
            )
            progress_value = round(progress_value, 2)

            update_job_status(
                session_id,
                "downloading",
                f"Downloaded {successful} files from {total_datasets} datasets",
                progress_value,
                stats={
                    "downloaded_files": successful,
                    # We're still tracking failed files internally but not displaying them
                    "failed_files": failed,
                    "found_datasets": total_datasets,
                },
            )

        try:
            links_file, download_results = download_and_save_datasets(
                pmids_file=pmids_file,
                output_dir=session_dataset_dir,
                progress_callback=download_progress_callback,
            )

            # Final download stats update - modified message to remove mention of failures
            if download_results:
                total_files = sum(len(files) for files in download_results.values())
                update_job_status(
                    session_id,
                    "processing",
                    f"Completed downloads: {total_files} files across {len(download_results)} datasets",
                    50,
                    stats={
                        "downloaded_files": total_files,
                        "found_datasets": len(download_results),
                    },
                )

            print(f"Download results: {download_results}")

            if not download_results:
                update_job_status(
                    session_id,
                    "failed",
                    "No datasets were found for the provided PMIDs. Please try different PubMed IDs.",
                    25,
                    None,
                )
                return
        except Exception as e:
            error_msg = f"Error during download: {str(e)}"
            print(error_msg)
            import traceback

            print(traceback.format_exc())
            update_job_status(session_id, "failed", error_msg, 25, None)
            return

        # Step 2: Process datasets
        update_job_status(session_id, "processing", "Processing datasets...", 50)
        csv_file = process_datasets(
            dataset_dir=session_dataset_dir, output_dir=session_output_dir
        )

        if not csv_file:
            update_job_status(
                session_id, "failed", "Failed to process datasets", 50, None
            )
            return

        # Step 3: Analyze datasets
        update_job_status(session_id, "analyzing", "Analyzing datasets...", 75)
        visualization_files = analyze_dataset(
            data_file=csv_file, output_dir=session_analysis_dir
        )

        if not visualization_files or not all(visualization_files):
            update_job_status(
                session_id, "failed", "Failed to create visualizations", 75, None
            )
            return

        # Copy visualization files to static directory for web access with simplified names
        import shutil

        viz_paths = []
        for i, viz_file in enumerate(visualization_files):
            # Use simpler filenames to avoid path issues
            viz_type = "cluster" if i == 0 else "network"
            dest_filename = f"{session_id}_{viz_type}.png"
            dest_file = os.path.join(app.config["VISUALIZATION_FOLDER"], dest_filename)
            shutil.copy(viz_file, dest_file)
            viz_paths.append(dest_filename)  # Store just the filename without path
            print(f"Copied visualization to {dest_file}")

        # Update job status with results
        update_job_status(
            session_id,
            "completed",
            "Analysis completed successfully",
            100,
            (links_file, csv_file, viz_paths),
        )

    except Exception as e:
        update_job_status(session_id, "failed", f"Error: {str(e)}", 0, None)
        import traceback

        print(traceback.format_exc())


# Clean up old job status entries periodically
def clean_old_jobs():
    """
    Clean up job status entries that are older than 24 hours.
    """
    while True:
        now = datetime.now()
        to_delete = []

        for session_id, data in job_status.items():
            job_time = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (now - job_time).total_seconds() > 86400:  # 24 hours
                to_delete.append(session_id)

        for session_id in to_delete:
            del job_status[session_id]

        time.sleep(3600)  # Run every hour


if __name__ == "__main__":
    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=clean_old_jobs)
    cleanup_thread.daemon = True
    cleanup_thread.start()

    # Start the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)
