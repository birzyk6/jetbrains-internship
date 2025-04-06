import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime
from sklearn.metrics import silhouette_score
import traceback

# Import visualization functions from the new module
from visualize import visualize_clusters, reduce_dimensions


def load_data(series_data_file: str) -> pd.DataFrame:
    """
    Load the series data with PMID information already included.
    Filter out rows where any key column is empty.

    Args:
        series_data_file: Path to the CSV file containing series data

    Returns:
        DataFrame with combined text data
    """
    # Load series data
    if not os.path.exists(series_data_file):
        raise FileNotFoundError(f"Series data file not found: {series_data_file}")

    series_df = pd.read_csv(series_data_file)
    original_count = len(series_df)
    print(f"Loaded {original_count} series records from {series_data_file}")

    # Filter out rows where important columns are empty
    key_columns = ["title", "summary", "overall_design", "types", "sample_organisms"]

    # Check each key column and remove rows where any of them are empty
    for col in key_columns:
        series_df = series_df[
            series_df[col].notna()
            & (series_df[col] != "")
            & (~series_df[col].astype(str).str.isspace())
        ]

    filtered_count = len(series_df)
    if filtered_count < original_count:
        print(
            f"Filtered out {original_count - filtered_count} records with empty fields"
        )
        print(f"Proceeding with {filtered_count} complete records")

    if filtered_count == 0:
        raise ValueError(
            "No records remain after filtering out entries with empty columns"
        )

    # Create a combined text column for TF-IDF
    series_df["text"] = series_df.apply(
        lambda row: " ".join(
            filter(
                lambda x: isinstance(x, str) and x,
                [
                    row["title"],
                    row["summary"],
                    row["overall_design"],
                    row["types"],
                    row["sample_organisms"],
                ],
            )
        ),
        axis=1,
    )

    # Clean up the text
    series_df["text"] = series_df["text"].apply(clean_text)

    return series_df


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, numbers, and extra whitespace.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove special characters except spaces
    text = re.sub(r"[^\w\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def create_tfidf_vectors(
    df: pd.DataFrame, max_features=1000, output_dir="analysis"
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF vectors from text data and save results.

    Args:
        df: DataFrame with 'text' column
        max_features: Maximum number of features for TF-IDF
        output_dir: Directory to save TF-IDF results

    Returns:
        Tuple of (TF-IDF matrix, vectorizer object)
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.9,  # Ignore terms that appear in more than 90% of documents
    )

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(df["text"].fillna(""))

    print(
        f"Created TF-IDF matrix with {tfidf_matrix.shape[0]} samples and {tfidf_matrix.shape[1]} features"
    )

    # Save TF-IDF results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save feature names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    features_file = os.path.join(output_dir, f"tfidf_features_{timestamp}.csv")
    pd.DataFrame(
        {
            "feature": vectorizer.get_feature_names_out(),
        }
    ).to_csv(features_file, index=False)

    # Save TF-IDF values
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=df["series_id"],
    )
    tfidf_values_file = os.path.join(output_dir, f"tfidf_values_{timestamp}.csv")
    tfidf_df.to_csv(tfidf_values_file)

    print(f"Saved TF-IDF features to {features_file}")
    print(f"Saved TF-IDF values to {tfidf_values_file}")

    return tfidf_matrix, vectorizer


def compute_similarities(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between TF-IDF vectors.

    Args:
        tfidf_matrix: Matrix of TF-IDF vectors

    Returns:
        Similarity matrix
    """
    similarities = cosine_similarity(tfidf_matrix)
    return similarities


def perform_clustering(
    tfidf_matrix: np.ndarray, n_clusters=5
) -> Tuple[Any, np.ndarray]:
    """
    Perform clustering on TF-IDF vectors.

    Args:
        tfidf_matrix: Matrix of TF-IDF vectors
        n_clusters: Number of clusters for KMeans

    Returns:
        Tuple of (clustering model, cluster labels)
    """
    # Try KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix.toarray())

    print(f"Performed clustering with {n_clusters} clusters")

    return kmeans, labels


def analyze_dataset(data_file: str, output_dir: str = "analysis") -> Tuple[str, str]:
    """
    Main function to analyze dataset using TF-IDF, clustering, and visualization.
    Skips analysis if the dataset doesn't have enough complete records.

    Args:
        data_file: Path to the CSV file containing series data
        output_dir: Directory to save output files

    Returns:
        Tuple of paths to the generated visualization files
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Loading data from {data_file}...")
        try:
            df = load_data(data_file)
        except ValueError as e:
            print(f"Error: {e}")
            print("Analysis cannot proceed without complete records.")
            return None, None

        # Check if we have enough data for meaningful analysis
        if len(df) < 3:
            print(
                f"Only {len(df)} complete records found. Need at least 3 records for meaningful analysis."
            )
            return None, None

        print("Creating TF-IDF vectors...")
        tfidf_matrix, vectorizer = create_tfidf_vectors(df, output_dir=output_dir)

        # Save similarities matrix
        print("Computing similarities...")
        similarities = compute_similarities(tfidf_matrix)

        # Save similarity matrix to CSV
        similarities_df = pd.DataFrame(
            similarities, index=df["series_id"], columns=df["series_id"]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        similarities_file = os.path.join(output_dir, f"similarities_{timestamp}.csv")
        similarities_df.to_csv(similarities_file)
        print(f"Saved similarities matrix to {similarities_file}")

        # Determine the optimal number of clusters
        n_clusters_range = range(2, min(10, len(df) // 2))
        silhouette_scores = []

        for n_clusters in n_clusters_range:
            # Handle the case where there are too few samples
            if n_clusters >= tfidf_matrix.shape[0]:
                break

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())

            # Skip if only one cluster was found
            if len(set(cluster_labels)) < 2:
                continue

            score = silhouette_score(tfidf_matrix.toarray(), cluster_labels)
            silhouette_scores.append(score)
            print(f"Silhouette score for {n_clusters} clusters: {score:.3f}")

        # Choose the number of clusters with the highest silhouette score
        if silhouette_scores:
            optimal_clusters = list(n_clusters_range)[
                silhouette_scores.index(max(silhouette_scores))
            ]
        else:
            optimal_clusters = min(5, len(df) // 2)

        print(f"Selected {optimal_clusters} clusters based on silhouette score")

        print("Performing clustering...")
        _, labels = perform_clustering(tfidf_matrix, n_clusters=optimal_clusters)

        print("Reducing dimensions for visualization...")
        reduced_data = reduce_dimensions(tfidf_matrix, method="tsne")

        print("Creating visualizations...")
        cluster_plot, network_plot = visualize_clusters(
            df, reduced_data, labels, output_dir
        )

        # Save clustering results to CSV
        df["cluster"] = labels
        cluster_csv_path = os.path.join(
            output_dir,
            f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        df[["series_id", "title", "pubmed_ids", "cluster"]].to_csv(
            cluster_csv_path, index=False
        )

        print(f"Clustering results saved to {cluster_csv_path}")
        print(f"Visualizations saved to {cluster_plot} and {network_plot}")

        return cluster_plot, network_plot

    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Use the dedicated analysis directory
    analysis_dir = "analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Find the most recent series data file in the output directory
    output_dir = "output"
    series_files = [f for f in os.listdir(output_dir) if f.startswith("series_data_")]

    if series_files:
        # Sort by timestamp to get the most recent
        latest_file = sorted(series_files)[-1]
        data_file = os.path.join(output_dir, latest_file)
        print(f"Found series data file: {data_file}")

        # Run the analysis
        print("Starting dataset analysis...")
        cluster_plot, network_plot = analyze_dataset(data_file, analysis_dir)

        if cluster_plot and network_plot:
            print(f"Analysis completed successfully.")
            print(f"Cluster visualization: {cluster_plot}")
            print(f"Network visualization: {network_plot}")
        else:
            print("Analysis failed.")
    else:
        print("No series data files found in the output directory.")
