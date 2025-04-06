import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def reduce_dimensions(
    tfidf_matrix: np.ndarray, method="tsne", n_components=2
) -> np.ndarray:
    """
    Reduce dimensions of TF-IDF vectors for visualization.

    Args:
        tfidf_matrix: Matrix of TF-IDF vectors
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        n_components: Number of components for the reduced representation

    Returns:
        Reduced-dimension representation of the data
    """
    # First apply PCA for initial dimensionality reduction if the feature space is large
    if tfidf_matrix.shape[1] > 50:
        pca = PCA(n_components=min(50, tfidf_matrix.shape[0] - 1))
        reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    else:
        reduced_data = tfidf_matrix.toarray()

    # Apply the specified dimensionality reduction method
    if method == "tsne":
        tsne = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, tfidf_matrix.shape[0] // 2),
        )
        reduced_data = tsne.fit_transform(reduced_data)
        print(f"Reduced dimensions using t-SNE to {n_components} components")

    elif method == "umap":
        try:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(reduced_data)
            print(f"Reduced dimensions using UMAP to {n_components} components")
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            tsne = TSNE(n_components=n_components, random_state=42)
            reduced_data = tsne.fit_transform(reduced_data)

    elif method == "pca":
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(reduced_data)
        print(f"Reduced dimensions using PCA to {n_components} components")

    return reduced_data


def visualize_clusters(
    df: pd.DataFrame,
    reduced_data: np.ndarray,
    labels: np.ndarray,
    output_dir: str = "analysis",
) -> Tuple[str, str]:
    """
    Create visualizations of clusters and save them to files.

    Args:
        df: DataFrame with dataset information
        reduced_data: Reduced-dimension representation of the data
        labels: Cluster labels
        output_dir: Directory to save output files

    Returns:
        Tuple of paths to the saved visualization files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_dir, f"cluster_plot_{timestamp}.png")

    # Create a scatter plot of the clusters
    plt.figure(figsize=(16, 14))  # Increase figure size for better label visibility

    # Get unique cluster labels for discrete colormap
    unique_clusters = sorted(set(labels))
    n_clusters = len(unique_clusters)

    # Create discrete colormap - using compatible approach
    cmap = plt.cm.get_cmap("viridis", n_clusters)
    norm = plt.Normalize(min(unique_clusters), max(unique_clusters))

    # Create a scatter plot with different colors for each cluster
    scatter = plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=labels,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=100,  # Increase point size
    )

    # Add cluster centers if KMeans was used
    try:
        centers = reduce_dimensions(
            KMeans(n_clusters=len(set(labels)))
            .fit(df["text"].values.reshape(-1, 1))
            .cluster_centers_,
            method="tsne",
        )
        plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, alpha=0.7)
    except:
        pass

    # Create a dict to track label positions to avoid overlap
    # Key is the (x, y) rounded to 2 decimals, value is the count of labels in that area
    label_positions = {}

    # Add labels for all points with anti-overlap adjustment
    for i, txt in enumerate(df["series_id"]):
        # Base position
        x_pos = reduced_data[i, 0] + 0.02
        y_pos = reduced_data[i, 1] + 0.02

        # Round to 1 decimal place to create grid cells for detecting proximity
        pos_key = (round(x_pos, 1), round(y_pos, 1))

        # If this position already has a label, shift this one down
        if pos_key in label_positions:
            # Add vertical offset based on how many labels are already in this area
            y_offset = 0.1 * (label_positions[pos_key] + 1)
            y_pos += y_offset
            label_positions[pos_key] += 1
        else:
            label_positions[pos_key] = 0

        plt.annotate(
            txt,
            (x_pos, y_pos),
            fontsize=8,
            alpha=0.7,
            bbox=dict(facecolor="white", alpha=0.2, edgecolor="none", pad=0),
        )

    # Create a discrete colorbar with integer labels
    cbar = plt.colorbar(scatter, label="Cluster", ticks=unique_clusters)
    cbar.set_label("Cluster", size=12)

    plt.title("Dataset Clusters based on TF-IDF Vectors", fontsize=15)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)  # Add light grid
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()

    # Create a network visualization showing dataset-PMID connections
    # Use actual PubMed IDs from the data
    network_file = create_network_visualization(df, labels, output_dir, timestamp)

    return plot_file, network_file


def create_network_visualization(
    df: pd.DataFrame, labels: np.ndarray, output_dir: str, timestamp: str
) -> str:
    """
    Create a network visualization showing dataset-PMID connections using actual PubMed IDs.
    """
    network_file = os.path.join(output_dir, f"dataset_pmid_network_{timestamp}.png")

    # Create a graph
    G = nx.Graph()

    # Create a mapping of series_id to cluster label
    cluster_mapping = dict(zip(df["series_id"].values, labels))

    # Add dataset nodes
    for i, row in df.iterrows():
        series_id = row["series_id"]
        # Use series_id to get the correct cluster instead of using index
        cluster = cluster_mapping.get(series_id, 0)  # Default to cluster 0 if not found
        G.add_node(series_id, type="dataset", cluster=cluster)

        # Add PMID nodes and connections - use actual PubMed IDs from the data
        if isinstance(row["pubmed_ids"], str) and row["pubmed_ids"]:
            pmids = row["pubmed_ids"].split(",")
            for pmid in pmids:
                pmid = pmid.strip()
                if pmid:
                    pmid_node = f"PMID:{pmid}"  # Add prefix for clarity
                    G.add_node(pmid_node, type="pmid")
                    G.add_edge(series_id, pmid_node)

    # Set up the plot with larger figure size for better label readability
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)

    # Position nodes using spring layout with adjusted parameters for better separation
    pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)

    # Draw dataset nodes as circles, colored by cluster
    dataset_nodes = [
        node for node, data in G.nodes(data=True) if data.get("type") == "dataset"
    ]

    # Check if there are any dataset nodes before proceeding
    if not dataset_nodes:
        plt.text(
            0.5,
            0.5,
            "No dataset nodes found with PubMed IDs",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        plt.savefig(network_file, dpi=300)
        plt.close()
        return network_file

    dataset_colors = [G.nodes[node]["cluster"] for node in dataset_nodes]

    # Get unique cluster labels for discrete colormap
    unique_clusters = sorted(set(dataset_colors))
    n_clusters = len(unique_clusters)

    # Only proceed with colormap if we have multiple clusters
    if n_clusters > 0:
        # Use compatible approach to create a discrete colormap
        cmap = plt.cm.get_cmap("viridis", max(n_clusters, 2))  # Min 2 colors

        # Draw dataset nodes as circles
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=dataset_nodes,
            node_color=dataset_colors,
            node_size=150,  # Increase size for better visibility
            alpha=0.8,
            cmap=cmap,
            ax=ax,
            label="Datasets (GSE)",
        )
    else:
        # If there are no clusters, use a default color
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=dataset_nodes,
            node_color="blue",
            node_size=150,
            alpha=0.8,
            ax=ax,
            label="Datasets (GSE)",
        )

    # Draw PMID nodes as diamonds (rhombus)
    pmid_nodes = [
        node for node, data in G.nodes(data=True) if data.get("type") == "pmid"
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=pmid_nodes,
        node_color="red",
        node_shape="d",  # Diamond shape for PMIDs
        node_size=100,  # Slightly smaller than GSE nodes
        alpha=0.7,
        ax=ax,
        label="PMIDs",
    )

    # Draw edges with increased transparency
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.4, edge_color="gray", ax=ax)

    # Add labels to all dataset nodes
    dataset_labels = {node: node for node in dataset_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=dataset_labels,
        font_size=8,
        font_color="darkblue",
        font_weight="bold",
        ax=ax,
    )

    # Add labels to all PMID nodes, but make them smaller
    pmid_labels = {node: node.replace("PMID:", "") for node in pmid_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=pmid_labels,
        font_size=6,  # Smaller font for PMIDs
        font_color="darkred",
        alpha=0.9,
        ax=ax,
    )

    plt.title("Dataset-PMID Network Graph", fontsize=16)
    plt.legend(scatterpoints=1, loc="lower left", fontsize=10)
    plt.axis("off")

    # Add a discrete colorbar for clusters if we have multiple clusters
    if n_clusters > 1:
        # Create a mappable object for the colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_clusters - 1)
        )
        sm.set_array([])

        # Create space for the colorbar
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.3])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cax, ticks=range(n_clusters))
        cbar.set_label("Cluster", size=12)

    plt.tight_layout()
    plt.savefig(network_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Network visualization saved to {network_file}")
    return network_file
