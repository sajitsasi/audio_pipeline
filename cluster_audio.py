import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os


def load_processed_data(file_path):
    """Load processed audio data from JSON and extract the results list."""
    with open(file_path, "r") as f:
        data = json.load(f)
    if "results" not in data:
        raise KeyError("The JSON file does not contain the 'results' key.")
    return data["results"]


def extract_embeddings(data, embedding_type):
    """Extract embeddings from the processed data based on the embedding type."""
    embeddings = [item[embedding_type] for item in data if embedding_type in item]
    return np.array(embeddings)


def normalize_embeddings(embeddings):
    """Normalize embeddings to ensure consistent clustering results."""
    scaler = StandardScaler()
    return scaler.fit_transform(embeddings)


def determine_optimal_clusters(embeddings, max_clusters=10, embedding_name="Audio Embeddings"):
    """Determine the optimal number of clusters using the elbow method and silhouette scores."""
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    inertia = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(normalized_embeddings)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(normalized_embeddings, clusters))

    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, inertia, marker='o')
    plt.title(f'Elbow Method for Optimal Clusters using {embedding_name}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()

    # Plot Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title(f'Silhouette Scores for Optimal Clusters using {embedding_name}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.show()

    # Print optimal silhouette score and corresponding cluster count
    optimal_index = np.argmax(silhouette_scores)
    optimal_clusters = cluster_range[optimal_index]
    print(f"Optimal Cluster Count based on Silhouette Score: {optimal_clusters} with Silhouette Score: {silhouette_scores[optimal_index]:.4f}")

    return cluster_range, inertia, silhouette_scores, optimal_clusters


def cluster_embeddings(embeddings, n_clusters=3):
    """Perform KMeans clustering and return clusters and silhouette score."""
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(normalized_embeddings)
    silhouette = silhouette_score(normalized_embeddings, clusters)
    return clusters, normalized_embeddings, silhouette


def plot_clusters(embeddings, clusters, title):
    """Plot clusters in 2D using PCA for dimensionality reduction."""
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    for cluster in np.unique(clusters):
        cluster_points = reduced_embeddings[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid()
    plt.show()


def save_clustering_results(data, audio_clusters, transcription_clusters, output_path):
    """Save clustering results to a JSON file."""
    for item, audio_label, transcription_label in zip(data, audio_clusters, transcription_clusters):
        item["audio_embedding_cluster"] = int(audio_label)
        item["transcription_embedding_cluster"] = int(transcription_label)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    load_dotenv()
    # Load data
    data_path = os.getenv("PROCESSED_AUDIO_FILE")  # Processed data JSON file path
    output_path = os.getenv("CLUSTERED_AUDIO_FILE")  # Output clustered data JSON file path

    data = load_processed_data(data_path)

    # Extract embeddings and file names
    audio_embeddings = extract_embeddings(data, "audio_embedding")
    transcription_embeddings = extract_embeddings(data, "transcription_embedding")

    # Normalize embeddings
    audio_embeddings = normalize_embeddings(audio_embeddings)
    transcription_embeddings = normalize_embeddings(transcription_embeddings)

    # Determine optimal clusters
    print("Determining optimal clusters for audio embeddings...")
    _, _, _, optimal_audio_clusters = determine_optimal_clusters(audio_embeddings, max_clusters=10, embedding_name="Audio Embeddings")

    print("Determining optimal clusters for transcription embeddings...")
    _, _, _, optimal_transcription_clusters = determine_optimal_clusters(transcription_embeddings, max_clusters=10, embedding_name="Transcription Embeddings")

    # Prompt user for cluster counts
    try:
        audio_n_clusters = input(f"Enter the number of clusters for audio embeddings (or press Enter for recommended {optimal_audio_clusters}): ")
        audio_n_clusters = int(audio_n_clusters) if audio_n_clusters.strip() else optimal_audio_clusters
    except ValueError:
        print(f"Invalid input. Defaulting to recommended {optimal_audio_clusters} clusters for audio embeddings.")
        audio_n_clusters = optimal_audio_clusters

    try:
        transcription_n_clusters = input(f"Enter the number of clusters for transcription embeddings (or press Enter for recommended {optimal_transcription_clusters}): ")
        transcription_n_clusters = int(transcription_n_clusters) if transcription_n_clusters.strip() else optimal_transcription_clusters
    except ValueError:
        print(f"Invalid input. Defaulting to recommended {optimal_transcription_clusters} clusters for transcription embeddings.")
        transcription_n_clusters = optimal_transcription_clusters

    # Perform clustering
    audio_clusters, audio_normalized_embeddings, audio_silhouette = cluster_embeddings(audio_embeddings, audio_n_clusters)
    transcription_clusters, transcription_normalized_embeddings, transcription_silhouette = cluster_embeddings(transcription_embeddings, transcription_n_clusters)

    # Print silhouette scores
    print(f"Audio Embeddings - Silhouette Score: {audio_silhouette}")
    print(f"Transcription Embeddings - Silhouette Score: {transcription_silhouette}")

    # Plot clusters
    print("Plotting audio clusters...")
    plot_clusters(audio_normalized_embeddings, audio_clusters, "Audio Embeddings Clustering")

    print("Plotting transcription clusters...")
    plot_clusters(transcription_normalized_embeddings, transcription_clusters, "Transcription Embeddings Clustering")

    # Save results
    save_clustering_results(data, audio_clusters, transcription_clusters, output_path)
    print(f"Clustering results saved to {output_path}.")


if __name__ == "__main__":
    main()