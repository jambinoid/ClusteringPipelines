import argparse
import csv
import os

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import umap
import yaml

from clustering_pipelines.analysis.markov_moment import get_stability_segments
from clustering_pipelines.encoders import Encoder
from clustering_pipelines.encoders.use import USE
from clustering_pipelines.encoders.bert import SmallBERT
from clustering_pipelines.dataset_loaders import get_text_dataset_loader


def get_encoder(encoder_name: str, **kwargs) -> Encoder | SentenceTransformer:
    encoders = {
        "USE": USE,
        "SmallBERT": SmallBERT,
        "SentenceTransformer": SentenceTransformer
    }


    if encoder_name not in encoders:
        raise ValueError(
            f"Specified wrong encoder {encoder_name}, should be one of:" +
            "\n    ".dataset_loaders.keys())
    
    return encoders[encoder_name](**kwargs)


def cluster(embeddings, grid):
    max_score = -1
    for params in tqdm(grid, leave=False):
        clusterer = hdbscan.HDBSCAN(
            gen_min_span_tree=True,
            **params
        )
        clusterer.fit(embeddings)
        
        if clusterer.relative_validity_ > max_score:
            max_score = clusterer.relative_validity_
            best_params = params
            best_labels = clusterer.labels_

    print(f"Best Parameters: {best_params}")
    print(f"DBCV score: {max_score}")

    return best_labels, max_score, best_params


def main(config: dict):
    # Read configs
    dataset_loaders_config = config["dataset_loaders"]
    encoders_config = config["encoders"]
    umap_config = config.get("umap", [None])
    hdbscan_cv_params = config["hbscan_cv_params"]

    batch_size = config.get("encoder_batch_size")
    output_dir = config["output_dir"]

    # Get text dataset loader
    dataset_loaders = [
        get_text_dataset_loader(loader_name, **loader_params)
        for loader_config in dataset_loaders_config
        for loader_name, loader_params in loader_config.items()
    ]

    # Get encoders
    encoders = {
        encoder_name:get_encoder(encoder_name, **params)
        for encoder_config in encoders_config
        for encoder_name, params in encoder_config.items()
    }

    # Create parameters grid
    grid = ParameterGrid(hdbscan_cv_params)

    # Perform clustering
    for dataset_loader in dataset_loaders:
        texts, y_true = dataset_loader()
        for encoder_name, encoder in encoders.items():
            embeddings = encoder.encode(texts, batch_size=batch_size)
            for umap_params in umap_config:
                if umap_params is None:
                    X = embeddings
                else:
                    reducer = umap.UMAP(**umap_params)
                    X = reducer(embeddings)

                    labels, dbcv, params = cluster(embeddings=X, grid=grid)
                    
                    clustered = (labels >= 0)
                    noise_ratio = 1 - sum(clustered) / len(labels)

                    (
                        homogeneity,
                        completeness,
                        v_measure
                    ) = homogeneity_completeness_v_measure(
                        y_true[clustered], labels[clustered])
                    
                    results_table = {
                        "Number of clusters": len(set(labels[clustered])),
                        "DBCV": dbcv,
                        "Noise ratio": noise_ratio,
                        "Homogeneity": homogeneity,
                        "Completeness": completeness, 
                        "V-measure": v_measure,
                        **{"params."+k:v for k, v in params.items()}
                    }
                        
                    # Save result of clustering
                    dir = os.path.join(
                        output_dir,
                        dataset_loader.name,
                        encoder_name,
                        "no_umap" if umap_params is None
                        else "umap" + "_".join(umap_config.values()) 
                    )
                    os.makedirs(dir, exist_ok=True)
                    csv_path = os.path.join(dir, f"HDBSCAN-metrics.csv")
                    with open(csv_path, "w") as csv_file:
                       writer = csv.writer(csv_file)
                       writer.writerow(results_table.keys())
                       writer.writerows(zip(*results_table.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hierarchical text clustering")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the .yaml config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)