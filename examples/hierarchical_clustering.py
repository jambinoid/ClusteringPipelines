# Usage: DATASETS_CACHE_DIR=<cache_dir> python hierarchical_clustering.py -c config_examples/hierarchical_clustering_config.yaml
import argparse
import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.cluster import hierarchy
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import silhouette_score
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


def get_metrics_per_stability_segment(
    X: np.ndarray,
    linkage_matrix: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    criterions: list[str],
    n_nodes_list: list[int]
) -> dict[tuple, list]:
    n = 0
    values_per_criterion = dict()
    for criterion in criterions:
        for n_nodes in n_nodes_list:
            populations, segments = get_stability_segments(
                linkage_matrix[:, 2], criterion, n_nodes)

            populations = populations[::-1]
            segments = segments[::-1]

            if len(populations) > n:
                n = len(populations)

            qls, qrs = tuple(zip(*segments))
            silhouette_scores = list()
            homogeneities = list()
            completenesses = list()
            v_measures = list()
            for k in populations:
                cluster_labels = hierarchy.fcluster(
                    linkage_matrix,
                    t=k,
                    criterion="maxclust_monocrit",
                    monocrit=np.arange(len(linkage_matrix), dtype="double")
                )
                homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, cluster_labels)
                silhouette_scores.append(silhouette_score(X, cluster_labels, metric=metric))
                homogeneities.append(homogeneity)
                completenesses.append(completeness)
                v_measures.append(v_measure)
            
            values_per_criterion[(f"{n_nodes} nodes {criterion}", "Number of clusters")] = populations
            values_per_criterion[(f"{n_nodes} nodes {criterion}", "Segment len")] = [qr - ql for ql, qr in zip(qls, qrs)]
            values_per_criterion[(f"{n_nodes} nodes {criterion}", "Silhouette")] = silhouette_scores
            values_per_criterion[(f"{n_nodes} nodes {criterion}", "Homogeneity")] = homogeneities
            values_per_criterion[(f"{n_nodes} nodes {criterion}", "Completeness")] = completenesses
            values_per_criterion[(f"{n_nodes} nodes {criterion}", "V-measure")] = v_measures

        for k, v in values_per_criterion.items():
            values_per_criterion[k].extend([None] * (n - len(v)))

    return values_per_criterion


def main(config: dict):
    # Read configs
    dataset_loaders_config = config["dataset_loaders"]
    encoders_config = config["encoders"]
    umap_config = config.get("umap", [None])
    hac_metrics_config = config["hac"]["metrics"]
    hac_methods_config = config["hac"]["methods"]
    hac_criterions_list = config["hac"]["criterions"]
    hac_n_nodes_list = config["hac"]["n_nodes"]

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

    # Perform clustering
    for dataset_loader in dataset_loaders:
        texts, labels = dataset_loader()
        for encoder_name, encoder in encoders.items():
            embeddings = encoder.encode(texts, batch_size=batch_size)
            for umap_params in umap_config:
                if umap_params is None:
                    X = embeddings
                else:
                    reducer = umap.UMAP(**umap_params)
                    X = reducer.fit_transform(embeddings)
                for metric in hac_metrics_config:
                    for method in hac_methods_config:
                        L = hierarchy.linkage(X, method=method, metric=metric)
                        results_table = get_metrics_per_stability_segment(
                            X=X,
                            linkage_matrix=L,
                            y_true=labels,
                            metric=metric,
                            criterions=hac_criterions_list,
                            n_nodes_list=hac_n_nodes_list
                        )

                        # Save result of clustering
                        dir = os.path.join(
                            output_dir,
                            dataset_loader.name,
                            encoder_name,
                            "no_umap" if umap_params is None
                            else "umap" + "_".join(umap_params.values()) 
                        )
                        os.makedirs(dir, exist_ok=True)
                        csv_path = os.path.join(
                            dir, f"{method}-{metric}-metrics.csv")
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