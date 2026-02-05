import math
import random
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from pathlib import Path


def load_gene_mapping() -> dict[int, str]:
    gene_id = pd.read_csv("./data/node/gene_id.csv")
    index_to_gene = dict(zip(gene_id["index"], gene_id["gene_name"]))
    return index_to_gene


def load_go_mapping() -> dict[int, str]:
    go_id = pd.read_csv("./data/node/go_bp_id.csv")
    index_to_go = dict(zip(go_id["index"], go_id["go_id"]))
    return index_to_go


def get_drug_name(file_name: str) -> str:
    if "\\" in file_name:
        file_name = file_name.split("\\")[-1]
    return file_name.split("_", 1)[0]


def get_unique_files(folder_path: str, file_extension: str = ".csv") -> dict[str, str]:
    unique_files = {}
    for file_path in Path(folder_path).glob(f"*{file_extension}"):
        file_name = file_path.stem
        parts = file_name.split("_", 1)
        drug_name = parts[0]
        if drug_name not in unique_files:
            unique_files[drug_name] = str(file_path)
    return unique_files


def get_DEGs(drug_path: str) -> list[str]:
    df = pd.read_csv(drug_path)
    df = df[(df["log2FC"].abs() >= 0.5) & (df["FDR"] <= 0.05)]
    return df["gene_name"].tolist()


def get_FCGs(drug_path: str, num_genes: int = 500) -> list[str]:
    df = pd.read_csv(drug_path)
    df = df.sort_values(by="log2FC", key=lambda x: x.abs(), ascending=False)
    return df["gene_name"].tolist()[:num_genes]


def get_GO_Genes(drug_path: str) -> list[str]:
    df = pd.read_csv(drug_path)
    return list({gene for gene_str in df["geneID"] for gene in gene_str.split("/")})


def get_GO_IDs(drug_path: str) -> list[str]:
    df = pd.read_csv(drug_path)
    return df["ID"].tolist()


def load_ground_truth(
    drug_paths: list[str], method: str = "DEGs"
) -> dict[str, list[str]]:
    if method == "DEGs":
        get_function = get_DEGs
    elif method == "FCGs":
        get_function = get_FCGs
    elif method == "GO_Genes":
        get_function = get_GO_Genes
    elif method == "GO_IDs":
        get_function = get_GO_IDs
    else:
        raise ValueError(f"Unknown method: {method}")
    return {
        get_drug_name(drug_path): get_function(drug_path)
        for drug_path in tqdm(
            drug_paths, desc=f"load ground truth for {method}", ncols=100
        )
    }


def hr_score(ground_truth: list[str], prediction_sorted: list[str], k: int) -> float:
    """命中率分数（HR）"""
    if not ground_truth or k < 0:
        return 0.0
    if k == 0:
        k = len(ground_truth)
    top_k = prediction_sorted[:k]
    hits = sum(1 for elem in ground_truth if elem in top_k)
    return hits / len(ground_truth)


def mrr_score(ground_truth: list[str], prediction_sorted: list[str], k: int) -> float:
    """平均倒数排名分数（MRR）"""
    if not ground_truth or k < 0:
        return 0.0
    if k == 0:
        k = len(ground_truth)
    top_k = prediction_sorted[:k]
    reciprocal_ranks = []
    for elem in ground_truth:
        if elem in top_k:
            rank = top_k.index(elem) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_score(ground_truth: list[str], prediction_sorted: list[str], k: int) -> float:
    """归一化折现累积增益分数（NDCG）"""
    if not ground_truth or k < 0:
        return 0.0
    if k == 0:
        k = len(ground_truth)
    top_k = prediction_sorted[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, gene in enumerate(top_k, 1)
        if gene in ground_truth
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ground_truth), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(
    truth: dict[str, list[str]], prediction: dict[str, list[str]], method: str = "FCGs"
) -> dict[str, list[float]]:
    if method != "FCGs":
        k = 0
    else:
        k = 500
    keys = set(truth.keys()).intersection(prediction.keys())
    hr_list = []
    mrr_list = []
    ndcg_list = []
    for key in tqdm(keys, desc=f"{method} compute all metrics", ncols=100):
        hr_list.append(hr_score(truth[key], prediction[key], k=k))
        mrr_list.append(mrr_score(truth[key], prediction[key], k=k))
        ndcg_list.append(ndcg_score(truth[key], prediction[key], k=k))
    return {
        "HR": hr_list,
        "MRR": mrr_list,
        "NDCG": ndcg_list,
    }


def nodes_sorted(node_map: dict[int, str], node_score: dict[int, float]) -> list[str]:
    return [
        node_map[idx]
        for idx, _ in sorted(node_score.items(), key=lambda x: x[1], reverse=True)
    ]


def load_vnn(
    drug_path: str, node_map: dict[int, str], node_type: str = "gene"
) -> list[str]:
    with open(drug_path, "rb") as f:
        data_dict = pkl.load(f)
    node_score = data_dict[node_type]
    return nodes_sorted(node_map, node_score)


def load_binn(
    drug_path: str, node_map: dict[int, str], node_type: str = "gene"
) -> list[str]:
    with open(drug_path, "rb") as f:
        data_dict = pkl.load(f)
    node_score_npy = data_dict[node_type]
    node_score = {
        idx: abs(float(node_score_npy[idx])) for idx in range(len(node_score_npy))
    }
    return nodes_sorted(node_map, node_score)


def load_kavnn(
    drug_path: str, node_map: dict[int, str], node_type: str = "gene"
) -> list[str]:
    with open(drug_path, "rb") as f:
        data_dict = pkl.load(f)
    node_score_npy = data_dict[node_type]
    node_score = {
        idx: abs(float(node_score_npy[idx])) for idx in range(len(node_score_npy))
    }
    return nodes_sorted(node_map, node_score)


def load_random(
    drug_path: str, node_map: dict[int, str], node_type: str = "gene"
) -> list[str]:
    all_genes = list(node_map.values())
    random.shuffle(all_genes)
    return all_genes


def load_prediction(
    drug_paths: list[str], model_type: str, method: str = "DEGs"
) -> dict[str, list[str]]:
    if model_type == "VNN":
        load_predictions_fn = load_vnn
    elif model_type == "BINN":
        load_predictions_fn = load_binn
    elif model_type == "KAVNN":
        load_predictions_fn = load_kavnn
    elif model_type == "Random":
        load_predictions_fn = load_random
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    if method == "DEGs":
        node_map = load_gene_mapping()
        node_type = "gene"
    elif method == "FCGs":
        node_map = load_gene_mapping()
        node_type = "gene"
    elif method == "GO_Genes":
        node_map = load_gene_mapping()
        node_type = "gene"
    elif method == "GO_IDs":
        node_map = load_go_mapping()
        node_type = "go"
    else:
        raise ValueError(f"Unknown method: {method}")
    return {
        get_drug_name(drug_path): load_predictions_fn(drug_path, node_map, node_type)
        for drug_path in tqdm(
            drug_paths, desc=f"load {model_type} prediction for {method}", ncols=100
        )
    }


def save_results(
    drug_file_paths: dict[str, str],
    vnn_explain_paths: dict[str, str],
    binn_explain_paths: dict[str, str],
    kavnn_explain_paths: dict[str, str],
    method: str = "DEGs",
):
    deg_ground_truth = load_ground_truth(list(drug_file_paths.values()), method)
    vnn_predictions = load_prediction(list(vnn_explain_paths.values()), "VNN", method)
    vnn_results = compute_all_metrics(deg_ground_truth, vnn_predictions, method)
    binn_predictions = load_prediction(
        list(binn_explain_paths.values()), "BINN", method
    )
    binn_results = compute_all_metrics(deg_ground_truth, binn_predictions, method)
    kavnn_predictions = load_prediction(
        list(kavnn_explain_paths.values()), "KAVNN", method
    )
    kavnn_results = compute_all_metrics(deg_ground_truth, kavnn_predictions, method)
    random_predictions = load_prediction(
        list(drug_file_paths.values()), "Random", method
    )
    random_results = compute_all_metrics(deg_ground_truth, random_predictions, method)
    deg_hr = pd.DataFrame(
        {
            "VNN": vnn_results["HR"],
            "BINN": binn_results["HR"],
            "KAVNN": kavnn_results["HR"],
            "Random": random_results["HR"],
        }
    )
    deg_mrr = pd.DataFrame(
        {
            "VNN": vnn_results["MRR"],
            "BINN": binn_results["MRR"],
            "KAVNN": kavnn_results["MRR"],
            "Random": random_results["MRR"],
        }
    )
    deg_ndcg = pd.DataFrame(
        {
            "VNN": vnn_results["NDCG"],
            "BINN": binn_results["NDCG"],
            "KAVNN": kavnn_results["NDCG"],
            "Random": random_results["NDCG"],
        }
    )
    deg_hr.to_csv(f"./overlap/{method}_HR.csv", index=False)
    deg_mrr.to_csv(f"./overlap/{method}_MRR.csv", index=False)
    deg_ndcg.to_csv(f"./overlap/{method}_NDCG.csv", index=False)


if __name__ == "__main__":
    drug_file_paths = get_unique_files("./volcano")
    drug_go_paths = get_unique_files("./go")
    vnn_explain_paths = get_unique_files("./explain/VNN", ".pkl")
    binn_explain_paths = get_unique_files("./explain/BINN/SHAP", ".pkl")
    kavnn_explain_paths = get_unique_files("./explain/KAVNN/SG", ".pkl")
    # save results
    save_results(
        drug_file_paths,
        vnn_explain_paths,
        binn_explain_paths,
        kavnn_explain_paths,
        "DEGs",
    )
    save_results(
        drug_file_paths,
        vnn_explain_paths,
        binn_explain_paths,
        kavnn_explain_paths,
        "FCGs",
    )
    save_results(
        drug_go_paths,
        vnn_explain_paths,
        binn_explain_paths,
        kavnn_explain_paths,
        "GO_Genes",
    )
    save_results(
        drug_go_paths,
        vnn_explain_paths,
        binn_explain_paths,
        kavnn_explain_paths,
        "GO_IDs",
    )
