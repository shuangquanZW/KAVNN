import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm


def get_importance(truth: list[str], predict: list[str]) -> list[int]:
    result = [0] * len(truth)
    for i in range(len(truth)):
        if truth[i] in predict:
            result[i] = 1
    return result


if __name__ == "__main__":
    gene_id = pd.read_csv("./data/node/gene_id.csv")
    index_gene = {k: v for k, v in zip(gene_id["index"], gene_id["gene_name"])}

    drugs = []
    for item in os.listdir("./volcano"):
        item_path = os.path.join("./volcano", item)
        if os.path.isfile(item_path):
            file_name = os.path.splitext(item)[0]
            drugs.append(file_name)

    fc_values = {}
    ground_truth = {}
    for drug in tqdm(drugs, desc="处理真实差异基因", ncols=100):
        df = pd.read_csv(f"./volcano/{drug}.csv")
        df.sort_values(
            by="log2FC", key=lambda x: x.abs(), ascending=False, inplace=True
        )
        ground_truth[drug] = df["gene_name"].tolist()
        fc_values[drug] = df["log2FC"].abs().tolist()
    gene_len = len(ground_truth[list(ground_truth.keys())[0]])
    k = int(0.05 * gene_len)

    vnn = {}
    for drug in tqdm(drugs, desc="处理 VNN 解释结果", ncols=100):
        if not os.path.exists(f"./explain/VNN/{drug.split('_')[0]}_node_attribute.pkl"):
            vnn[drug] = [0] * gene_len
        else:
            with open(
                f"./explain/VNN/{drug.split("_")[0]}_node_attribute.pkl", "rb"
            ) as f:
                data_dict = pkl.load(f)
            gene_score_dict: dict[int, float] = data_dict["gene"]
            gene_score_list_tuple = sorted(
                gene_score_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            gene_score_sorted = [
                index_gene[gene_score_tuple[0]]
                for gene_score_tuple in gene_score_list_tuple
            ][:k]
            vnn[drug] = get_importance(ground_truth[drug], gene_score_sorted)

    binn = {}
    for drug in tqdm(drugs, desc="处理 BINN 解释结果", ncols=100):
        if not os.path.exists(
            f"./explain/BINN/{drug.split('_')[0]}_node_attribute.pkl"
        ):
            binn[drug] = [0] * gene_len
        else:
            with open(
                f"./explain/BINN/{drug.split('_')[0]}_node_attribute.pkl", "rb"
            ) as f:
                data_dict = pkl.load(f)
            gene_score_npy: np.ndarray = data_dict["gene"]
            gene_score_dict = {k: abs(gene_score_npy[k].item()) for k in range(15375)}
            gene_score_list_tuple = sorted(
                gene_score_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            gene_score_sorted = [
                index_gene[gene_score_tuple[0]]
                for gene_score_tuple in gene_score_list_tuple
            ][:k]
            binn[drug] = get_importance(ground_truth[drug], gene_score_sorted)

    kavnn = {}
    for drug in tqdm(drugs, desc="处理 KAVNN 解释结果", ncols=100):
        with open(
            f"./explain/KAVNN/SHAP/{drug.split("_")[0]}_{drug.split("_")[1]}_node_attribute.pkl",
            "rb",
        ) as f:
            data_dict = pkl.load(f)
        if data_dict is None:
            kavnn[drug] = [0] * gene_len
        else:
            gene_score_npy: np.ndarray = data_dict["gene"]
            gene_score_dict = {k: abs(gene_score_npy[k].item()) for k in range(15375)}
            gene_score_list_tuple = sorted(
                gene_score_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            gene_score_sorted = [
                index_gene[gene_score_tuple[0]]
                for gene_score_tuple in gene_score_list_tuple
            ][:k]
            kavnn[drug] = get_importance(ground_truth[drug], gene_score_sorted)

    random_result = [0] * gene_len
    for _ in range(k):
        random_result[random.randint(0, gene_len - 1)] = 1

    for drug in tqdm(drugs, desc="生成最终结果", ncols=100):
        df = pd.DataFrame(
            {
                "Gene": ground_truth[drug],
                "log2FC": fc_values[drug],
                "VNN": vnn[drug],
                "BINN": binn[drug],
                "KAVNN": kavnn[drug],
                "Random": random_result,
            }
        )
        df.to_csv(f"./waterfall/{drug}.csv", index=False)
