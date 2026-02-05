import os
import numpy as np
import pandas as pd
import pickle as pkl


def get_csv(drug_name: str):
    with open(f"explain/KAVNN/SG/{drug_name}_node_attribute.pkl", "rb") as f:
        data = pkl.load(f)
    gene_attribute = []
    go_attribute = []
    ke_attribute = []
    for index, value in data["gene"].items():
        attr = np.abs(value)
        gene_attribute.append((index, attr))
    for index, value in data["go"].items():
        attr = np.abs(value)
        go_attribute.append((index, attr))
    for index, value in data["ke"].items():
        attr = np.abs(value)
        ke_attribute.append((index, attr))

    gene_id_df = pd.read_csv("./data/node/gene_id.csv")
    gene_id_key = gene_id_df["index"].tolist()
    gene_id_value = gene_id_df["gene_name"].tolist()
    gene_id_dict = dict(zip(gene_id_key, gene_id_value))
    gene_attribute = {gene_id_dict[i[0]]: i[1] for i in gene_attribute}

    go_id_df = pd.read_csv("./data/node/go_bp_id.csv")
    go_id_key = go_id_df["index"].tolist()
    go_id_value = go_id_df["go_id"].tolist()
    go_id_dict = dict(zip(go_id_key, go_id_value))
    go_attribute = {go_id_dict[i[0]]: i[1] for i in go_attribute}

    ke_id_df = pd.read_csv("./data/node/ke_id.csv")
    ke_id_key = ke_id_df["index"].tolist()
    ke_id_value = ke_id_df["ke_name"].tolist()
    ke_id_dict = dict(zip(ke_id_key, ke_id_value))
    mie_list = np.load("./data/edge_index/mie.npy").tolist()
    cell_list = np.load("./data/edge_index/cell.npy").tolist()
    tissue_list = np.load("./data/edge_index/tissue.npy").tolist()
    mie_attribute = {ke_id_dict[i[0]]: i[1] for i in ke_attribute if i[0] in mie_list}
    cell_attribute = {ke_id_dict[i[0]]: i[1] for i in ke_attribute if i[0] in cell_list}
    tissue_attribute = {
        ke_id_dict[i[0]]: i[1] for i in ke_attribute if i[0] in tissue_list
    }

    with open(f"data/pkl/paths.pkl", "rb") as f:
        paths = pkl.load(f)

    tsv_data = []
    for path in paths:
        gene, go, mie, cell, tissue = path
        gene = gene_id_dict[gene]
        go = go_id_dict[go]
        mie = ke_id_dict[mie]
        cell = ke_id_dict[cell]
        tissue = ke_id_dict[tissue]
        attribute_gene = gene_attribute[gene]
        attribute_go = go_attribute.get(go, -1)
        attribute_mie = mie_attribute[mie]
        attribute_cell = cell_attribute[cell]
        attribute_tissue = tissue_attribute[tissue]
        value = (
            attribute_gene
            + attribute_go
            + attribute_mie
            + attribute_cell
            + attribute_tissue
        )
        if attribute_go == -1:
            continue
        tsv_data.append((gene, go, mie, cell, tissue, value))
    tsv_data = list(set(tsv_data))

    df = pd.DataFrame(
        sorted(tsv_data, key=lambda x: x[-1], reverse=True),
        columns=[
            "gene",
            "go",
            "mie",
            "cell",
            "tissue",
            "value",
        ],
    )
    df.to_csv(f"./sankey/{drug_name}.csv", index=False)


if __name__ == "__main__":
    drugs = []
    for item in os.listdir("./volcano"):
        item_path = os.path.join("./volcano", item)
        if os.path.isfile(item_path):
            file_name = os.path.splitext(item)[0]
            drugs.append(file_name)

    for drug in drugs:
        get_csv(drug)
