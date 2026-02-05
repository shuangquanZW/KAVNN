import argparse
import numpy as np
import pandas as pd

import torch
from pytorch_lightning import Trainer
from model import KAVNN_Embedding
from util import (
    load_data,
    create_tensor,
    load_baseline_dict,
    get_sample_index,
)


def get_train_model(args: argparse.Namespace, organ: str) -> KAVNN_Embedding.KA_VNN:
    torch.set_float32_matmul_precision("medium")  # 平衡性能和精度
    # 数据模块
    dm = KAVNN_Embedding.DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_state,
        organ=organ,
    )
    # 模型实例化
    model = KAVNN_Embedding.KA_VNN(
        num_go=args.num_go,
        num_ke=args.num_ke,
        num_neurals=2,
        grid_size=2,
        num_labels=1,
    )
    # 训练器配置
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        enable_progress_bar=True,
    )
    # 训练、测试和保存结果
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return model


def load_parser(organ: str):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="KA_VNN_Binary")
    parser.add_argument("--data_dir", type=str, default="data", help="数据保存路径")
    parser.add_argument("--num_go", type=int, default=28539, help="GO术语数量")
    parser.add_argument("--num_ke", type=int, default=1381, help="KE术语数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载器线程数")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--accelerator", type=str, default="gpu", help="加速器类型")
    parser.add_argument("--devices", type=int, default=[1], help="使用的设备数量")
    return get_train_model(parser.parse_args(), organ)


def get_embedding(model: KAVNN_Embedding.KA_VNN, drug: str) -> np.ndarray:
    expr, _, _, compound = load_data()
    expr, compound = create_tensor(expr, compound, device="cuda:0")
    baseline_dict = load_baseline_dict()
    index = get_sample_index(baseline_dict, drug)
    x = expr[index]
    c = compound[index]

    model.to("cuda:0")
    y = model.kan.get_embedding(x, *model.edge_indices, c).detach().cpu().numpy()
    return np.mean(y, axis=0)


def main():
    result = {}

    liver_drugs = []
    with open("./liver_only.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            liver_drugs.append(line)
    kidney_drugs = []
    with open("./kidney_only.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            kidney_drugs.append(line)

    model = load_parser("liver")
    for drug in liver_drugs:
        embedding = get_embedding(model, drug)
        result[drug] = embedding.tolist()

    model = load_parser("kidney")
    for drug in kidney_drugs:
        embedding = get_embedding(model, drug)
        result[drug] = embedding.tolist()

    data = []
    for drug, embedding in result.items():
        data.append([drug] + embedding)
    df = pd.DataFrame(data, columns=["drug"] + [f"dim_{i}" for i in range(32)])
    df.to_csv("./drug_embedding.csv", index=False)


if __name__ == "__main__":
    main()
