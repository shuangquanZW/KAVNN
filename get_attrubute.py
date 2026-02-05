import os
import argparse
import numpy as np
import torch
import pickle as pkl
from model.Attribution import LogisticSurrogate
from pytorch_lightning import Trainer
from model import KAVNN_Binary


def get_train_model() -> tuple[KAVNN_Binary.KA_VNN, KAVNN_Binary.DataModule]:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="获取KAVNN模型")
    parser.add_argument("--data_dir", type=str, default="data", help="数据保存路径")
    parser.add_argument("--num_go", type=int, default=28539, help="GO术语数量")
    parser.add_argument("--num_ke", type=int, default=1381, help="KE术语数量")
    parser.add_argument(
        "--num_neurals", type=int, default=2, help="生物神经网络节点的神经元数量"
    )
    parser.add_argument("--grid_size", type=int, default=2, help="网格大小")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载器线程数")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--accelerator", type=str, default="gpu", help="加速器类型")
    parser.add_argument("--devices", type=list, default=[3], help="使用的设备数量")
    args = parser.parse_args()
    # 训练模型
    torch.set_float32_matmul_precision("medium")
    dm = KAVNN_Binary.DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_state,
        organ=None,
    )
    model = KAVNN_Binary.KA_VNN(
        num_go=args.num_go,
        num_ke=args.num_ke,
        num_neurals=args.num_neurals,
        grid_size=args.grid_size,
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return model, dm


def main():
    drugs = []
    for item in os.listdir("./volcano"):
        item_path = os.path.join("./volcano", item)
        if os.path.isfile(item_path):
            file_name = os.path.splitext(item)[0]
            drugs.append(file_name)

    model, dm = get_train_model()
    train_loader = dm.train_dataloader()

    bio_data_list = []
    chem_data_list = []
    label_list = []
    for batch in train_loader:
        bio_data, chem_data, label = batch
        bio_data = bio_data.cpu().numpy()
        chem_data = chem_data.cpu().numpy()
        label = label.cpu().numpy()
        bio_data_list.append(bio_data)
        chem_data_list.append(chem_data)
        label_list.append(label)
    bio_data = np.concatenate(bio_data_list, axis=0)
    chem_data = np.concatenate(chem_data_list, axis=0)
    label = np.concatenate(label_list, axis=0).reshape(-1)
    bio_data_tensor = torch.from_numpy(bio_data).to(model.device)
    chem_data_tensor = torch.from_numpy(chem_data).to(model.device)
    node_states = model.get_states(bio_data_tensor, chem_data_tensor)
    node_states = node_states.cpu().numpy()

    surrogate = LogisticSurrogate()
    surrogate.fit(node_states, label)

    for drug in drugs:
        node_attribute = model.compute_attribution(drug, surrogate)
        save_path = f"./explain/KAVNN/Ridge/{drug}_node_attribute.pkl"
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pkl.dump(node_attribute, f)
        print(f"Drug {drug} done!")


if __name__ == "__main__":
    main()
