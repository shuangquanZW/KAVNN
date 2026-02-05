import argparse

import torch
from pytorch_lightning import Trainer
from dl.VNN import VNN


def train_vnn(args: argparse.Namespace, organ: str | None = None):
    torch.set_float32_matmul_precision("medium")  # 平衡性能和精度

    # 数据模块
    dm = VNN.VNNDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_state,
        organ=organ,
    )
    # 模型实例化
    model = VNN.VNN(
        num_gene_go=args.num_gene_go,
        num_go=args.num_go,
        num_go_ke=args.num_go_ke,
        num_ke=args.num_ke,
        num_ke_ke=args.num_ke_ke,
        num_neurals=args.num_neurals,
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
    test_results = trainer.test(model, datamodule=dm)[0]
    predictions = trainer.predict(model, datamodule=dm)
    all_predictions = torch.cat(predictions, dim=0).cpu()  # type: ignore
    return test_results, all_predictions


def load_parser(organ: str | None = None, random_state: int = 42):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="VNN")
    parser.add_argument("--log_path", type=str, default="tb_logs", help="日志保存路径")
    parser.add_argument("--log_name", type=str, default="BINN", help="日志名称")
    parser.add_argument("--data_dir", type=str, default="data", help="数据保存路径")
    parser.add_argument("--num_gene_go", type=int, default=25335, help="基因-GO数量")
    parser.add_argument("--num_go", type=int, default=28539, help="GO术语数量")
    parser.add_argument("--num_go_ke", type=int, default=29959, help="GO术语-KE数量")
    parser.add_argument("--num_ke", type=int, default=1381, help="KE术语数量")
    parser.add_argument("--num_ke_ke", type=int, default=1769, help="KE术语-KE数量")
    parser.add_argument("--num_neurals", type=int, default=8, help="神经网络层数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载器线程数")
    parser.add_argument("--max_epochs", type=int, default=50, help="最大训练轮数")
    parser.add_argument(
        "--random_state", type=int, default=random_state, help="随机种子"
    )
    parser.add_argument("--accelerator", type=str, default="gpu", help="加速器类型")
    parser.add_argument("--devices", type=int, default=1, help="使用的设备数量")
    return train_vnn(parser.parse_args(), organ)


def main(organ: str | None = None, random_state: int = 42):
    _, predict_results = load_parser(organ, random_state)
    torch.save(predict_results, f"./binary/pt/{organ}/vnn/{random_state}.pt")


if __name__ == "__main__":
    for random_state in [2, 12, 22, 32, 42]:
        main("liver", random_state=random_state)
        main("kidney", random_state=random_state)
