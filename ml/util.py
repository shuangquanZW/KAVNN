from typing import Tuple, Callable, Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
)


EXPR = "../data/expr.npy"
COMP = "../data/compound.npy"
LABEL = "../data/label.npy"
LIVER = "../data/edge_index/liver_index.npy"
KIDNEY = "../data/edge_index/kidney_index.npy"


def load_data(organ: str | None = None) -> Tuple[np.ndarray[Any, Any], ...]:
    if organ == "liver":
        index = np.load("../data/liver_index.npy")
        label = np.load("../data/liver_label.npy")
    elif organ == "kidney":
        index = np.load("../data/kidney_index.npy")
        label = np.load("../data/kidney_label.npy")
    else:
        index = np.load("../data/multi_index.npy")
        label = np.load("../data/multi_label.npy")
    expr = np.load(EXPR)[index]
    comp = np.load(COMP)[index]
    return expr, comp, label


def split_data_index(
    random_state: int, organ: str | None = None
) -> Tuple[list[int], list[int], list[int]]:
    _, _, label = load_data(organ)
    length = label.shape[0]
    index = np.arange(length)
    train, no_train = train_test_split(
        index,
        test_size=0.4,
        random_state=random_state,
    )
    valid, test = train_test_split(
        no_train,
        test_size=0.5,
        random_state=random_state,
    )
    return list(train), list(valid), list(test)


def preprocess_data(
    random_state: int, organ: str | None = None
) -> Tuple[np.ndarray[Any, Any], ...]:
    expr, comp, label = load_data(organ)
    features = np.concatenate((expr, comp), axis=1)
    train_index, _, test_index = split_data_index(random_state, organ)
    x_train = features[train_index]
    x_test = features[test_index]
    y_train = label[train_index]
    y_test = label[test_index]
    if organ == "liver":
        expr_pred = np.load("../data/drug_matrix_liver.npy")
        comp_pred = np.load("../data/drug_matrix_liver_compound.npy")
        label_pred = np.load("../data/drug_matrix_liver_multilabel.npy")
    elif organ == "kidney":
        expr_pred = np.load("../data/drug_matrix_kidney.npy")
        comp_pred = np.load("../data/drug_matrix_kidney_compound.npy")
        label_pred = np.load("../data/drug_matrix_kidney_multilabel.npy")
    else:
        expr_pred = None
        comp_pred = None
        label_pred = None
    x_pred = np.concatenate((expr_pred, comp_pred), axis=1)  # type: ignore
    y_pred = label_pred
    return x_train, x_test, y_train, y_test, x_pred, y_pred  # type: ignore


def is_multilabel(y: np.ndarray) -> bool:
    """检查是否为多标签问题"""
    return len(y.shape) > 1 and y.shape[1] > 1


def evaluate(
    y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], random_state: int
) -> None:
    # 检查是否为多标签问题
    multilabel = is_multilabel(y_true)
    # 计算预测的二元结果
    y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
    if multilabel:
        avg = "macro"
        metrics = {
            "Ham": hamming_loss(y_true, y_pred),
            "ACC": accuracy_score(y_true, y_pred_binary),
            "PR": precision_score(y_true, y_pred_binary, average=avg, zero_division=0),
            "RE": recall_score(y_true, y_pred_binary, average=avg, zero_division=0),
            "F1": f1_score(y_true, y_pred_binary, average=avg, zero_division=0),
        }
    else:
        metrics = {
            "AUC": roc_auc_score(y_true, y_pred),
            "ACC": accuracy_score(y_true, y_pred_binary),
            "PR": precision_score(y_true, y_pred_binary),
            "RE": recall_score(y_true, y_pred_binary),
            "F1": f1_score(y_true, y_pred_binary),
        }
    # 格式化并打印结果
    formatted_metrics = {k: f"{float(np.round(v, 4))}" for k, v in metrics.items()}
    print(f"Random State: {random_state}")
    print(", ".join(f"{k}: {v}" for k, v in formatted_metrics.items()))


def machine_learning(organ: str | None = None) -> Callable:
    if organ is None:
        ValueError("organ must be specified")

    def wrapper(define_model: Callable) -> Callable:

        def inner(*args, **kwargs) -> None:
            for random_state in [2, 12, 22, 32, 42]:
                x_train, x_test, y_train, y_test, x_pred, y_hat_label = preprocess_data(
                    random_state, organ
                )
                multilabel = is_multilabel(y_test)
                model = define_model(*args, **kwargs)
                if multilabel:
                    clf = MultiOutputClassifier(model)
                    clf.fit(x_train, y_train)
                    # y_pred = clf.predict(x_test)
                    y_hat_pred = clf.predict(x_pred)
                else:
                    y_train = y_train.ravel()
                    # y_test = y_test.ravel()
                    model.fit(x_train, y_train)
                    # y_pred = model.predict(x_test)
                    y_hat_pred = model.predict(x_pred)
                # evaluate(y_test, y_pred, random_state)  # type: ignore
                evaluate(y_hat_label, y_hat_pred, random_state)  # type: ignore

        return inner

    return wrapper
