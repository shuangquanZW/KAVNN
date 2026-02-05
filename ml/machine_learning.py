import warnings

warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

import lightgbm as lgb
import xgboost as xgb

from util import machine_learning  # type: ignore

organ = None


@machine_learning(organ)
def dt_classifier():
    return DecisionTreeClassifier(max_depth=5, max_features="sqrt", random_state=42)


@machine_learning(organ)
def rf_classifier():
    return RandomForestClassifier(
        max_depth=5, max_features="sqrt", n_jobs=-1, random_state=42
    )


@machine_learning(organ)
def knn_classifier():
    return KNeighborsClassifier(algorithm="kd_tree")


@machine_learning(organ)
def lr_classifier():
    return LogisticRegression(solver="saga", max_iter=100, n_jobs=-1, random_state=42)


@machine_learning(organ)
def nb_classifier():
    return BernoulliNB()


@machine_learning(organ)
def lsvc_classifier():
    return LinearSVC(max_iter=100, random_state=42)


@machine_learning(organ)
def lgb_classifier():
    return lgb.LGBMClassifier(
        n_jobs=-1, verbose=-1, max_iter=100, learning_rate=5e-3, random_state=42
    )


@machine_learning(organ)
def xgb_classifier():
    return xgb.XGBClassifier(
        n_jobs=-1, verbosity=0, max_iter=100, learning_rate=5e-3, random_state=42
    )


if __name__ == "__main__":
    # print("decision tree:")
    # dt_classifier()
    # print("random forest:")
    # rf_classifier()
    # print("k-nearest neighbors:")
    # knn_classifier()
    # print("logistic regression:")
    # lr_classifier()
    # print("naive bayes:")
    # nb_classifier()
    print("lightgbm:")
    lgb_classifier()
    print("xgboost:")
    xgb_classifier()
