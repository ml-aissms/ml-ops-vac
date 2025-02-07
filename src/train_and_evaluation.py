import os
import argparse
import pandas as pd
import numpy as np
from pkgutil import get_data
from get_data import get_data, read_param
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import mlflow #type:ignore
from urllib.parse import urlparse


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_eval(config_path):
    config = read_param(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    raw_data_path = config["load_data"]["cleaned_data"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    df = pd.read_csv(raw_data_path,sep=",",encoding="utf-8")
    model_dir = config["model_path"]

    alpha = config["estimator"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimator"]["ElasticNet"]["params"]["l1_ratio"]

    target = config["base"]["target_col"]
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_y = train_data[target]
    test_y = test_data[target]

    train_x = train_data.drop(target, axis=1)
    test_x = test_data.drop(target, axis=1)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x,train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))

    score_files = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(score_files, "w") as f:
        scores = {
            "rmse":rmse,
            "mae":mae,
            "r2":r2
        }
        json.dump(scores,f)
    
    with open(params_file, "w") as f:
        params = {
            "alpha":alpha,
            "l1_ratio":l1_ratio
        }
        json.dump(params,f)
    
    joblib.dump(lr,model_dir)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()
    train_and_eval(config_path=parsed_args.config)