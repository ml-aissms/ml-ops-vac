import os
import argparse
import pandas as pd
from pkgutil import get_data
from get_data import get_data, read_param
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import mlflow #type:ignore
from urllib.parse import urlparse

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
    train_data = pd.read_csv("train_data_path")
    test_data = pd.read_csv("test_data_path")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()
    train_and_eval(config_path=parsed_args.config)