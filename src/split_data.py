import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import get_data, read_param
from sklearn.model_selection import train_test_split

def split_data(config_path):
    config = read_param(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    raw_data_path = config["load_data"]["cleaned_data"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    df = pd.read_csv(raw_data_path,sep=",",encoding="utf-8")
    train_data, test_data = train_test_split(df, test_size=split_ratio, random_state=random_state)

    train_data.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test_data.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")
                             
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args = args.parse_args()
    split_data(config_path=parsed_args.config)