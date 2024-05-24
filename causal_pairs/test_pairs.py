import sys

sys.path.append("..")
from resit import MaxNormNCDRESIT
from lingam import RESIT
import gzip, bz2, lzma, zstandard
import time
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from io import StringIO
import re


# Data Cleaning
with open("pairs/README", "r") as f:
    raw_readme = f.read().split("\n")[29:]

cols = raw_readme.pop(0)
cols = (
    ["pair"] + [c.strip() for c in cols.replace("\t", "|").split("|") if c] + ["group"]
)

group_num = 0
rows = []
for line in raw_readme:
    if line.strip() == "":
        group_num += 1
        continue
    rows.append(
        [
            c.strip()
            for c in line.replace(" " * 5, "\t").replace("\t", "|").split("|")
            if c
        ]
        + [group_num]
    )

pairs_info_df = pd.DataFrame(rows, columns=cols)

pair_np_arrs = dict()
for pair in pairs_info_df.pair:
    file = f"pairs/{pair}.txt"
    with open(file, "r") as f:
        raw_str = re.sub(r"[^\S\n]+", " ", f.read())
        buffer = StringIO(raw_str)
    pair_data = (
        pd.read_csv(buffer, sep=None, engine="python", header=None)
        .dropna(axis=1)
        .to_numpy()
    )
    pair_np_arrs[pair] = pair_data

pairs_info_df = pairs_info_df.assign(
    sample_size=lambda x: x.pair.map(lambda y: pair_np_arrs[y].shape[0]),
    num_features=lambda x: x.pair.map(lambda y: pair_np_arrs[y].shape[1]),
    weight=1.0,
    pair_num=lambda x: x.pair.str[5:].astype(int),
)

similar_pairs = [
    [49, 50, 51],
    [56, 57, 58, 59, 60, 61, 62, 63],
    [81, 82, 83],
    [89, 90],
    [97, 98],
]


for spairs in similar_pairs:
    weight = 1 / len(spairs)
    pairs_info_df.loc[pairs_info_df.pair_num.isin(spairs), "weight"] = weight


# Testing on Different Compressors

pairs = pairs_info_df.loc[pairs_info_df.num_features == 2, "pair"]


def order2arrow(order):
    return "->" if tuple(order) == (0, 1) else "<-"


def process_pair(pair, sample_size):
    arr = pair_np_arrs[pair]
    if sample_size > 0:
        sample_idx = np.random.choice(
            arr.shape[0], min(arr.shape[0], sample_size), replace=False
        )
        arr = arr[sample_idx]
    arr_norm = (arr - arr.mean(axis=0)) / arr.std(axis=0)

    seed = 2024

    model = lambda: RandomForestRegressor(random_state=69)
    model_configs = {
        "resit": lambda: RESIT(model(), random_state=seed),
        "resit_gzip": lambda: MaxNormNCDRESIT(
            model(), compressor=gzip, random_state=seed, mi_agg=np.mean
        ),
        "resit_bz2": lambda: MaxNormNCDRESIT(
            model(), compressor=bz2, random_state=seed, mi_agg=np.mean
        ),
        "resit_lzma": lambda: MaxNormNCDRESIT(
            model(), compressor=lzma, random_state=seed, mi_agg=np.mean
        ),
        "resit_zstandard": lambda: MaxNormNCDRESIT(
            model(), compressor=zstandard, random_state=seed, mi_agg=np.mean
        ),
    }

    ground_truth = pairs_info_df.loc[pairs_info_df.pair == pair, "ground truth"].item()
    random = np.random.choice(2, size=2, replace=False)

    result = {
        "pair": pair,
        "subsample_size": sample_size,
        "random_acc": order2arrow(random) == ground_truth,
    }
    for resit_model_name, resit_model in model_configs.items():
        start_time = time.time()
        causal_order = resit_model().fit(arr).causal_order_
        end_time = time.time()
        total_time = end_time - start_time

        result[f"{resit_model_name}_is_correct"] = (
            order2arrow(causal_order) == ground_truth
        )
        result[f"time_{resit_model_name}"] = total_time

        start_time = time.time()
        causal_order = resit_model().fit(arr_norm).causal_order_
        end_time = time.time()
        total_time = end_time - start_time

        result[f"{resit_model_name}_norm_is_correct"] = (
            order2arrow(causal_order) == ground_truth
        )
        result[f"time_{resit_model_name}_norm"] = total_time

    return result


sample_sizes = [50, 100, 500, -1]
param_combinations = list(itertools.product(pairs, sample_sizes))
results = Parallel(n_jobs=30)(
    delayed(process_pair)(pair, sample_size)
    for pair, sample_size in tqdm(param_combinations)
)


pairs_info_df.merge(pd.DataFrame(results), on="pair", how="left").to_pickle(
    "causal_pair_results.pickle"
)

print("Done!")
