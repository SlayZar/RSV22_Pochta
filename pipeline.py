import pandas as pd
import numpy as np
import torch
import joblib
import os
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task
from config import (
    cat_features,
    fit_model,
    MODEL_PATH,
    TRAIN_DATAPATH,
    TEST_DATAPATH,
    ss_path,
    drop_cols,
)


def read_data(TRAIN_DATAPATH, TEST_DATAPATH, drop_cols):
    df = pd.read_csv(TRAIN_DATAPATH)
    test = pd.read_csv(TEST_DATAPATH)
    df = df[df.mailtype.isin(test.mailtype)]
    df.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)
    return df, test


def prep(test):
    test["postmark"] = test["postmark"].astype(int)
    test["oper_type"] = test["oper_type + oper_attr"].apply(
        lambda x: int(x.split("_")[0])
    )
    test["oper_attr"] = test["oper_type + oper_attr"].apply(
        lambda x: int(x.split("_")[0])
    )
    test["index_oper"] = test.index_oper.astype(str).apply(
        lambda x: x.split(".0")[0] if ".0" in x else x
    )
    test["name_mfi"] = test["name_mfi"].apply(lambda x: x.lower())
    test["postmark"] = test["postmark"].astype(int)
    test["first"] = test["name_mfi"].apply(lambda x: x.split()[0])
    test["last"] = test["name_mfi"].apply(
        lambda x: x.split()[-1] if len(x.split()) > 1 else "none"
    )
    test["len"] = test["name_mfi"].apply(lambda x: len(x))
    test["words"] = test["name_mfi"].apply(lambda x: len(x.split()))
    test["first"] = test["first"].apply(lambda x: x.split(",")[0].split("(")[0])
    test["last"] = test["last"].apply(lambda x: x.split(",")[0].split("(")[0])
    test["total_qty_oper_login_1"] = test["total_qty_oper_login_1"].replace(0, np.nan)
    for feat in ["mailctg", "class"]:
        test[feat] = test[feat].astype(int)
    for feat in ["is_in_yandex", "is_return"]:
        test[feat] = (test[feat] == "Y").astype(int)
    return test


def extra_prep(df, test, cat_features):
    df["label1"] = -1
    df.loc[(df.is_in_yandex == "0"), "label1"] = 0
    df.loc[(df.directctg == "0"), "label1"] = 0
    df.loc[(df.mailctg == "5"), "label1"] = 0
    for i in cat_features:
        df = df[df[i].isin(test[i].unique())]
    for i in cat_features:
        tmp = df.groupby([i])["label"].agg(["mean", "count"])
        tmp = tmp[(tmp["count"] > 10000) & (tmp["mean"] == 0)].index
        df.loc[df[i].isin(tmp), "label1"] = 0
        test = test[~test[i].isin(tmp)]
    return df, test


def final_prep(df, test):
    sv = (
        df.groupby(["oper_type"])["label"].agg(["count", "mean"]).sort_values(by="mean")
    )
    sv2 = (
        df.groupby(["oper_attr"])["label"].agg(["count", "mean"]).sort_values(by="mean")
    )
    df.loc[
        (
            df["oper_type"].isin(
                sv[
                    ((sv["count"] > 1000) & (sv["mean"] <= 0.0006)) | (sv["mean"] == 0)
                ].index
            )
        ),
        "label1",
    ] = 0
    df.loc[
        (
            df["oper_attr"].isin(
                sv2[
                    ((sv2["count"] > 1000) & (sv2["mean"] <= 0.0006))
                    | (sv2["mean"] == 0)
                ].index
            )
        ),
        "label1",
    ] = 0
    df = df[df.oper_type.isin(test.oper_type)]
    df = df[df.oper_attr.isin(test.oper_attr)]
    df = df[df["first"].isin(test["first"])]
    df = df[df["last"].isin(test["last"])]
    test = test[
        ~(
            test["oper_type"].isin(
                sv[((sv["count"] > 1000) & (sv["mean"] <= 0)) | (sv["mean"] == 0)].index
            )
        )
    ]
    test = test[
        ~(
            test["oper_attr"].isin(
                sv2[
                    ((sv2["count"] > 1000) & (sv2["mean"] <= 0)) | (sv2["mean"] == 0)
                ].index
            )
        )
    ]

    st1 = df["name_mfi"].value_counts().to_frame("g")
    names = st1[st1["g"] > 10].index
    df.loc[(~df.name_mfi.isin(names)), "name_mfi"] = "other"
    test.loc[(~test.name_mfi.isin(names)), "name_mfi"] = "other"
    return df, test


def all_preprocessing(df, test, cat_features):
    df = prep(df)
    test = prep(test)
    df, test = extra_prep(df, test, cat_features)
    df, test = final_prep(df, test)
    return df, test


def fit_scoring_lama_model(df, test, fit_model, MODEL_PATH):
    N_THREADS = 4  # threads cnt for lgbm and linear models
    N_FOLDS = 3  # folds cnt for AutoML
    RANDOM_STATE = 42  # fixed random state for various reasons
    TIMEOUT = 1700  # Time in seconds for automl run

    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(N_THREADS)
    task = Task("binary", loss="logloss", metric="auc", greater_is_better=True)
    roles = {"target": "label", "drop": ["id", "type", "name_mfi", "first", "last"]}
    automl2 = TabularUtilizedAutoML(
        task=task,
        timeout=TIMEOUT,
        cpu_limit=N_THREADS,
        reader_params={
            "n_jobs": N_THREADS,
            "cv": N_FOLDS,
            "random_state": RANDOM_STATE,
        },
    )
    if fit_model:
        _ = automl2.fit_predict(
            df[df.label1 == -1].drop(["label1"], axis=1), roles=roles
        )
        output2 = pd.DataFrame(
            {"id": test["id"], "label": automl2.predict(test).data[:, 0]}
        )
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
            joblib.dump(automl2, os.path.join(MODEL_PATH, 'lama_model'))
    elif os.path.exists(os.path.join(MODEL_PATH, 'lama_model')):
        automl2 = joblib.load(os.path.join(MODEL_PATH, 'lama_model'))
        output2 = pd.DataFrame(
            {"id": test["id"], "label": automl2.predict(test).data[:, 0]}
        )
    else:
        print(f"""Parameter FIT_MODEL is False
        but there is no saved model in {MODEL_PATH}""")
    return output2


def postprocessing_and_sc_result(df, test, output2, ss_path):
    ss = pd.read_csv(ss_path)
    mysols = ss.drop(["label"], axis=1).merge(
        output2.rename(columns={"label": "automl"}), on="id", how="left"
    )
    tresh = 0.1
    mysols["label"] = (mysols["automl"].fillna(0) > tresh).astype(int)
    fff = df.groupby(["first"])["label"].agg(["count", "mean"])
    good_first = fff[fff["mean"] > 0.5].sort_values(by="count").index
    mysols.loc[
        mysols["id"].isin(test[test["first"].isin(good_first)]["id"]), "label"
    ] = 1
    nums = mysols["label"].sum()
    mysols[["id", "label"]].to_csv(f"lama_{nums}_{tresh}_upd.csv", index=False)
    print(f"Predict file saved to [lama_{nums}_{tresh}_upd.csv]")


def main():
    df, test = read_data(TRAIN_DATAPATH, TEST_DATAPATH, drop_cols)
    print("Datasets readed")
    df, test = all_preprocessing(df, test, cat_features)
    print("First preprocessing readed\nFitting model")
    output2 = fit_scoring_lama_model(df, test, fit_model, MODEL_PATH)
    print("Scoring")
    postprocessing_and_sc_result(df, test, output2, ss_path)


if __name__ == "__main__":
    main()
