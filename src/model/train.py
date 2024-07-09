# Import libraries
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Tuple
from pathlib import Path


def split_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """
    Split the data into training and test sets.

    Parameters:
    - df: pandas DataFrame
        The input DataFrame containing the data.

    Returns:
    - X_train, X_test, y_train, y_test: numpy arrays
        The training and test sets for the features (X) and target (y).
    """
    # split features from target
    X, y = df[
        ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
         'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
            ].values, df['Diabetic'].values
    # mlflow logging
    mlflow.log_metric("num_samples", X.shape[0])
    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def main(args):
    # enable autologging
    mlflow.start_run()
    mlflow.sklearn.autolog()

    # read data
    training_data_path = Path.cwd() / args.training_data
    df = get_csvs_df(path=training_data_path)

    # split data
    X_train, X_test, y_train, y_test = split_data(df=df)

    # train model
    logistic = train_model(args.reg_rate, X_train, y_train)
    # evaluate model
    evaluate(logistic, X_test, y_test)
    # stop logging
    mlflow.end_run()


def get_csvs_df(path):
    path = Path(path)
    # path = Path.cwd() / path
    if not path.exists():
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def train_model(reg_rate, X_train, y_train):
    # train model
    logistic = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(
        X_train, y_train)
    return logistic


def evaluate(logistic, X_test, y_test):
    # evaluate model
    y_hat = logistic.predict(X_test)
    acc = np.average(y_hat == y_test)
    mlflow.log_metric("accuracy", acc)
    # roc auc score
    y_scores = logistic.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_scores[:, 1])
    mlflow.log_metric("roc_auc", roc_auc)
    return


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
