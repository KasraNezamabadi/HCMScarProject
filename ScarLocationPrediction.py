import statistics
import random

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

import GlobalPaths
from QTSegmentExtractor import QTSegmentExtractor
from fQRS import get_ecg_scar_dataset, get_ecg_feature_dataset

param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "gamma": [0, 0.25, 1, 3],
    "reg_lambda": [0, 1, 10, 30],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.5, 0.8, 1],
    "colsample_bytree": [0.3, 0.5, 0.8, 1],
}

# param_grid = {
#     "n_estimators": [50, 75, 100, 150, 200],
#     "criterion": ['gini', 'entropy', 'log_loss'],
#     "class_weight": [None, 'balanced', 'balanced_subsample'],
#     "max_depth": [None, 3, 5, 7, 10],
#     "n_jobs": [-1],
# }


def predict_scar_grid_search_tree(region_name: str, select_top_features: bool = False):
    dataset, continuous_columns, discrete_columns = get_ecg_scar_dataset(region_name=region_name, select_top_features=select_top_features)
    # Check if `dataset` is well organized.
    if '_' in dataset.columns.values[-1]:
        raise AssertionError(f'"{dataset.columns.values[-1]}" is not a valid region name')
    if dataset.columns.values[-2] != 'Record_ID':
        raise AssertionError('Column -2 must be "Record_ID"')

    acc_runs, f1_runs, auc_runs = [], [], []
    for run in range(10):
        print(f'\n--- Run {run+1} ---')
        train_x, test_x, train_y, test_y = train_test_split(dataset.iloc[:, 0:-2].values,
                                                            dataset.iloc[:, -1].values,
                                                            test_size=0.33,
                                                            shuffle=True)

        cl = xgb.XGBClassifier(objective="binary:logistic")
        # cl = RandomForestClassifier()
        grid_cv = GridSearchCV(cl, param_grid, n_jobs=-1, scoring='roc_auc', cv=5, verbose=1)
        _ = grid_cv.fit(X=train_x, y=train_y)
        print(f'GridSearch Best Score = {grid_cv.best_score_}')
        print(f'For Parameters:\n{grid_cv.best_params_}')

        model = grid_cv.best_estimator_
        preds = model.predict(test_x)
        acc = accuracy_score(test_y, preds)
        f1 = f1_score(test_y, preds)
        auc = roc_auc_score(test_y, preds)
        acc_runs.append(acc)
        f1_runs.append(f1)
        auc_runs.append(auc)
        print(f'Accuracy = {round(acc_runs[0] * 100, 2)}%')
        print(f'F1 = {round(f1_runs[0] * 100, 2)}%')
        print(f'AUC = {round(auc_runs[0] * 100, 2)}%')
        print(classification_report(test_y, preds))
        v = 9
    print(f'Accuracy = {round(statistics.mean(acc_runs) * 100, 2)}% ± {round(statistics.stdev(acc_runs) * 100, 2)}%')
    print(f'F1 = {round(statistics.mean(f1_runs) * 100, 2)}% ± {round(statistics.stdev(f1_runs) * 100, 2)}%')
    print(f'AUC = {round(statistics.mean(auc_runs) * 100, 2)}% ± {round(statistics.stdev(auc_runs) * 100, 2)}%')


def predict_scar_grid_search_xgb_vcg_augmentation(region_name: str, select_top_features: bool = False):
    # Step 1: Get original dataset of merged ECG features and scar locations.
    # NOTE: Feature selection is only done using original dataset.
    dataset, continuous_columns, discrete_columns = get_ecg_scar_dataset(region_name=region_name, select_top_features=select_top_features)
    if '_' in dataset.columns.values[-1]:
        raise AssertionError(f'"{dataset.columns.values[-1]}" is not a valid region name')
    if dataset.columns.values[-2] != 'Record_ID':
        raise AssertionError('Column -2 must be "Record_ID"')

    # Step 2: Get augmented dataset of merged ECG features and their corresponding scar locations.
    try:
        dataset_augmented = pd.read_excel('cached_dataset_vcg_augmented.xlsx')
    except FileNotFoundError:
        scar_location_augmented_ds = pd.read_excel(GlobalPaths.scar_location_augmented)
        extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg_augmented,
                                       ann_dir_path=GlobalPaths.pla_annotation_augmented,
                                       metadata_path=GlobalPaths.scar_ecg_augmented_meta,
                                       verbose=True)
        extracted_segments_dict = extractor.extract_segments()
        ecg_augmented_feature_ds = get_ecg_feature_dataset(extracted_segments_dict)
        dataset_augmented = pd.merge(left=scar_location_augmented_ds, right=ecg_augmented_feature_ds, how="inner",
                                     on=["Record_ID"])
        dataset_augmented = dataset_augmented[continuous_columns + discrete_columns + ['Record_ID', region_name]]
        dataset_augmented.to_excel('cached_dataset_vcg_augmented.xlsx', index=False)

    augmented_pids = set(dataset_augmented['Record_ID'].values)

    acc_runs, f1_runs, auc_runs = [], [], []
    for run in range(10):
        print(f'\n--- Run {run+1} ---')
        train, test = train_test_split(dataset, test_size=0.33, shuffle=True)
        real_pids = list(set(train['Record_ID'].values))
        selected_pids = random.sample(real_pids, round(len(real_pids) * 0.5))
        selected_augmented_ds = []
        for pid in selected_pids:
            pid_augmented = pid * 1000 + random.randint(1, 3)
            if pid_augmented in augmented_pids:
                row = dataset_augmented.loc[dataset_augmented['Record_ID'] == pid_augmented].values[0]
                selected_augmented_ds.append(row)
        selected_augmented_ds = pd.DataFrame(selected_augmented_ds, columns=dataset_augmented.columns)
        train = pd.concat([train, selected_augmented_ds], ignore_index=True)
        train = train.sample(frac=1).reset_index(drop=True)

        xgb_cl = xgb.XGBClassifier(objective="binary:logistic")
        grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, scoring='roc_auc', cv=5, verbose=1)
        _ = grid_cv.fit(X=train.iloc[:, 0:-2].values, y=train.iloc[:, -1].values)
        print(f'GridSearch Best Score = {grid_cv.best_score_}')
        print(f'For Parameters:\n{grid_cv.best_params_}')

        test_x, test_y = test.iloc[:, 0:-2].values, test.iloc[:, -1].values
        model = grid_cv.best_estimator_
        preds = model.predict(test_x)
        acc = accuracy_score(test_y, preds)
        f1 = f1_score(test_y, preds)
        auc = roc_auc_score(test_y, preds)
        acc_runs.append(acc)
        f1_runs.append(f1)
        auc_runs.append(auc)
        print(f'Accuracy = {round(acc_runs[0] * 100, 2)}%')
        print(f'F1 = {round(f1_runs[0] * 100, 2)}%')
        print(f'AUC = {round(auc_runs[0] * 100, 2)}%')
    print(f'Accuracy = {round(statistics.mean(acc_runs) * 100, 2)}% ± {round(statistics.stdev(acc_runs) * 100, 2)}%')
    print(f'F1 = {round(statistics.mean(f1_runs) * 100, 2)}% ± {round(statistics.stdev(f1_runs) * 100, 2)}%')
    print(f'AUC = {round(statistics.mean(auc_runs) * 100, 2)}% ± {round(statistics.stdev(auc_runs) * 100, 2)}%')


def predict_scar_tree(region_name: str, select_top_features: bool = False):
    dataset, continuous_columns, discrete_columns = get_ecg_scar_dataset(region_name=region_name, select_top_features=select_top_features)
    # Check if `dataset` is well organized.
    if '_' in dataset.columns.values[-1]:
        raise AssertionError(f'"{dataset.columns.values[-1]}" is not a valid region name')
    if dataset.columns.values[-2] != 'Record_ID':
        raise AssertionError('Column -2 must be "Record_ID"')

    model = xgb.XGBClassifier(objective="binary:logistic",
                              gamma=0.25,
                              colsample_bytree=0.8,
                              learning_rate=0.1,
                              max_depth=5,
                              reg_lambda=30,
                              scale_pos_weight=0.8,
                              subsample=1)

    # model = RandomForestClassifier(n_estimators=100)
    conf_matrix_list = []
    run = 10
    for i in range(run):
        # print(f'\n--- Run {i+1} ---')
        train_x, test_x, train_y, test_y = train_test_split(dataset.iloc[:, 0:-2].values,
                                                            dataset.iloc[:, -1].values,
                                                            test_size=0.3,
                                                            shuffle=True)

        model.fit(train_x, train_y)
        preds = model.predict(test_x)
        a = np.array(confusion_matrix(test_y, preds))
        conf_matrix_list.append(a)
        # print(confusion_matrix(test_y, preds))
        # print(classification_report(test_y, preds))
    conf_all = np.zeros((2, 2))
    for conf in conf_matrix_list:
        conf_all = np.add(conf_all, conf)
    result = np.round(np.divide(conf_all, run))
    print(result)
    v = 9


if __name__ == '__main__':
    # predict_scar_grid_search_tree(region_name='Mid', select_top_features=True)
    predict_scar_tree(region_name='Mid', select_top_features=True)

