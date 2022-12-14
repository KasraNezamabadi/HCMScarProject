import pandas as pd
import numpy as np
from sklearn import mixture
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    dataset = pd.read_excel('Data/overall_ecg_feature_uncertain_scar_location.xlsx')
    dataset.dropna(inplace=True)

    region_name = 'Mid'
    # Keep only ECG features, their confidence, record ID, and the region name.
    dataset = dataset[[col for col in dataset.columns if
                       ('(' in col and ')' in col) or
                       col in [region_name, 'Record_ID', 'ECG_ID', 'ECG_Count']]]

    dataset = dataset[[col for col in dataset.columns if '[conf]' not in col]]

    # dataset = dataset[[col for col in dataset.columns if
    #                    '(II)' in col or
    #                    # '(aVF)' in col or
    #                    '(V2)' in col or
    #                    '(V6)' in col or
    #                    col == region_name]]
    #
    # dataset = dataset[[col for col in dataset.columns if
    #                    'QRS_energy' not in col and
    #                    'QRS_prominence' not in col and
    #                    '_notches' not in col and
    #                    'has_crossed' not in col and
    #                    't_prominence' not in col and
    #                    't_energy' not in col and
    #                    't_duration' not in col and
    #                    'st_rvalue' not in col and
    #                    'st_slope_max' not in col and
    #                    'st_slope_min' not in col
    #                    ]]

    # Identify continuous features whose p-value (using T-test) between scar/no-scar is below 0.05.
    ttest_result = []
    for col in dataset.columns:
        if col not in ['Record_ID', 'ECG_ID', 'ECG_Count', region_name] and 'has_crossed' not in col and '_notches' not in col and '[conf]' not in col:
            _, p_value = ttest_ind(a=dataset.loc[dataset[region_name] == 0][col].values,
                                   b=dataset.loc[dataset[region_name] == 1][col].values,
                                   equal_var=False)
            ttest_result.append((col, p_value))
    ttest_result = sorted(ttest_result, key=lambda item: item[1])
    # selected_features = [x[0] for x in ttest_result if x[1] < 0.001]
    ttest_result = ttest_result[:15]
    selected_features = [x[0] for x in ttest_result]

    dataset = dataset[['Record_ID', 'ECG_ID', region_name] + selected_features]
    pids = set(dataset['Record_ID'].values)

    train_pids, test_pids = train_test_split(pids, test_size=0.2, shuffle=True)

    train = pd.merge(left=pd.DataFrame(train_pids, columns=['Record_ID']), right=dataset, how='inner', on=['Record_ID'])
    test = pd.merge(left=pd.DataFrame(test_pids, columns=['Record_ID']), right=dataset, how='inner', on=['Record_ID'])

    train_certain = []
    for pid in pids:
        target_df = train.loc[train['Record_ID'] == pid].reset_index(drop=True)
        if target_df.shape[0] > 4:
            X = target_df[[col for col in target_df.columns if col not in ['Record_ID', 'ECG_ID', region_name]]].values
            dpgmm = mixture.BayesianGaussianMixture(n_components=3, covariance_type="full").fit(X)
            pred_label = list(dpgmm.predict(X))
            dominant_cluster = max(set(pred_label), key=pred_label.count)
            for i in range(len(pred_label)):
                if pred_label[i] == dominant_cluster:
                    row = target_df.iloc[i].values
                    train_certain.append(row)

    train_certain = pd.DataFrame(train_certain, columns=dataset.columns.values)



    dataset_0 = dataset.loc[dataset[region_name] == 0][[col for col in dataset.columns if col != region_name]]
    dataset_1 = dataset.loc[dataset[region_name] == 1][[col for col in dataset.columns if col != region_name]]

    dataset_0.reset_index(drop=True, inplace=True)
    dataset_1.reset_index(drop=True, inplace=True)

    X_0 = dataset_0.values
    X_1 = dataset_1.values
    dpgmm_0 = mixture.BayesianGaussianMixture(n_components=10, covariance_type="full").fit(X_0)
    print(dpgmm_0.weights_)
    print(sum(dpgmm_0.weights_))
    pred_0 = dpgmm_0.predict_proba(X_0)
    pred_0_1 = dpgmm_0.predict_proba(X_1)

    dpgmm_1 = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(X_1)
    pred_1 = dpgmm_1.predict_proba(X_1)

    for i, row in dataset_1.iterrows():
        a = pred_0_1[i]
        b = pred_1[i]
        v = 9



    v = 9

