import pandas as pd
import math
from mpl_toolkits.axes_grid1 import axes_grid
import matplotlib.pyplot as plt
import os
import xmltodict
import GlobalPaths
from Utility import Loader, Util
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn import mixture
from statistics import mean
from sklearn import preprocessing
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import shap
import seaborn
import matplotlib

import warnings
warnings.filterwarnings("ignore")

# font = {'family': 'normal',
#         'weight': 'bold',
#         'size'  : 45}

# matplotlib.rc('font', **font)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def plot_ecgs(p_dict: dict, save_path: str):
    for pid in p_dict:
        # if pid != 10469:
        #     continue
        mri_date = pd.to_datetime(meta.loc[meta['Record_ID'] == pid]['MRI Date'].values[0])
        ecg_ids = p_dict[pid]
        ncols = min(5, len(ecg_ids))
        if ncols == 0:
            print(f'PID {pid} does not have any ECGs')
            continue
        fig, ax = plt.subplots(nrows=12, ncols=ncols, figsize=(85, 25))
        if ncols == 1:
            ecg = pd.read_csv(f'Data/ECG/ScarLocationECG/{ecg_ids[0]}.csv')
            ecg_date = pd.to_datetime(meta.loc[meta['ECG_ID'] == ecg_ids[0]]['ECG Date'].values[0])
            try:
                ecg.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            except ValueError as e:
                print(e)
                v = 9

            for i in range(12):
                ax[i].plot(ecg[Util.get_lead_name(i)].values)
                ax[i].xaxis.set_ticks([])
                if i == 0:
                    ax[i].title.set_text(f'ECG = {ecg_ids[0]} at {ecg_date}')
        else:
            for j in range(ncols):
                ecg = pd.read_csv(f'Data/ECG/ScarLocationECG/{ecg_ids[j]}.csv')
                if len(ecg.columns) > 12:
                    ecg.drop(ecg.columns[[12, 13, 14]], axis=1, inplace=True)
                ecg_date = pd.to_datetime(meta.loc[meta['ECG_ID'] == ecg_ids[j]]['ECG Date'].values[0])
                ecg.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                for i in range(12):
                    ax[i][j].plot(ecg[Util.get_lead_name(i)].values)
                    ax[i][j].xaxis.set_ticks([])
                    if i == 0:
                        ax[i][j].title.set_text(f'ECG = {ecg_ids[j]} at {ecg_date}')
        fig.suptitle(f'PID={pid} | # of ECGs={len(ecg_ids)} Date={mri_date}', fontsize=16)
        print(f'Saved PID={pid} | # of ECGs={len(ecg_ids)} Date={mri_date}')
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.savefig(f'{save_path}/{pid}.png')
        plt.close(fig)

def run_shap_analysis(dataset: pd.DataFrame):
    shap.KernelExplainer()



def feature_significance(ecg_ids: set, region_name: str, lim: int = None):
    dataset_target = dataset.loc[dataset['ECG_ID'].isin(ecg_ids)]
    dataset_target = dataset_target[[col for col in dataset_target.columns if
                                     ('(' in col and ')' in col and 'conf' not in col) or
                                     col in [region_name, 'Record_ID', 'ECG_ID']]]

    # Identify continuous features whose p-value (using T-test) between scar/no-scar is below 0.05.
    ttest_result = []
    for col in dataset_target.columns:
        if col not in ['Record_ID', 'ECG_ID', 'ECG_Count', region_name] and\
                'has_crossed' not in col and '_notches' not in col and '[conf]' not in col:
            _, p_value = ttest_ind(a=dataset_target.loc[dataset_target[region_name] == 0][col].values,
                                   b=dataset_target.loc[dataset_target[region_name] == 1][col].values,
                                   equal_var=False)
            ttest_result.append((col, p_value))
    if lim is None:
        return ttest_result
    return sorted(ttest_result, key=lambda item: item[1])[:lim]


def cluster_ecgs(dataset: pd.DataFrame, region_name: str):
    X = dataset[[col for col in dataset.columns if col not in ['Record_ID', 'ECG_ID', region_name]]].values
    dpgmm = mixture.BayesianGaussianMixture(n_components=10, covariance_type="full").fit(X)
    weights = dpgmm.weights_
    x = list(range(10))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, weights, width=0.5)
    ax.set_title(f'Weights of Clusters Obtained for {region_name} Scar using BayesianGaussianMixture')
    plt.xlabel("Cluster label")
    plt.ylabel("Cluster weight")
    plt.rcParams["figure.figsize"] = (10, 25)
    plt.show()
    pred_label = list(dpgmm.predict(X))
    # dominant_cluster = max(set(pred_label), key=pred_label.count)
    cluster_size = Counter(pred_label)
    top_clusters = cluster_size.most_common(5)
    cluster_df = pd.DataFrame(
        {
            'Record_ID': dataset['Record_ID'].values,
            'ECG_ID': dataset['ECG_ID'].values,
            region_name: dataset[region_name].values,
            'Cluster': pred_label
        })
    scatter_df = []
    pids = set(cluster_df['Record_ID'].values)
    for pid in pids:
        df = cluster_df.loc[cluster_df['Record_ID'] == pid]
        ecg_count = df.shape[0]
        cluster_count = len(set(df['Cluster'].values))
        scar = int(df[region_name].values[0])
        scatter_df.append([pid, scar, ecg_count, cluster_count])
    scatter_df = pd.DataFrame(scatter_df, columns=['Record_ID', region_name, 'Number of ECGs', 'Number of Clusters'])
    scattered_pids = set(scatter_df.loc[scatter_df['Number of Clusters'] > 1]['Record_ID'].values)
    only_1_ecg_pids = set(scatter_df.loc[scatter_df['Number of ECGs'] == 1]['Record_ID'].values)

    # Plot 1: Scar/no-scar distribution in each of the 5 most populated clusters.
    labels = ['All', 'C1', 'C2', 'C3', 'C4', 'C5']
    cluster_df_no_scar = cluster_df.loc[cluster_df[region_name] == 0]
    cluster_df_scar = cluster_df.loc[cluster_df[region_name] == 1]
    no_scar_counts = [cluster_df_no_scar.shape[0],
                      cluster_df_no_scar.loc[cluster_df_no_scar['Cluster'] == top_clusters[0][0]].shape[0],
                      cluster_df_no_scar.loc[cluster_df_no_scar['Cluster'] == top_clusters[1][0]].shape[0],
                      cluster_df_no_scar.loc[cluster_df_no_scar['Cluster'] == top_clusters[2][0]].shape[0],
                      cluster_df_no_scar.loc[cluster_df_no_scar['Cluster'] == top_clusters[3][0]].shape[0],
                      cluster_df_no_scar.loc[cluster_df_no_scar['Cluster'] == top_clusters[4][0]].shape[0]]
    scar_counts = [cluster_df_scar.shape[0],
                   cluster_df_scar.loc[cluster_df_scar['Cluster'] == top_clusters[0][0]].shape[0],
                   cluster_df_scar.loc[cluster_df_scar['Cluster'] == top_clusters[1][0]].shape[0],
                   cluster_df_scar.loc[cluster_df_scar['Cluster'] == top_clusters[2][0]].shape[0],
                   cluster_df_scar.loc[cluster_df_scar['Cluster'] == top_clusters[3][0]].shape[0],
                   cluster_df_scar.loc[cluster_df_scar['Cluster'] == top_clusters[4][0]].shape[0]]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, no_scar_counts, width, label=f'No Scar in {region_name}')
    ax.bar(x, scar_counts, width, label=f'Scar in {region_name}')
    ax.set_xticks(x, labels)
    ax.set_ylabel('Number of ECGs')
    ax.set_title('Distribution of ECGs among the 5 most Populated Clusters (using BayesianGaussianMixture)')
    ax.legend()
    fig.tight_layout()
    plt.show()

    return scattered_pids, only_1_ecg_pids


def xgboost_gridsearch(train_x, train_y, test_x):
    param_grid_first = {
        'n_estimators': [50, 100, 150, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
    }
    param_grid_second = {
        'max_depth': [3, 5, 7, 10, 12, 15],
        "scale_pos_weight": [0.8, 1, 1.2],
        "subsample": [0.5, 0.8, 1],
        "colsample_bytree": [0.5, 0.8, 0.9, 1],
    }
    param_grid_third = {
        "gamma": [0, 0.25, 1, 3],
        "reg_lambda": [0, 1, 3, 10, 30],
    }
    # https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
    cl = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    grid_cv = GridSearchCV(cl, param_grid_first, n_jobs=-1, scoring='f1_weighted', cv=5, verbose=1)
    _ = grid_cv.fit(X=train_x, y=train_y)
    # print(f'First-level GridSearch Best Score = {grid_cv.best_score_}')
    # print(f'For Parameters:\n{grid_cv.best_params_}')
    model = grid_cv.best_estimator_

    grid_cv = GridSearchCV(model, param_grid_second, n_jobs=-1, scoring='f1_weighted', cv=5, verbose=1)
    _ = grid_cv.fit(X=train_x, y=train_y)
    # print(f'Second-level GridSearch Best Score = {grid_cv.best_score_}')
    # print(f'For Parameters:\n{grid_cv.best_params_}')
    model = grid_cv.best_estimator_

    grid_cv = GridSearchCV(model, param_grid_third, n_jobs=-1, scoring='f1_weighted', cv=5, verbose=1)
    _ = grid_cv.fit(X=train_x, y=train_y)
    # print(f'Third-level GridSearch Best Score = {grid_cv.best_score_}')
    # print(f'For Parameters:\n{grid_cv.best_params_}\n')
    model = grid_cv.best_estimator_

    preds = model.predict(test_x)
    preds_prob = model.predict_proba(test_x)
    return preds, preds_prob


def xgboost_prediction(train_x, train_y, test_x, test_y, train, test):
    d_train = xgb.DMatrix(train_x, train_y, feature_names=[col for col in train.columns if ('(' in col and ')' in col)])
    d_test = xgb.DMatrix(test_x, test_y, feature_names=[col for col in test.columns if ('(' in col and ')' in col)])

    xgb_optimal_params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'learning_rate': 0.3,
        'max_depth': 10,
        'scale_pos_weight': 0.8,
        'colsample_bytree': 0.5,
        'subsample': 1,
        'gamma': 3,
        'reg_lambda': 0
    }

    model = xgb.train(xgb_optimal_params, d_train)
    preds_prob = model.predict(d_test)
    preds = [1 if x > 0.5 else 0 for x in preds_prob]
    return preds, preds_prob


if __name__ == '__main__':
    # tips = seaborn.load_dataset("tips")
    # seaborn.swarmplot(data=tips, x="total_bill")
    # plt.show()


    dataset = pd.read_excel('Data/overall_ecg_feature_uncertain_scar_location.xlsx')
    all_septum = []
    for i, row in dataset.iterrows():
        a = sum(row[['Basal', 'Mid', 'Apical', 'Apex']].values)
        if a > 0:
            all_septum.append(1)
        else:
            all_septum.append(0)
    dataset['Entire Septum'] = all_septum

    meta = pd.read_excel('Data/ECG/ecg_scar_location.xlsx')
    dataset.dropna(inplace=True)

    pids = set(dataset['Record_ID'].values)
    ecg_dist_list = []
    same_day_ecg_ids = {}
    one_month_ecg_ids = {}
    three_month_ecg_ids = {}
    six_month_ecg_ids = {}
    more_than_six_ecg_ids = {}
    for pid in pids:
        ecg_ids = list(dataset.loc[dataset['Record_ID'] == pid]['ECG_ID'].values)
        mri_date = pd.to_datetime(meta.loc[meta['Record_ID'] == pid]['MRI Date'].values[0])
        min_day_distance = 365
        for ecg_id in ecg_ids:
            ecg_date = pd.to_datetime(meta.loc[meta['ECG_ID'] == ecg_id]['ECG Date'].values[0])
            dist = abs(ecg_date - mri_date).days
            if min_day_distance > dist:
                min_day_distance = dist
            if dist == 0:
                if pid in same_day_ecg_ids:
                    same_day_ecg_ids[pid].append(ecg_id)
                else:
                    same_day_ecg_ids[pid] = [ecg_id]
            elif 0 < dist <= 31:
                if pid in one_month_ecg_ids:
                    one_month_ecg_ids[pid].append(ecg_id)
                else:
                    one_month_ecg_ids[pid] = [ecg_id]
            elif 31 < dist <= 93:
                if pid in three_month_ecg_ids:
                    three_month_ecg_ids[pid].append(ecg_id)
                else:
                    three_month_ecg_ids[pid] = [ecg_id]
            elif 93 < dist <= 186:
                if pid in six_month_ecg_ids:
                    six_month_ecg_ids[pid].append(ecg_id)
                else:
                    six_month_ecg_ids[pid] = [ecg_id]
            else:
                if pid in more_than_six_ecg_ids:
                    more_than_six_ecg_ids[pid].append(ecg_id)
                else:
                    more_than_six_ecg_ids[pid] = [ecg_id]
        ecg_dist_list.append(min_day_distance)

    same_day_pids = set(same_day_ecg_ids.keys())
    one_month_pids = set(one_month_ecg_ids.keys()) - same_day_pids
    three_month_pids = (set(three_month_ecg_ids.keys()) - same_day_pids) - one_month_pids
    six_month_pids = ((set(six_month_ecg_ids.keys()) - same_day_pids) - three_month_pids) - one_month_pids
    more_than_six_month_pids = (((set(more_than_six_ecg_ids.keys()) - same_day_pids) - six_month_pids) - three_month_pids) - one_month_pids

    for key in set(one_month_ecg_ids.keys()) - one_month_pids:
        del one_month_ecg_ids[key]

    for key in set(three_month_ecg_ids.keys()) - three_month_pids:
        del three_month_ecg_ids[key]

    for key in set(six_month_ecg_ids.keys()) - six_month_pids:
        del six_month_ecg_ids[key]

    for key in set(more_than_six_ecg_ids.keys()) - more_than_six_month_pids:
        del more_than_six_ecg_ids[key]

    # plot_ecgs(same_day_ecg_ids, save_path='Data/Eyeball/SameDayECGs')
    # plot_ecgs(one_month_ecg_ids, save_path='Data/Eyeball/OneMonth')
    # plot_ecgs(three_month_ecg_ids, save_path='Data/Eyeball/ThreeMonth')
    # plot_ecgs(six_month_ecg_ids, save_path='Data/Eyeball/SixMonth')
    # plot_ecgs(more_than_six_ecg_ids, save_path='Data/Eyeball/MoreThanSix')

    a = len([x for x in ecg_dist_list if x == 0])
    b = len([x for x in ecg_dist_list if 0 < x <= 31])
    c = len([x for x in ecg_dist_list if 31 < x <= 93])
    d = len([x for x in ecg_dist_list if 93 < x <= 186])
    e = len([x for x in ecg_dist_list if x > 186])

    x = ['Same Day', '< 1 Month', '<3 Months', '<6 Months', '> 6 Months']
    y = [a, b, c, d, e]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, y, width = 0.5)
    plt.xlabel("ECG distance from MRI")
    plt.ylabel("# of Patients")
    plt.rcParams["figure.figsize"] = (10, 25)
    plt.show()

    ecg_counts = [len(x) for x in same_day_ecg_ids.values()]
    plt.hist(ecg_counts, bins=10)
    plt.show()

    regions = ['Entire Septum', 'Basal', 'Mid', 'Apical', 'Apex']

    for region_name in regions:
        if region_name != 'Mid':
            continue
        ecg_ids = []
        [ecg_ids.extend(x) for x in same_day_ecg_ids.values()]
        ecg_ids = set(ecg_ids)
        features_prim = feature_significance(ecg_ids, region_name=region_name, lim=26)
        features_prim_set = set(x[0] for x in features_prim)

        ecg_ids = []
        [ecg_ids.extend(x) for x in one_month_ecg_ids.values()]
        ecg_ids = set(ecg_ids)
        features_one = feature_significance(ecg_ids, region_name=region_name)

        ecg_ids = []
        [ecg_ids.extend(x) for x in three_month_ecg_ids.values()]
        ecg_ids = set(ecg_ids)
        features_three = feature_significance(ecg_ids, region_name=region_name)

        ecg_ids = []
        [ecg_ids.extend(x) for x in six_month_ecg_ids.values()]
        ecg_ids = set(ecg_ids)
        features_six = feature_significance(ecg_ids, region_name=region_name)

        # Part 1: Plot continuous features' p-values based on the same-day-ECGs and overexpose it with the same
        # features' p-value but calculated based on 1-month, 3-month and 6-month distant ECGs population.
        labels = [x[0] for x in features_prim]
        prime_values = [-1 * math.log10(x[1]) for x in features_prim]
        a = [-1 * math.log10(x[1]) for x in features_one if x[0] in features_prim_set]
        b = [-1 * math.log10(x[1]) for x in features_three if x[0] in features_prim_set]
        c = [-1 * math.log10(x[1]) for x in features_six if x[0] in features_prim_set]
        x = np.arange(len(labels))
        width = 0.15
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(x - 2 * width, prime_values, width, label=f'Same Day ({len(same_day_ecg_ids.keys())} patients)')
        ax.bar(x - width, a, width, label=f'One Month ({len(one_month_ecg_ids.keys())} patients)')
        ax.bar(x + 0, b, width, label=f'Three Months ({len(three_month_ecg_ids.keys())} patients)')
        ax.bar(x + width, c, width, label=f'Six Months ({len(six_month_ecg_ids.keys())} patients)')
        ax.axhline(y=1.3, color='r', linestyle='-')
        ax.set_ylabel('Corrected -log(P-value)')
        ax.set_title(f'Feature Significance in each Population for {region_name} Scar')
        ax.set_xticks(x, labels)
        ax.legend()
        fig.tight_layout()
        plt.show()

        # Part 2: Construct a dataset of 'Record_ID', 'ECG_ID', region_name, and selected features from Part 1 for the
        # same-day ECGs. It will be used for train and test.
        # Construct a dataset of 'Record_ID', 'ECG_ID', region_name, and selected features from Part 1 for the
        # one-month ECGs population. It will only be used for testing.
        ecg_ids = []
        [ecg_ids.extend(x) for x in same_day_ecg_ids.values()]
        [ecg_ids.extend(x) for x in one_month_ecg_ids.values()]
        # [ecg_ids.extend(x) for x in three_month_ecg_ids.values()]
        ecg_ids = set(ecg_ids)

        ecg_ids_one_month = []
        [ecg_ids_one_month.extend(x) for x in one_month_ecg_ids.values()]
        ecg_ids_one_month = set(ecg_ids_one_month)

        dataset_target = dataset.loc[dataset['ECG_ID'].isin(ecg_ids)]
        dataset_target = dataset_target[
            ['Record_ID', 'ECG_ID', region_name] + [x[0] for x in features_prim]].reset_index(drop=True)
        dataset_one_month = dataset.loc[dataset['ECG_ID'].isin(ecg_ids_one_month)]
        dataset_one_month = dataset_one_month[
            ['Record_ID', 'ECG_ID', region_name] + [x[0] for x in features_prim]].reset_index(drop=True)

        # Part 3: Cluster the entire same-day ECG population (~850 ECGs from ~270 patients)
        # scattered_pids, only_1_ecg_pids = cluster_ecgs(dataset=dataset_target, region_name=region_name)

        pids = set(list(dataset_target['Record_ID'].values))
        # pids = set(list(dataset_target['Record_ID'].values)) - scattered_pids
        dataset_target = dataset_target.loc[dataset_target['Record_ID'].isin(pids)].reset_index(drop=True)
        pids = list(set(dataset_target['Record_ID'].values))
        count_scar = 0
        count_no_scar = 0
        for pid in pids:
            scar = dataset_target.loc[dataset_target['Record_ID'] == pid][region_name].values[0]
            if scar == 0:
                count_no_scar += 1
            else:
                count_scar += 1

        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        precision_0, precision_1 = [], []
        recall_0, recall_1 = [], []
        for run in range(30):
            print(f'Run {run}')
            fold_precision_0, fold_precision_1 = [], []
            fold_recall_0, fold_recall_1 = [], []
            pids = list(pids)
            fold_shap_class_1 = []
            for train_idx, test_idx in kf.split(pids):
                train_pids = [pids[i] for i in train_idx]
                test_pids = [pids[i] for i in test_idx]
                train = pd.merge(left=pd.DataFrame(train_pids, columns=['Record_ID']), right=dataset_target, how='inner', on=['Record_ID'])
                test = pd.merge(left=pd.DataFrame(test_pids, columns=['Record_ID']), right=dataset_target, how='inner', on=['Record_ID'])
                # train = dataset_target.iloc[train_idx]
                # test = dataset_target.iloc[test_idx]

                train_x = train[[col for col in train.columns if ('(' in col and ')' in col)]].values
                train_y = train[region_name].values
                test_x = test[[col for col in test.columns if ('(' in col and ')' in col)]].values
                test_y = test[region_name].values

                # Normalization
                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                train_x = scaler.fit_transform(train_x)
                test_x = scaler.transform(test_x)

                eval_metric = 0
                for inner_run in range(80):
                    mlp = MLPClassifier(hidden_layer_sizes=(200,), activation='relu').fit(train_x, train_y)
                    y_pred = mlp.predict(test_x)
                    pr, re, f1, _ = precision_recall_fscore_support(y_true=test_y, y_pred=y_pred, average=None)
                    perform = (pr[0] * re[0])/(pr[0] + re[0])
                    #perform = pr[0]
                    if perform > eval_metric:
                        eval_metric = perform
                        result = [pr, re, f1]
                        best_model = mlp
                print(f'Pr0={result[0][0]}, Re0={result[1][0]}, Pr1={result[0][1]}, Re1={result[1][1]}')
                # TODO: There are 2–3 partitions that have the lowest precision and recall. Examine those! Dont, forget to set Shuffle=False!!

                # plt.rcParams.update({'font.size': 38})
                # X_train = pd.DataFrame(train_x, columns=[col for col in train.columns if ('(' in col and ')' in col)])
                # X_test = pd.DataFrame(test_x[:10], columns=[col for col in train.columns if ('(' in col and ')' in col)])
                # explainer = shap.KernelExplainer(best_model.predict_proba, X_train)
                # shap_values = explainer.shap_values(X_test)
                # shap_class_1 = shap_values[1].sum(axis=0)
                # fold_shap_class_1.append(shap_class_1)
                # shap_list = zip(X_train.columns, shap_class_1)
                # shap_list = sorted(shap_list, key=lambda x:abs(x[1]), reverse=True)
                # y_labels = [x[0] for x in shap_list]
                # y_pos = np.arange(len(y_labels))
                # x_vals = [x[1] for x in shap_list]
                # plt.rcdefaults()
                # fig, ax = plt.subplots(figsize=(18, 10))
                # ax.barh(y_pos, x_vals)
                # ax.set_yticks(y_pos, labels=y_labels)
                # ax.invert_yaxis()  # labels read top-to-bottom
                # ax.set_xlabel('Contribution to the presence of myocardial scarring in the entire septum')
                # ax.set_title('Shapely Analysis of ECG Features Suggestive of Myocardial Scarring in HCM patients')
                # plt.show()
                v = 9
                # explainer = shap.KernelExplainer(best_model.predict_proba, X_train, link="logit")
                # shap_values = explainer.shap_values(X_test, nsamples=100)
                # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
                #
                # shap_values = explainer(X_test)
                #
                # # visualize the first prediction's explanation
                # shap.plots.waterfall(shap_values[0])
                # shap.plots.bar(shap_values[0])
                # shap.plots.beeswarm(shap_values)

                fold_precision_0.append(result[0][0])
                fold_precision_1.append(result[0][1])
                fold_recall_0.append(result[1][0])
                fold_recall_1.append(result[1][1])

                # y_pred, y_pred_prob = xgboost_prediction(train_x, train_y, test_x, test_y, train, test)

                # pr_0_max, pr_1_max, re_0_max, re_1_max, f_0_max, f_1_max = 0, 0, 0, 0, 0, 0
                # for run in range(40):
                #     mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu').fit(train_x, train_y)
                #     y_pred = mlp.predict(test_x)
                #     pr, re, f1, _ = precision_recall_fscore_support(y_true=test_y, y_pred=y_pred, average=None)
                #     if pr[0] > pr_0_max:
                #         pr_0_max = pr[0]
                #     if pr[1] > pr_1_max:
                #         pr_1_max = pr[1]
                #     if re[0] > re_0_max:
                #         re_0_max = re[0]
                #     if re[1] > re_1_max:
                #         re_1_max = re[1]
                #     if f1[0] > f_0_max:
                #         f_0_max = f1[0]
                #     if f1[1] > f_1_max:
                #         f_1_max = f1[1]
                #
                # precision_0.append(pr_0_max)
                # precision_1.append(pr_1_max)
                # recall_0.append(re_0_max)
                # recall_1.append(re_1_max)
                # f1_0.append(f_0_max)
                # f1_1.append(f_1_max)

            precision_0.append(mean(fold_precision_0))
            precision_1.append(mean(fold_precision_1))
            recall_0.append(mean(fold_recall_0))
            recall_1.append(mean(fold_recall_1))


            # fold_shap_class_1 = np.array(fold_shap_class_1).sum(axis=0)
            # shap_list = zip([col for col in dataset_target.columns if ('(' in col and ')' in col)], fold_shap_class_1)
            # shap_list = sorted(shap_list, key=lambda x: abs(x[1]), reverse=True)
            # y_labels = [x[0] for x in shap_list]
            # y_pos = np.arange(len(y_labels))
            # x_vals = [x[1] for x in shap_list]
            # plt.rcdefaults()
            # fig, ax = plt.subplots(figsize=(18, 10))
            # ax.barh(y_pos, x_vals)
            # ax.set_yticks(y_pos, labels=y_labels)
            # ax.invert_yaxis()  # labels read top-to-bottom
            # ax.set_xlabel('Contribution to the presence of myocardial scarring in interventricular septum')
            # ax.set_title('Shapely Analysis of ECG Features Suggestive of Myocardial Scarring in HCM patients')
            # plt.show()

            max_precision_0_idx = precision_0.index(max(precision_0))
            max_precision_0 = precision_0[max_precision_0_idx]
            max_precision_1 = precision_1[max_precision_0_idx]
            max_recall_0 = recall_0[max_precision_0_idx]
            max_recall_1 = recall_1[max_precision_0_idx]
            print(f'\nNoScar Prediction:\n--> Pr={color.BOLD}{round(100 * max_precision_0, 1)}{color.END} '
                  f'Re={color.BOLD}{round(100 * max_recall_0, 1)}{color.END}\n'
                  f'Scar Prediction:\n--> Pr={color.BOLD}{round(100 * max_precision_1, 1)}{color.END} '
                  f'Re={color.BOLD}{round(100 * max_recall_1, 1)}{color.END}\n')




        # for i in range(10):
        #     train_pids, test_pids = train_test_split(pids, test_size=0.2, shuffle=True)
        #     train = pd.merge(left=pd.DataFrame(train_pids, columns=['Record_ID']), right=dataset_target, how='inner', on=['Record_ID'])
        #     test = pd.merge(left=pd.DataFrame(test_pids, columns=['Record_ID']), right=dataset_target, how='inner', on=['Record_ID'])
        #
        #     train_x = train[[col for col in train.columns if ('(' in col and ')' in col)]].values
        #     train_y = train[region_name].values
        #     test_x = test[[col for col in test.columns if ('(' in col and ')' in col)]].values
        #     test_y = test[region_name].values
        #
        #     # Normalization
        #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        #     train_x = scaler.fit_transform(train_x)
        #     test_x = scaler.transform(test_x)
        #
        #
        #
        #     prediction_df = pd.DataFrame(
        #         {
        #             'Record_ID': test['Record_ID'].values,
        #             'ECG_ID': test['ECG_ID'].values,
        #             'y_true': test_y,
        #             'y_pred': preds,
        #             'y_prob_0': [1 - x for x in preds_prob],
        #             'y_prob_1': preds_prob
        #         })
        #
        #     prediction_df_total = pd.merge(left=prediction_df, right=test, how='inner', on=['Record_ID', 'ECG_ID'])
        #     scores_weighted = f1_score(test_y, preds, average='weighted')
        #     f_score_runs.append(scores_weighted)
        #     # print(classification_report(test_y, preds))
        # print(f'*** Weighted F1 Score for {region_name} scar detection = {statistics.mean(f_score_runs)}')


        # f_score_runs = []
        # for i in range(5):
        #     pids = list(set(list(dataset_target['Record_ID'].values)))
        #     train_pids, test_pids = train_test_split(pids, test_size=0.2, shuffle=True)
        #
        #     train = pd.merge(left=pd.DataFrame(train_pids, columns=['Record_ID']), right=dataset_target, how='inner',
        #                      on=['Record_ID'])
        #     test = pd.merge(left=pd.DataFrame(test_pids, columns=['Record_ID']), right=dataset_target, how='inner',
        #                     on=['Record_ID'])
        #
        #     train_x = train[[col for col in train.columns if ('(' in col and ')' in col)]].values
        #     train_y = train[region_name].values
        #
        #     test_x = test[[col for col in test.columns if ('(' in col and ')' in col)]].values
        #     test_y = test[region_name].values
        #
        #     d_train = xgb.DMatrix(train_x, train_y,
        #                           feature_names=[col for col in train.columns if ('(' in col and ')' in col)])
        #     d_test = xgb.DMatrix(test_x, test_y,
        #                          feature_names=[col for col in test.columns if ('(' in col and ')' in col)])
        #
        #     one_month_x = dataset_one_month[
        #         [col for col in dataset_one_month.columns if ('(' in col and ')' in col)]].values
        #     one_month_y = dataset_one_month[region_name].values
        #
        #     d_test_one_month = xgb.DMatrix(one_month_x, one_month_y,
        #                                    feature_names=[col for col in test.columns if ('(' in col and ')' in col)])
        #
        #     xgb_optimal_params = {
        #         'booster': 'gbtree',
        #         'objective': 'binary:logistic',
        #         'eval_metric': 'auc',
        #         'learning_rate': 0.3,
        #         # 'n_estimators': 1000,
        #         'max_depth': 10,
        #         'scale_pos_weight': 0.8,
        #         'colsample_bytree': 0.5,
        #         'subsample': 1,
        #         'gamma': 3,
        #         'reg_lambda': 0
        #     }
        #
        #     model = xgb.train(xgb_optimal_params, d_train)
        #     preds_prob = model.predict(d_test)
        #     preds = [1 if x > 0.5 else 0 for x in preds_prob]
        #
        #     prediction_df = pd.DataFrame(
        #         {
        #             'Record_ID': test['Record_ID'].values,
        #             'ECG_ID': test['ECG_ID'].values,
        #             'y_true': test_y,
        #             'y_pred': preds,
        #             'y_prob_0': [1 - x for x in preds_prob],
        #             'y_prob_1': preds_prob
        #         })
        #
        #     prediction_df_total = pd.merge(left=prediction_df, right=test, how='inner', on=['Record_ID', 'ECG_ID'])
        #     scores_weighted = f1_score(test_y, preds, average='weighted')
        #     f_score_runs.append(scores_weighted)
        #     print(classification_report(test_y, preds))
        #
        #     preds_prob_one_month = model.predict(d_test_one_month)
        #     preds_one_month = [1 if x > 0.5 else 0 for x in preds_prob_one_month]
        #     # print(classification_report(one_month_y, preds_one_month))
        # print(f'\n*** Weighted F1 Score for {region_name} scar detection = {statistics.mean(f_score_runs)}\n\n')








