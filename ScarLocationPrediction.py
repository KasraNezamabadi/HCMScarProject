import statistics
import random
from collections import Counter
import os

from scipy.stats import norm, ttest_ind
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
import MuseXMLX
from MuseXMLX import MuseXmlParser
from Utility import Loader
from GEMuseXMLReader import GEMuseXMLReader
from datetime import timedelta
from Utility import Util
import matplotlib.pyplot as plt

import base64

import xmltodict

import GlobalPaths
from QTSegmentExtractor import QTSegmentExtractor
from fQRS import get_ecg_scar_dataset, get_ecg_feature_dataset, get_scar_location_dataset, get_scar_subregion, get_ecg_feature_dataset_several_visit, get_ecg_feature_ds_uncertainty

param_grid_first = {
    'n_estimators': [50, 100, 150, 200, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
}

param_grid_second = {
    'max_depth': [3, 5, 7, 10, 15],
    "scale_pos_weight": [0.8, 1, 1.2],
    "subsample": [0.5, 0.8, 1],
    "colsample_bytree": [0.5, 0.8, 1],
}

param_grid_third = {
    "gamma": [0, 0.25, 1, 3],
    "reg_lambda": [0, 1, 3, 10, 30],
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


def add_augmented_ds(train: pd.DataFrame):
    dataset_augmented = pd.read_excel('cached_dataset_vcg_augmented.xlsx')
    augmented_pids = set(dataset_augmented['Record_ID'].values)
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
    return train


def predict_scar_tree(region_name: str, select_top_features: bool = False, use_augmentation: bool = True):
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
    # model = LogisticRegression(solver='liblinear', multi_class='ovr')
    # model = GaussianNB()

    # model = StackingClassifier(
    #     classifiers=[
    #         GaussianNB(),
    #         xgb.XGBClassifier(objective="binary:logistic",
    #                           gamma=0.25,
    #                           colsample_bytree=0.8,
    #                           learning_rate=0.1,
    #                           max_depth=5,
    #                           reg_lambda=30,
    #                           scale_pos_weight=0.8,
    #                           subsample=1),
    #         RandomForestClassifier(n_estimators=100),
    #         LogisticRegression(solver='liblinear', multi_class='ovr')
    #     ],
    #     use_probas=True,
    #     meta_classifier=GaussianNB()
    # )
    conf_matrix_list = []
    run = 10
    for i in range(run):
        # print(f'\n--- Run {i+1} ---')
        train, test = train_test_split(dataset, test_size=0.3, shuffle=True)
        if i == 0:
            print(f'Train: {Counter(train.iloc[:, -1].values)}')
            print(f'Test: {Counter(test.iloc[:, -1].values)}')
        if use_augmentation:
            train = add_augmented_ds(train)
            if i == 0:
                print(f'Train Augmented: {Counter(train.iloc[:, -1].values)}')

        train_x, train_y = train.iloc[:, 0:-2].values, train.iloc[:, -1].values
        model.fit(train_x, train_y)

        test_x, test_y = test.iloc[:, 0:-2].values, test.iloc[:, -1].values
        preds = model.predict(test_x)
        preds_prob = model.predict_proba(test_x)

        prediction_df = pd.DataFrame(
            {
                'Record_ID': test.iloc[:, -2].values,
                'y_true': test_y,
                'y_pred': preds,
                'y_prob_0': preds_prob[:, 0],
                'y_prob_1': preds_prob[:, 1]
            })
        scar_subregion_df = get_scar_subregion(region_name=region_name)
        prediction_df = pd.merge(left=prediction_df, right=scar_subregion_df, how="inner", on=["Record_ID"])

        mid_l_prediction = prediction_df.loc[prediction_df['Mid L'] == 1]
        mid_l_tp = len(mid_l_prediction.loc[mid_l_prediction['y_pred'] == 1])
        mid_l_fn = len(mid_l_prediction.loc[mid_l_prediction['y_pred'] == 0])

        print(classification_report(prediction_df['y_true'].values, prediction_df['y_pred'].values))

        fn_df = prediction_df.loc[(prediction_df['y_true'] == 1) & (prediction_df['y_pred'] == 0)]

        prediction_df_confident = prediction_df.loc[(prediction_df['y_prob_0'] > 0.7) | (prediction_df['y_prob_1'] > 0.7)]
        print(classification_report(prediction_df_confident['y_true'].values, prediction_df_confident['y_pred'].values))

        prediction_df_no_s = prediction_df.loc[prediction_df['Mid S'] == 0]
        # print(classification_report(prediction_df_no_s['y_true'].values, prediction_df_no_s['y_pred'].values))

        tp_df_no_s = prediction_df_no_s.loc[(prediction_df_no_s['y_true'] == 1) & (prediction_df_no_s['y_pred'] == 1)]
        tn_df_no_s = prediction_df_no_s.loc[(prediction_df_no_s['y_true'] == 0) & (prediction_df_no_s['y_pred'] == 0)]
        fn_df_no_s = prediction_df_no_s.loc[(prediction_df_no_s['y_true'] == 1) & (prediction_df_no_s['y_pred'] == 0)]
        fp_df_no_s = prediction_df_no_s.loc[(prediction_df_no_s['y_true'] == 0) & (prediction_df_no_s['y_pred'] == 1)]

        scar_df = pd.read_excel(GlobalPaths.scar_location)
        scar_df = scar_df[[col for col in scar_df.columns if 'Basal' in col or 'Apical' in col or 'Apex' in col] + ['Record_ID']]
        scar_df.dropna(inplace=True)
        fp_df_no_s = pd.merge(left=fp_df_no_s, right=scar_df, how="inner", on=["Record_ID"])

        a = np.array(confusion_matrix(test_y, preds))
        conf_matrix_list.append(a)
        print(confusion_matrix(test_y, preds))
        print(classification_report(test_y, preds))
    conf_all = np.zeros((2, 2))
    for conf in conf_matrix_list:
        conf_all = np.add(conf_all, conf)
    result = np.round(np.divide(conf_all, run))
    print(result)


def parse_muse_ecgs(mri_df: pd.DataFrame):
    mri_pids = set(mri_df['Record_ID'].values)
    try:
        ecg_scar_df = pd.read_excel('Data/ecg_scar_df.xlsx')
    except FileNotFoundError:
        xml_names = [f for f in os.listdir(GlobalPaths.muse) if not f.startswith('.')]
        ecg_dataset = []
        print(f'Parsing {len(xml_names)} MUSE files ...')
        count = 0
        ecg_ids = {}
        for xml_name in xml_names:
            muse_dict = xmltodict.parse(open(os.path.join(GlobalPaths.muse, xml_name), 'rb').read().decode('utf8'))['RestingECG']
            try:
                ecg_pid = int(muse_dict['PatientDemographics']['PatientID'])
                if ecg_pid not in mri_pids:
                    continue
                ecg_date = pd.to_datetime(muse_dict['TestDemographics']['AcquisitionDate'])
                frequency = int(muse_dict['Waveform'][1]['SampleBase'])
                Loader.translate_muse_measurements(path_muse=os.path.join(GlobalPaths.muse, xml_name), path_save='Data/temp_muse_ecg.csv')
                ecg_df = pd.read_csv('Data/temp_muse_ecg.csv')
                ecg_df.drop(ecg_df.columns[len(ecg_df.columns) - 1], axis=1, inplace=True)
                if ecg_pid in ecg_ids:
                    ecg_ids[ecg_pid] += 1
                else:
                    ecg_ids[ecg_pid] = 1
                # if int(f'{ecg_pid}00{ecg_ids[ecg_pid]}') == 10001003:
                #     fig, ax = plt.subplots(nrows=12, ncols=1, figsize=(15, 15))
                #     for lead_index in range(12):
                #         ax[lead_index].plot(ecg_df['ecg_denoised'][Util.get_lead_name(lead_index)].values)
                #         ax[lead_index].title.set_text(Util.get_lead_name(lead_index))
                #     plt.show()
                #     v = 9
                ecg_file_name = int(f'{ecg_pid}00{ecg_ids[ecg_pid]}')
                ecg_df.to_csv(f'Data/ECG/ScarLocationECG/{ecg_file_name}.csv', index=False)
                ecg_dataset.append([ecg_pid, ecg_date, frequency, ecg_file_name])
            except KeyError:
                continue
            count += 1
            if count % 50 == 0:
                print(f'   --- {count} files processed')

        ecg_df = pd.DataFrame(ecg_dataset, columns=['Record_ID', 'ECG Date', 'Frequency', 'ECG_ID'])
        ecg_scar_df = pd.merge(left=mri_df, right=ecg_df, how="inner", on=["Record_ID"])
        ecg_scar_df.to_excel('Data/ecg_scar_df.xlsx', index=False)

    myectomy_asa_df = pd.read_excel(GlobalPaths.ehr)[['Record_ID', 'Myectomy_date', 'ASA_date']]
    myectomy_asa_df.dropna(inplace=True)

    pids = set(ecg_scar_df['Record_ID'].values)
    remove_ecg_ids = []
    for pid in pids:
        target_df = ecg_scar_df.loc[ecg_scar_df['Record_ID'] == pid]
        mri_date = pd.to_datetime(target_df['MRI Date'].values[0])
        for index, row in target_df.iterrows():
            ecg_date = pd.to_datetime(row['ECG Date'])
            if abs(mri_date - ecg_date) > timedelta(days=360):
                remove_ecg_ids.append(row['ECG_ID'])
            elif pid in set(myectomy_asa_df['Record_ID'].values):
                try:
                    myectomy_date = pd.to_datetime(myectomy_asa_df.loc[myectomy_asa_df['Record_ID'] == pid]['Myectomy_date'].values[0])
                    if ecg_date > myectomy_date:
                        remove_ecg_ids.append(row['ECG_ID'])
                    else:
                        asa_date = pd.to_datetime(myectomy_asa_df.loc[myectomy_asa_df['Record_ID'] == pid]['Myectomy_date'].values[0])
                        if ecg_date > asa_date:
                            remove_ecg_ids.append(row['ECG_ID'])
                except ValueError:
                    pass

    print(f'{len(remove_ecg_ids)} ECGs are more than 1 year distant from MRI acquisition or after Myectomy/Asa.')
    remove_ecg_ids = set(remove_ecg_ids)
    final_ecg_scar_df = []
    for _, row in ecg_scar_df.iterrows():
        if row['ECG_ID'] not in remove_ecg_ids:
            final_ecg_scar_df.append(row.values)

    final_ecg_scar_df = pd.DataFrame(final_ecg_scar_df, columns=ecg_scar_df.columns)
    final_ecg_scar_df.to_excel('Data/ECG/ecg_scar_location.xlsx', index=False)
    return final_ecg_scar_df

import struct
from sklearn.feature_selection import mutual_info_classif

if __name__ == '__main__':
    xml_names = [f for f in os.listdir(GlobalPaths.muse) if not f.startswith('.')]
    for xml_name in xml_names:
        muse_dict = xmltodict.parse(open(os.path.join(GlobalPaths.muse, xml_name), 'rb').read().decode('utf8'))[
            'RestingECG']
        ecg_pid = int(muse_dict['PatientDemographics']['PatientID'])
        # if ecg_pid != 10003:
        #     continue
        ecg_date = pd.to_datetime(muse_dict['TestDemographics']['AcquisitionDate'] + ' ' + muse_dict['TestDemographics']['AcquisitionTime'])
        frequency = int(muse_dict['Waveform'][1]['SampleBase'])

        text = muse_dict['Waveform'][1]['LeadData'][2]['WaveFormData']
        gain = float(muse_dict['Waveform'][1]['LeadData'][2]['LeadAmplitudeUnitsPerBit'])
        text = text.strip()
        signal_byte = base64.b64decode(text)
        lead_signal = []
        for t in range(0, len(signal_byte), 2):
            sample = round(struct.unpack('h', bytes([signal_byte[t], signal_byte[t + 1]]))[0] * gain)
            lead_signal.append(sample)
        plt.figure(f'PID=10003 Date={ecg_date}', figsize=(15, 5))
        plt.plot(lead_signal)
        plt.title(f'PID=10003 Date={ecg_date}')
        plt.show()
        v = 9


    try:
        dataset = pd.read_excel('Data/overall_ecg_feature_uncertain_scar_location.xlsx')
    except FileNotFoundError:
        scar_df = pd.read_excel(GlobalPaths.scar_location)
        scar_df = scar_df[['Record_ID', 'MRI Date'] +
                          [col for col in scar_df.columns if 'Basal' in col or 'Mid' in col or 'Apical' in col or 'Apex' in col]]
        scar_df.dropna(inplace=True)
        scar_df.reset_index(drop=True, inplace=True)

        # Clean MRI Date.
        for index, row in scar_df.iterrows():
            try:
                mri_date = pd.to_datetime(row['MRI Date'])
                scar_df.iat[index, 1] = mri_date
            except pd.errors.ParserError:
                v = 9
            except ValueError:
                date_str = str(row['MRI Date'])
                if '.' in date_str:
                    date_str = date_str.replace('.', '')
                if ',' in date_str:
                    mri_date = pd.to_datetime(date_str.split(',')[0])
                else:
                    mri_date = pd.to_datetime(date_str.split(' ')[0])
                scar_df.iat[index, 1] = mri_date
                v = 9
        scar_df['MRI Date'] = pd.to_datetime(scar_df['MRI Date'])

        # Parse MUSE XML ECGs for patients in scar_df dataset. Each patient can have several ECGs.
        ecg_scar_meta_df = parse_muse_ecgs(scar_df)
        for region_name in ['Basal', 'Mid', 'Apical', 'Apex']:
            sum_ann = ecg_scar_meta_df[[col for col in ecg_scar_meta_df.columns if region_name in col]].sum(axis=1).values
            region = [1 if ann > 0 else 0 for ann in sum_ann]
            ecg_scar_meta_df[region_name] = region

        extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg_loc,
                                        ann_dir_path=GlobalPaths.pla_annotation_loc,
                                        metadata_path=GlobalPaths.ecg_meta_loc,
                                        verbose=True)
        extracted_segments_dict = extractor.extract_segments()
        # ecg_feature_ds = get_ecg_feature_dataset_several_visit(extracted_segments_dict)
        ecg_feature_ds = get_ecg_feature_ds_uncertainty(extracted_segments_dict)
        dataset = pd.merge(left=ecg_feature_ds, right=ecg_scar_meta_df, how="inner", on=['Record_ID', 'ECG_ID'])
        dataset.to_excel('Data/overall_ecg_feature_uncertain_scar_location.xlsx', index=False)

    region_name = 'Mid'
    # Keep only ECG features, their confidence, record ID, and the region name.
    dataset = dataset[[col for col in dataset.columns if ('(' in col and ')' in col) or col in [region_name, 'Record_ID', 'ECG_ID', 'ECG_Count']]]

    # Identify continuous features whose p-value (using T-test) between scar/no-scar is below 0.05.
    ttest_result = []
    for col in dataset.columns:
        if col not in ['Record_ID', 'ECG_ID', 'ECG_Count', region_name] and 'has_crossed' not in col and '_notches' not in col and '[conf]' not in col:
            _, p_value = ttest_ind(a=dataset.loc[dataset[region_name] == 0][col].values,
                                   b=dataset.loc[dataset[region_name] == 1][col].values,
                                   equal_var=False)
            ttest_result.append((col, p_value))
    ttest_result = sorted(ttest_result, key=lambda item: item[1])
    selected_continuous_features = [x[0] for x in ttest_result if x[1] < 0.05]

    # Identify discrete features whose information gain between scar/no-scar is above 0.05.
    dataset_discrete = dataset[[col for col in dataset.columns if ('has_crossed' in col or '_notches' in col) and '[conf]' not in col]]
    mi = mutual_info_classif(dataset_discrete.values, dataset[region_name].values, discrete_features=True)
    mi = list(zip(dataset_discrete.columns.values, mi))
    mi = sorted(mi, key=lambda x: x[1], reverse=True)
    selected_discrete_features = [x[0] for x in mi if x[1] > 0.05]

    selected_features = selected_continuous_features + selected_discrete_features
    selected_features_conf = [f'{feature}[conf]' for feature in selected_features]
    dataset = dataset[['Record_ID', 'ECG_ID', 'ECG_Count', region_name] + selected_features + selected_features_conf]

    pids = list(set(dataset['Record_ID'].values))
    for i in range(5):
        train_pids, test_pids = train_test_split(pids, test_size=0.2, shuffle=True)

        train = pd.merge(left=pd.DataFrame(train_pids, columns=['Record_ID']), right=dataset, how='inner', on=['Record_ID'])
        test = pd.merge(left=pd.DataFrame(test_pids, columns=['Record_ID']), right=dataset, how='inner', on=['Record_ID'])

        train_x_df = train[['ECG_Count'] + [col for col in train.columns if ('(' in col and ')' in col)]]
        feature_names = []
        for feature in train_x_df.columns.values:
            feature = feature.replace('[', '_')
            feature = feature.replace(']', '_')
            feature_names.append(feature)

        train_x = train_x_df.values
        train_y = train[region_name].values
        d_train = xgb.DMatrix(train_x, train_y, feature_names=feature_names)

        test_x_df = test[['ECG_Count'] + [col for col in test.columns if ('(' in col and ')' in col)]]
        test_x = test_x_df.values
        test_y = test[region_name].values
        d_test = xgb.DMatrix(test_x, test_y, feature_names=feature_names)

        # cl = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc')
        # grid_cv = GridSearchCV(cl, param_grid_first, n_jobs=-1, scoring='roc_auc', cv=5, verbose=1)
        # _ = grid_cv.fit(X=train_x, y=train_y)
        # print(f'First-level GridSearch Best Score = {grid_cv.best_score_}')
        # print(f'For Parameters:\n{grid_cv.best_params_}')
        # model = grid_cv.best_estimator_
        #
        # grid_cv = GridSearchCV(model, param_grid_second, n_jobs=-1, scoring='roc_auc', cv=5, verbose=1)
        # _ = grid_cv.fit(X=train_x, y=train_y)
        # print(f'Second-level GridSearch Best Score = {grid_cv.best_score_}')
        # print(f'For Parameters:\n{grid_cv.best_params_}')
        # model = grid_cv.best_estimator_

        # grid_cv = GridSearchCV(model, param_grid_third, n_jobs=-1, scoring='roc_auc', cv=5, verbose=1)
        # _ = grid_cv.fit(X=train_x, y=train_y)
        # print(f'Third-level GridSearch Best Score = {grid_cv.best_score_}')
        # print(f'For Parameters:\n{grid_cv.best_params_}\n')
        # model = grid_cv.best_estimator_

        # model = xgb.XGBClassifier(objective="binary:logistic",
        #                           eval_metric='auc',
        #                           n_estimators=1000,
        #                           learning_rate=0.3,
        #                           max_depth=10,
        #                           scale_pos_weight=0.8,
        #                           colsample_bytree=0.5,
        #                           subsample=1,
        #                           gamma=3,
        #                           reg_lambda=0)
        # model.fit(train_x, train_y)

        xgb_optimal_params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.3,
            'n_estimators': 1000,
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

        # preds = model.predict(test_x)
        # preds_prob = model.predict_proba(test_x)

        # print(classification_report(test_y, preds))

        prediction_df = pd.DataFrame(
            {
                'Record_ID': test['Record_ID'].values,
                'ECG_ID': test['ECG_ID'].values,
                'y_true': test_y,
                'y_pred': preds,
                # 'y_prob_0': preds_prob[:, 0],
                # 'y_prob_1': preds_prob[:, 1]
                'y_prob_0': [1 - x for x in preds_prob],
                'y_prob_1': preds_prob
            })

        prediction_df = pd.merge(left=prediction_df, right=test, how='inner', on=['Record_ID', 'ECG_ID'])

        # feature_important = model.get_booster().get_score(importance_type='gain')
        feature_important = model.get_score(importance_type='gain')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data.nlargest(40, columns="score").plot(kind='barh', figsize=(20, 10))
        # plt.show()  ## plot top 40 features

        v = 9

        y_true_list = []
        y_pred_list = []
        post_predict = []
        for pid in test_pids:
            y_true = prediction_df.loc[prediction_df['Record_ID'] == pid]['y_true'].values[0]
            y_pred_0_list = prediction_df.loc[prediction_df['Record_ID'] == pid]['y_prob_0'].values
            y_pred_1_list = prediction_df.loc[prediction_df['Record_ID'] == pid]['y_prob_1'].values

            y_pred_0_list = [x for x in y_pred_0_list if x >= 0.5]
            y_pred_1_list = [x for x in y_pred_1_list if x >= 0.5]

            if len(y_pred_0_list) == 0 or len(y_pred_1_list) == 0:
                continue

            pred_0_prob = statistics.mean(y_pred_0_list)
            pred_1_prob = statistics.mean(y_pred_1_list)
            if pred_0_prob > pred_1_prob:
                # y_pred_list.append(0)
                row = [pid, y_true, 0, pred_0_prob, pred_1_prob]
            else:
                row = [pid, y_true, 1, pred_0_prob, pred_1_prob]
                # y_pred_list.append(1)
            post_predict.append(row)
            # if len(y_pred_0_list) == 0 and len(y_pred_1_list) == 0:
            #     print(f'PID={pid} is predicted as unknown')
            #     continue
            # if len(y_pred_0_list) == 0 and len(y_pred_1_list) != 0:
            #     y_pred_list.append(1)
            # elif len(y_pred_0_list) != 0 and len(y_pred_1_list) == 0:
            #     y_pred_list.append(0)
            # else:
            #     pred_0_prob = statistics.mean(y_pred_0_list)
            #     pred_1_prob = statistics.mean(y_pred_1_list)
            #     if pred_0_prob > pred_1_prob:
            #         y_pred_list.append(0)
            #     else:
            #         y_pred_list.append(1)
            y_true_list.append(y_true)

        post_predict = pd.DataFrame(post_predict, columns=['Record_ID', 'y_true', 'y_pred', 'pred_0_prob', 'pred_1_prob'])
        temp_incorrect = post_predict.loc[post_predict['y_true'] != post_predict['y_pred']]
        temp_incorrect['confidence'] = abs(temp_incorrect['pred_0_prob'] - temp_incorrect['pred_1_prob'])
        temp_correct = post_predict.loc[post_predict['y_true'] == post_predict['y_pred']]
        temp_correct['confidence'] = abs(temp_correct['pred_0_prob'] - temp_correct['pred_1_prob'])

        target_df = post_predict.loc[post_predict['y_true'] == post_predict['y_pred']]
        incorrect_df = post_predict.loc[post_predict['y_true'] != post_predict['y_pred']]
        incorrect_df = incorrect_df.loc[abs(incorrect_df['pred_0_prob'] - incorrect_df['pred_1_prob']) > 0.1]
        target_df = pd.concat([target_df, incorrect_df], ignore_index=True)
        print(classification_report(target_df['y_true'].values, target_df['y_pred'].values))
        v = 9

        # y_true_list = []
        # y_pred_list = []
        # for pid in test_pids:
        #     y_true = prediction_df.loc[prediction_df['Record_ID'] == pid]['y_true'].values[0]
        #     y_true_list.append(y_true)
        #
        #     all_preds = list(prediction_df.loc[prediction_df['Record_ID'] == pid]['y_pred'].values)
        #     if len(all_preds) == 1:
        #         y_pred_list.append(all_preds[0])
        #     else:
        #         y_pred_list.append(max(set(all_preds), key=all_preds.count))
        #
        # print(classification_report(y_true_list, y_pred_list))



    v = 9

    # excluded_ecgs = pd.read_excel('Data/ECG/ScarECG/pids_with_far_mri.xlsx')
    # excluded_ecgs = pd.merge(left=excluded_ecgs, right=scar_df, how="inner", on=["Record_ID"])
    # v = 0
    #
    # try:
    #     ecg_feature_ds = pd.read_excel('cached_ecg_feature_only_dataset.xlsx')
    # except FileNotFoundError:
    #     extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg,
    #                                    ann_dir_path=GlobalPaths.pla_annotation,
    #                                    metadata_path=GlobalPaths.cached_scar_ecg_meta,
    #                                    verbose=True)
    #     extracted_segments_dict = extractor.extract_segments()
    #     ecg_feature_ds = get_ecg_feature_dataset(extracted_segments_dict)
    #     ecg_feature_ds.to_excel('cached_ecg_feature_only_dataset.xlsx', index=False)
    #
    # dataset = pd.merge(left=scar_df, right=ecg_feature_ds, how="inner", on=["Record_ID"])
    #
    # v = 9

    # predict_scar_grid_search_tree(region_name='Mid', select_top_features=True)
    # predict_scar_tree(region_name='Mid', select_top_features=True, use_augmentation=False)

