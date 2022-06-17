import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

import DataManagement as DM


class DatasetMaker:
    def __init__(self, ehr_df: pd.DataFrame, ecg_ds: np.ndarray, n_folds: int):
        if ehr_df.shape[0] != ecg_ds.shape[0]:
            assert 'EHR and ECG Dataset must have same number of data points'
        self.ehr_df = ehr_df
        self.ecg_ds = ecg_ds
        # -> Only for development !!!
        self.ehr_df = self.ehr_df.iloc[:50]
        self.ecg_ds = self.ecg_ds[:50]

        self.ehr_dataset = []
        self.ecg_dataset = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)
        for train_index, test_index in kf.split(self.ehr_df):
            ehr_train, ehr_test = self.ehr_df.iloc[train_index], self.ehr_df.iloc[test_index]
            ecg_train, ecg_test = self.ecg_ds[train_index], self.ecg_ds[test_index]
            self.ehr_dataset.append((ehr_train, ehr_test))
            self.ecg_dataset.append((ecg_train, ecg_test))

    def get_fold(self, fold_index: int, augment_mode: str = None):
        ehr_train, ehr_test = self.ehr_dataset[fold_index][0], self.ehr_dataset[fold_index][1]
        ecg_train, ecg_test = self.ecg_dataset[fold_index][0], self.ecg_dataset[fold_index][1]

        if augment_mode is not None:
            minority_ehr_train = ehr_train.loc[ehr_train['DE'] == 0]
            minority_ecg_train = np.array([qt_seg_object for qt_seg_object in ecg_train if qt_seg_object['de'] == 0])
            augmentor = DM.ScarAugmentor(ehr_df=minority_ehr_train, ecg_ds=minority_ecg_train)
            ehr_augmented, ecg_augmented = augmentor.smote_ehr_ecg(dist_mode=augment_mode)

            ehr_train = pd.concat([ehr_train, ehr_augmented], axis=0)
            ehr_train = ehr_train.sample(frac=1).reset_index(drop=True)
            ecg_train = np.concatenate((ecg_train, ecg_augmented))

        ehr_train = ehr_train.drop(columns='Reason for termination')
        ehr_test = ehr_test.drop(columns='Reason for termination')

        ehr_train, ehr_test = self._standardize_one_hot_ehr(ehr_train=ehr_train, ehr_test=ehr_test)

        x_train_ecg, x_train_ehr, y_train = self.get_numpy_x_y(ecg_ds=ecg_train, ehr_df=ehr_train)
        x_test_ecg, x_test_ehr, y_test = self.get_numpy_x_y(ecg_ds=ecg_test, ehr_df=ehr_test)

        return x_train_ecg, x_train_ehr, y_train, x_test_ecg, x_test_ehr, y_test

    @staticmethod
    def get_numpy_x_y(ecg_ds: np.ndarray, ehr_df: pd.DataFrame):
        x_ecg = []
        x_ehr = []
        y = []
        for ecg_object in ecg_ds:
            pid = ecg_object['pid']
            de = float(ecg_object['de'])
            ehr_row = list(ehr_df.loc[ehr_df['Record_ID'] == pid].values[0])
            qt_segments = ecg_object['preprocessed']
            qt_segment = qt_segments[random.randint(0, len(qt_segments) - 1)]

            qt_segment = np.array(qt_segment)
            if qt_segment.shape[1] != 4:
                qt_segment = np.transpose(qt_segment)
            ehr_row = np.array(ehr_row)

            x_ecg.append(qt_segment)
            x_ehr.append(ehr_row)
            y.append(de)
        x_ecg = np.array(x_ecg).astype('float32')
        x_ehr = np.array(x_ehr).astype('float32')
        y = np.array(y).astype('float32')
        return x_ecg, x_ehr, y

    @staticmethod
    def _standardize_one_hot_ehr(ehr_train: pd.DataFrame, ehr_test: pd.DataFrame):
        train_record_ids = ehr_train['Record_ID'].values
        test_record_ids = ehr_test['Record_ID'].values

        # -> Standardize Continuous Part of EHR DataFrame
        ehr_train_continuous = DM.EHRAttributeManager.get_continuous_df(df=ehr_train)
        ehr_test_continuous = DM.EHRAttributeManager.get_continuous_df(df=ehr_test)
        standard_scaler = StandardScaler()
        standard_scaler.fit(ehr_train_continuous)
        ehr_train_continuous = pd.DataFrame(data=standard_scaler.transform(ehr_train_continuous),
                                            columns=ehr_train_continuous.columns.values)
        ehr_test_continuous = pd.DataFrame(data=standard_scaler.transform(ehr_test_continuous),
                                           columns=ehr_test_continuous.columns.values)

        # -> One-Hot Encode Nominal Features
        train_nominal_df = DM.EHRAttributeManager.get_nominal_df(df=ehr_train)
        test_nominal_df = DM.EHRAttributeManager.get_nominal_df(df=ehr_test)

        one_hot_encoder = OneHotEncoder(drop='if_binary')
        one_hot_encoder.fit(pd.concat([train_nominal_df, test_nominal_df]))

        ehr_train_nominal = pd.DataFrame(one_hot_encoder.transform(train_nominal_df).toarray())
        ehr_test_nominal = pd.DataFrame(one_hot_encoder.transform(test_nominal_df).toarray())

        result_train = pd.concat([ehr_train_continuous, ehr_train_nominal], axis=1)
        result_test = pd.concat([ehr_test_continuous, ehr_test_nominal], axis=1)
        result_train['Record_ID'] = train_record_ids
        result_test['Record_ID'] = test_record_ids

        return result_train, result_test


if __name__ == '__main__':
    parser = DM.EHRECGParser()
    ds_maker = DatasetMaker(ehr_df=parser.ehr_df_imputed, ecg_ds=parser.qt_dataset, n_folds=5)
    x_train_ecg, x_train_ehr, y_train, x_test_ecg, x_test_ehr, y_test = ds_maker.get_fold(fold_index=0, augment_mode='ecg')

