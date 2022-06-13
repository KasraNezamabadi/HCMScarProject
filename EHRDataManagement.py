import random
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.stats import ttest_ind
from datetime import datetime
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from distython import HVDM
from dataclasses import dataclass
from collections import Counter
from time import time
import random
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb


class EHRDistance:
    @staticmethod
    def compute_hvdm_matrix(input_df: pd.DataFrame, y_ix: [int], cat_ix: [int]):
        print(f'Computing pair-wise HVDM distance for {input_df.shape[0]} data points')
        ehr_matrix = input_df.values
        # y_ix = [63]
        # cat_ix = list(range(28, 63))
        hvdm_metric = HVDM(X=ehr_matrix, y_ix=y_ix, cat_ix=cat_ix)

        dist_matrix = np.empty(shape=(len(ehr_matrix), len(ehr_matrix)), dtype=object)
        time_log = []
        for i in range(len(ehr_matrix)):
            for j in range(len(ehr_matrix)):
                if j > i:
                    s_time = time()
                    x = ehr_matrix[i]
                    y = ehr_matrix[j]
                    dist = hvdm_metric.hvdm(x=x, y=y, missing_values=None)
                    dist_matrix[i, j] = (dist, input_df.iloc[j]['Record_ID'], input_df.iloc[j])
                    e_time = time()
                    if len(time_log) < 3:
                        time_log.append(e_time - s_time)

                    if len(time_log) == 3:
                        elapsed_per_pair = statistics.mean(time_log)
                        total_time = round(len(ehr_matrix) * len(ehr_matrix) * 0.5 * elapsed_per_pair)
                        print(f'ETA = {total_time} seconds')
                        time_log.append(-1)

        print('Copying lower triangle')
        for i in range(len(ehr_matrix) - 1, 0, -1):
            for j in range(len(ehr_matrix)):
                if j < i:
                    dist_matrix[i, j] = (dist_matrix[j, i][0], input_df.iloc[j]['Record_ID'], input_df.iloc[j])
        return dist_matrix


@dataclass
class EHRType:
    biometric = 'Biometric'
    demographic = 'Demographic'
    exercise = 'Exercise Test'
    echo = 'ECHO'
    symptom = 'Symptom'
    history = 'History'
    family_history = 'Family History'
    drug = 'Therapy'
    other = 'Other'


@dataclass
class DataType:
    nominal = 'Nominal'
    continuous = 'Continuous'


class EHRObject:
    def __init__(self, ehr_type: str, data_type: str, record_name: str, description: str = None):
        self.ehr_type = ehr_type
        self.data_type = data_type
        self.record_name = record_name
        if description is None:
            self.description = self.record_name
        else:
            self.description = description

    def is_nominal(self) -> bool:
        if self.data_type == DataType.nominal:
            return True
        return False


@dataclass
class EHRAttributeManager:
    _ehr_attributes = [EHRObject(EHRType.demographic, DataType.nominal, 'Race'),
                       EHRObject(EHRType.demographic, DataType.continuous, 'AGE', 'Age (years)'),
                       EHRObject(EHRType.demographic, DataType.nominal, 'GENDER', 'Sex'),
                       EHRObject(EHRType.biometric, DataType.continuous, 'BMI', 'Body Mass Index (kg/m2)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'IVS_max', 'Maximum IVS thickness (mm)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'PW',
                                 'Maximum LV posterior wall thickness (mm)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'Apex', 'LV apical wall thickness (mm)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'LA', 'Left Atrial diameter (mm)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'EF', 'LV ejection fraction (%)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'LVEDV', 'LV end-diastolic volume (ml)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'LVESV', 'LV end-systolic volume (ml)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'MV_E',
                                 'Mitral valve early-stage flow velocity (m/s)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'MV_A',
                                 'Mitral valve late-stage flow velocity'),
                       EHRObject(EHRType.echo, DataType.continuous, 'E/A', 'LV diastolic function'),
                       EHRObject(EHRType.echo, DataType.continuous, 'MV_DecT',
                                 'Early mitral inflow deceleration time (ms)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'E/E\'', 'LV diastolic function'),
                       EHRObject(EHRType.echo, DataType.continuous, 'LVOT_Rest',
                                 'LV outflow tract gradient at rest (mmHg)'),
                       EHRObject(EHRType.echo, DataType.continuous, 'LVOT_Stress',
                                 'LV outflow tract gradient at peak stress (mmHg)'),
                       EHRObject(EHRType.echo, DataType.nominal, 'HCM type',
                                 'HCM type (obstructive, non-obstructive, labile obstructive)'),
                       EHRObject(EHRType.echo, DataType.nominal, 'SAM',
                                 'Systolic anterior motion of mitral valve (SAM)'),
                       EHRObject(EHRType.echo, DataType.nominal, 'MR', 'Severity of mitral regurgitation'),
                       EHRObject(EHRType.exercise, DataType.nominal, 'Protocol',
                                 'Exercise test, treadmill protocol'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'METS', 'Metabolic equivalents'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'HR_rest', 'Heart rate at rest (bpm)'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'HR_stress',
                                 'Heart rate at peak exercise (bpm)'),
                       EHRObject(EHRType.exercise, DataType.continuous, '%_of_MaxHr',
                                 'Peak heart rate achieved, expressed as % of MPHR (bpm)'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'HR_recovery_60s',
                                 'Heart rate after 1 min recovery (bpm)'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'SBP_rest', 'Systolic BP at rest (mmHg)'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'DBP_rest', 'Diastolic BP at rest (mmHg)'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'SBP_stress',
                                 'Systolic BP at peak exercise (mmHg)'),
                       EHRObject(EHRType.exercise, DataType.continuous, 'DBP_stress',
                                 'Diastolic BP at peak exercise (mmHg)'),
                       EHRObject(EHRType.exercise, DataType.nominal, 'Reason for termination',
                                 'Reason for terminating test'),
                       EHRObject(EHRType.exercise, DataType.nominal, 'EIH_SBP 210',
                                 'Systolic BP >210 mmHg at peak exercise'),
                       EHRObject(EHRType.exercise, DataType.nominal, 'EIH_DBP  90',
                                 'Diastolic BP >90 mmHg at peak exercise (mmHg)'),
                       EHRObject(EHRType.exercise, DataType.nominal, 'NYHA',
                                 'New York Heart Association Functional Classification'),
                       EHRObject(EHRType.symptom, DataType.nominal, 'ANGINA', 'Chest pain by reduced blood flow'),
                       EHRObject(EHRType.symptom, DataType.nominal, 'DYSPNEA at exertion',
                                 'Shortness of breath during physical activity'),
                       EHRObject(EHRType.symptom, DataType.nominal, 'DIZZINESS', 'Lightheadedness'),
                       EHRObject(EHRType.symptom, DataType.nominal, 'PRESYNCOPE', 'Feeling of about to faint'),
                       EHRObject(EHRType.symptom, DataType.nominal, 'SYNCOPE', 'Loss of consciousness'),
                       EHRObject(EHRType.drug, DataType.nominal, 'Bbloq', 'Beta blocker'),
                       EHRObject(EHRType.drug, DataType.nominal, 'Cabloq', 'Calcium channel blocker'),
                       EHRObject(EHRType.drug, DataType.nominal, 'ACEi_ARB',
                                 'ACE-inhibitor, angiotensin receptor blockade'),
                       EHRObject(EHRType.drug, DataType.nominal, 'Anti-plt', 'Antiplatelet medications'),
                       EHRObject(EHRType.drug, DataType.nominal, 'Diur', 'Diuretic'),
                       EHRObject(EHRType.drug, DataType.nominal, 'Disopyramide', 'Disopyramide'),
                       EHRObject(EHRType.drug, DataType.nominal, 'Insulin', 'Insulin'),
                       EHRObject(EHRType.drug, DataType.nominal, 'Statin', 'Statin'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_DM', 'History of diabetes'),
                       EHRObject(EHRType.history, DataType.nominal, 'Smoking', 'History of smoking'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_MI', 'History of myocardial infarction'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_CAD', 'History of obstructive CAD'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_HTN', 'History of hypertension'),
                       EHRObject(EHRType.history, DataType.nominal, 'ICDbase', 'History of  ICD implantation'),
                       EHRObject(EHRType.history, DataType.continuous, 'H_ICD Shock_N',
                                 'Number of ICD shock in past'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_Afib', 'History of AFib'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_NSVT', 'History of NSVT'),
                       EHRObject(EHRType.history, DataType.continuous, 'H_NSVT_N', 'Number of NSVT in past'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_VT/VF', 'History of VT/VF'),
                       EHRObject(EHRType.history, DataType.continuous, 'H_VT/VF_N', 'Number of VT/VF in past'),
                       EHRObject(EHRType.history, DataType.nominal, 'H_ATP', 'History of Antitachycardia pacing'),
                       EHRObject(EHRType.family_history, DataType.nominal, 'FHx_HCM', 'Family history of HCM'),
                       EHRObject(EHRType.family_history, DataType.nominal, 'FHx_SCD', 'Family history of SCD')]

    @classmethod
    def get_all_attrs(cls, include_record_id: bool = True):
        if include_record_id:
            result = ['Record_ID'] + [ehr_object.record_name for ehr_object in cls._ehr_attributes]
        else:
            result = [ehr_object.record_name for ehr_object in cls._ehr_attributes]
        return result

    @classmethod
    def get_nominal_attrs(cls, include_record_id: bool = True):
        if include_record_id:
            result = ['Record_ID'] + [ehr_object.record_name for ehr_object in cls._ehr_attributes if ehr_object.data_type == DataType.nominal]
        else:
            result = [ehr_object.record_name for ehr_object in cls._ehr_attributes if ehr_object.data_type == DataType.nominal]
        return result

    @classmethod
    def get_continuous_attrs(cls, include_record_id: bool = True):
        if include_record_id:
            result = ['Record_ID'] + [ehr_object.record_name for ehr_object in cls._ehr_attributes if ehr_object.data_type == DataType.continuous]
        else:
            result = [ehr_object.record_name for ehr_object in cls._ehr_attributes if ehr_object.data_type == DataType.continuous]
        return result

    @classmethod
    def get_attr_object(cls, for_record_name: str):
        for ehr_object in cls._ehr_attributes:
            if ehr_object.record_name == for_record_name:
                return ehr_object
        return None

    @classmethod
    def get_nominal_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        s1 = set(df.columns.values)
        s2 = set(cls.get_nominal_attrs(include_record_id=False))
        s3 = s1 & s2
        selected_cols = list(s3)
        result = df.reindex(columns=selected_cols)
        return result

    @classmethod
    def get_continuous_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        s1 = set(df.columns.values)
        s2 = set(cls.get_continuous_attrs(include_record_id=False))
        s3 = s1 & s2
        selected_cols = list(s3)
        result = df.reindex(columns=selected_cols)
        return result


class EHRScarParser:
    def __init__(self, path_to_ehr: str, path_to_mri: str, path_to_ecg: str, num_patients: int = None):
        print('Start Parsing')
        self.ehr_df = pd.read_excel(path_to_ehr)
        self.mri_df = pd.read_excel(path_to_mri)
        self.ecg_df = pd.read_excel(path_to_ecg)
        if num_patients is not None:
            print(f'Working wih only {num_patients} patients')
            self.ehr_df = self.ehr_df.iloc[:num_patients]
            self.mri_df = self.mri_df.iloc[:num_patients]
        print('Datasets Fetched')

        # -> Init Actions:
        self.mri_df = self.mri_df.loc[(self.mri_df['DE'] == 0) | (self.mri_df['DE'] == 1)]
        print('\nCleaning Date Attributes')
        self._clean_date_attributes()
        print('\nFiltering out patients with no MRI, no EHR, and not ECG')
        self._join_with_mri()

        # -> Keep in EHR dataset only the attributes indicated in categorical and continuous lists above
        self.ehr_df = self.ehr_df.reindex(columns=EHRAttributeManager.get_all_attrs())

        print('\nImputing missing values in EHR dataset ...')
        try:
            self.ehr_df_imputed = pd.read_excel('Data/CachedEHR/ehr_df.xlsx')
            print('--- Fetched Cached Imputed Data Frame')
        except FileNotFoundError:
            print('--- No cache found. Perform Imputation')
            self._prepare_ehr_for_imputation()
            self.ehr_df_imputed = self._impute_ehr_df()
        self.ehr_df_imputed = self.ehr_df_imputed.reset_index(drop=True)
        self.mri_df = self.mri_df.reset_index(drop=True)
        print('\nParser Ready!')

    def get_categorical_ehr_df(self) -> pd.DataFrame:
        return self.ehr_df_imputed.reindex(columns=EHRAttributeManager.get_nominal_attrs())

    def get_continuous_ehr_df(self) -> pd.DataFrame:
        return self.ehr_df_imputed.reindex(columns=EHRAttributeManager.get_continuous_attrs())

    def get_scar_labels(self) -> pd.DataFrame:
        return self.mri_df.reindex(columns=['Record_ID', 'DE'])

    def _clean_date_attributes(self):
        """
        Clean all date-related attributes in `self.ehr_df`, `self.mri_df`, and `self.ecg_df` using pd.to_datetime.
        If exception is raised either try to convert manually or replace with `np.datetime64('NAT')`

        - If `self.ehr_df`:
            - Step 1 -> Directly convert `Myectomy_date` column to datetime
            - Step 2 -> Loop through `ASA_date` column and try to convert to datetime; if exception is raised, replace
            with NaT.
        - If `self.mri_df`: Loop through `MRI Date` column and manually convert to datetime.
        - If `self.ecg_df`: Directly convert `Acquisition Date` column to datetime.

        Lastly, sort all data frames based on `Record_ID` (or `PatientID`) and reset the index.

        """

        self.ehr_df['Myectomy_date'] = pd.to_datetime(self.ehr_df['Myectomy_date'])
        for index, row in self.ehr_df.iterrows():
            try:
                temp = row['ASA_date']
                _ = pd.to_datetime(temp)
            except:
                self.ehr_df.loc[index, 'ASA_date'] = np.datetime64('NAT')

        for index, row in self.mri_df.iterrows():
            try:
                self.mri_df.loc[index, 'MRI Date'] = pd.to_datetime(row['MRI Date'])
            except:
                date_str = str(row['MRI Date'])
                if ',' in date_str:
                    date = date_str.split(',')[0]
                    self.mri_df.loc[index, 'MRI Date'] = pd.to_datetime(date)
                elif '.' in date_str:
                    date_str = date_str.replace('.', '')
                    self.mri_df.loc[index, 'MRI Date'] = pd.to_datetime(date_str)
                else:
                    date = date_str.split(' ')[0]
                    self.mri_df.loc[index, 'MRI Date'] = pd.to_datetime(date)

        self.ecg_df['Acquisition Date'] = pd.to_datetime(self.ecg_df['Acquisition Date'])

        self.ehr_df = self.ehr_df.sort_values('Record_ID')
        self.ehr_df = self.ehr_df.reset_index(drop=True)
        self.mri_df = self.mri_df.sort_values('Record_ID')
        self.mri_df = self.mri_df.reset_index(drop=True)
        self.ecg_df = self.ecg_df.sort_values(['PatientID', 'Acquisition Date'])
        self.ecg_df = self.ecg_df.reset_index(drop=True)

    def _join_with_mri(self):
        """
        The MRI dataset determines what patients had MRI. These patients must have EHR and ECG data. \
        For ECG, the ECG must have been taken before Myectomy and ASA.
            - Step 1 -> Exclude patients who do not have EHR data in `self.ehr_df`.
            - Step 2 -> Exclude patients who do not have ECG at all or no ECG after filtering out those taken after Myectomy and/or ASA.
            - Step 3 -> Drop the patients found in Step 1 and 2 from MRI dataset.
            - Step 4 -> Keep in EHR dataset only the patients who are in MRI dataset.
        """
        missing_ehr = []
        index_to_drop = []
        for index, row in self.mri_df.iterrows():
            pid = int(row['Record_ID'])

            # -> Step 1
            ehr_target_df = self.ehr_df.loc[self.ehr_df['Record_ID'] == pid]
            if ehr_target_df.empty:
                index_to_drop.append(index)
                missing_ehr.append(pid)
                continue

            # -> Step 2
            ecg_target_df = self.ecg_df.loc[self.ecg_df['PatientID'] == pid]
            Myectomy_date = ehr_target_df['Myectomy_date'].values[0]
            ASA_date = ehr_target_df['ASA_date'].values[0]

            if not np.isnat(Myectomy_date):
                ecg_target_df = ecg_target_df.loc[ecg_target_df['Acquisition Date'] < Myectomy_date]

            if ASA_date is not None and ASA_date != 0:
                try:
                    ecg_target_df = ecg_target_df.loc[ecg_target_df['Acquisition Date'] < ASA_date]
                except:
                    pass

            if ecg_target_df.shape[0] == 0:
                index_to_drop.append(index)

        # -> Step 3
        self.mri_df = self.mri_df.drop(index_to_drop)
        # -> Step 4
        pids = self.mri_df['Record_ID'].values
        self.ehr_df = self.ehr_df[self.ehr_df['Record_ID'].isin(pids)]

    def _prepare_ehr_for_imputation(self):
        """
        Prepare EHR dataset (`self.ehr_df`) for KNN imputation.
        - Step 1 -> Remove rows (i.e., patient records) who has more than 60% of attributes as NaN.
        - Step 2 -> Try to convert every single entry in `self.ehr_df` to float. When fails:
            - If the entry is NaN or Datetime: leave it.
            - If the entry has invalid characters: try to manually convert. If not possible, remove from EHR dataset.
        - Step 3 -> Keep in MRI dataset only the patients who are in EHR dataset.
        """
        index_to_drop = []
        for index, row in self.ehr_df.iterrows():
            # -> Step 1
            nan_count = row.isna().sum()
            ratio = nan_count / len(row)
            if ratio > 0.6:
                index_to_drop.append(index)

            # -> Step 2
            for attribute_name, value in row.iteritems():
                try:
                    _ = float(value)
                except (TypeError, ValueError):
                    if not pd.isna(value) and not isinstance(value, datetime):
                        if value == '.' or value == '?':
                            self.ehr_df.loc[index, attribute_name] = np.nan
                        else:
                            try:
                                value = str(value)
                                if ',' in value:
                                    value = value.replace(' ', '')
                                    extended_termination_reason = {'0,1': 12,
                                                                   '0,3': 13,
                                                                   '1,3': 14,
                                                                   '1,10': 15,
                                                                   '2,3': 16,
                                                                   '3,4': 17,
                                                                   '3,7': 18,
                                                                   '3,8': 19,
                                                                   '3,11': 20,
                                                                   '4,8': 21,
                                                                   '0,1,3': 22}
                                    try:
                                        val = extended_termination_reason[value]
                                        self.ehr_df.loc[index, attribute_name] = val
                                    except KeyError:
                                        index_to_drop.append(index)
                                else:
                                    index_to_drop.append(index)
                            except TypeError:
                                index_to_drop.append(index)

        self.ehr_df = self.ehr_df.drop(index_to_drop)
        # -> Step 3
        pids = self.ehr_df['Record_ID'].values
        self.mri_df = self.mri_df[self.mri_df['Record_ID'].isin(pids)]

    def _impute_ehr_df(self, k_nearest: int = 5) -> pd.DataFrame:
        """
        Imputation using KNNImputer from sklearn.impute.
            - Step 1 -> Convert `self.ehr_df` to 2D matrix (n_samples x n_features) for KNNImputer. IMPORTANT: Remove
            Record_ID attribute.
            - Step 2 -> Impute all variables (continuous and categorical) using KNNImputer.
            - Step 3 -> Convert back the imputed matrix to data frame. Add Record_ID back to the df.
            - Step 4 -> For categorical variables, the imputation averages the k closest neighbors' corresponding values. \
            I then round the resulting average (e.g., closest five for smoking: 1, 1, 0, 0, 2 -> mean = 0.8 -> round = 1).

        Parameters
        ----
        - `k_nearest`: integer value for k in KNN algorithm. Default = 5.

        Returns
        -------
        Imputed pd.DataFrame with the same shape as input.
        :rtype pd.DataFrame
        """
        # -> Step 1
        temp_mri_df = self.get_scar_labels()
        temp_ehr_df = pd.merge(left=self.ehr_df, right=temp_mri_df, on='Record_ID')
        temp_ehr_df = temp_ehr_df.reset_index(drop=True)
        record_ids = temp_ehr_df['Record_ID'].values
        temp_ehr_df = temp_ehr_df.drop(columns=['Record_ID'])
        rearranged_cols = EHRAttributeManager.get_continuous_attrs(include_record_id=False) + \
                          EHRAttributeManager.get_nominal_attrs(include_record_id=False) + \
                          ['DE']
        temp_ehr_df = temp_ehr_df[rearranged_cols]
        dist_matrix = Distance.compute_hvdm_matrix(input_df=temp_ehr_df, y_ix=[63], cat_ix=list(range(28, 63)))

        # -> Step 2
        for index, row in temp_ehr_df.iterrows():
            if row.isna().any():
                # -> Sort ascending all neighbors based on distance
                temp = dist_matrix[index, :]
                temp = np.array([x for x in temp if not pd.isna(x)], dtype=object)
                sorted_nn = sorted(temp, key=lambda item: item[0])
                missing_series = row[row.isna()]
                for missing_attr, _ in missing_series.iteritems():
                    missing_attr_object = EHRAttributeManager.get_attr_object(for_record_name=missing_attr)
                    # -> Get k nearest neighbors that have the missing attribute NOT nan
                    nn_list = []
                    for nn in sorted_nn:
                        neighbor = nn[1]
                        if not pd.isna(neighbor[missing_attr_object.record_name]):
                            nn_list.append(neighbor)
                        if len(nn_list) == k_nearest:
                            break
                    if missing_attr_object.is_nominal():  # -> vote among nearest neighbors
                        nn_attr_counter = Counter([int(neighbor[missing_attr_object.record_name]) for neighbor in nn_list])
                        most_common = nn_attr_counter.most_common(n=1)
                        voted_value = most_common[0][0]
                    else:  # -> average among nearest neighbors
                        voted_value = statistics.mean([neighbor[missing_attr_object.record_name] for neighbor in nn_list])
                        if missing_attr_object.record_name == 'H_NSVT_N' or missing_attr_object.record_name == 'H_VT/VF_N':
                            voted_value = round(voted_value)
                    temp_ehr_df.loc[index, missing_attr_object.record_name] = voted_value

        # -> Step 3
        temp_ehr_df['Record_ID'] = record_ids
        temp_ehr_df.to_excel('Data/CachedEHR/ehr_df.xlsx')
        return temp_ehr_df


class ScarEHRFeatureSelection:
    def __init__(self, ehr_df: pd.DataFrame):
        self.ehr_df = ehr_df

    def _get_x_y(self, df: pd.DataFrame):
        # TODO -> Complete the document
        """
        Accepts a data frame and returns a 2D matrix `x`
        """
        y = df['DE'].values
        df = df.drop(columns=['DE', 'Record_ID'])
        x = df.values
        features = df.columns.values
        x = x.astype(dtype=int)
        y = y.astype(dtype=float)
        return x, y, features

    def compute_information_gain(self, for_discrete_vars: bool):
        if for_discrete_vars:
            x, y, feature_labels = self._get_x_y(
                df=self.ehr_df.reindex(columns=['DE'] + EHRAttributeManager.get_nominal_attrs()))
        else:
            x, y, feature_labels = self._get_x_y(
                df=self.ehr_df.reindex(columns=['DE'] + EHRAttributeManager.get_continuous_attrs()))
        mi = mutual_info_classif(X=x, y=y, discrete_features=for_discrete_vars)
        mi_features = list(zip(feature_labels, mi))
        mi_features = sorted(mi_features, reverse=True, key=lambda item: item[1])
        return mi_features

    def compute_welch_t_test(self):
        continuous_df = self.ehr_df.reindex(columns=['DE'] + EHRAttributeManager.get_continuous_attrs())
        continuous_df_class_0 = continuous_df.loc[continuous_df['DE'] == 0]
        continuous_df_class_1 = continuous_df.loc[continuous_df['DE'] == 1]
        continuous_df_class_0 = continuous_df_class_0.drop(columns=['DE', 'Record_ID'])
        continuous_df_class_1 = continuous_df_class_1.drop(columns=['DE', 'Record_ID'])
        feature_labels = continuous_df_class_0.columns.values
        p_values = []
        t_values = []
        for feature in feature_labels:
            sample_0 = continuous_df_class_0[feature].values
            sample_1 = continuous_df_class_1[feature].values
            t_value, p_value = ttest_ind(a=sample_0, b=sample_1, equal_var=False)
            p_values.append(p_value)
            t_values.append(t_value)
        ttest_features = list(zip(feature_labels, p_values))
        ttest_features = sorted(ttest_features, key=lambda item: item[1])
        return ttest_features

    # def compute_chi_square(self):
    #     x, y, feature_labels = self._get_x_y(df=self.categorical_df)
    #     chi_test = SelectKBest(score_func=chi2, k='all').fit(x, y)
    #     p_values = chi_test.pvalues_
    #     chi_features = list(zip(feature_labels, p_values))
    #     chi_features = sorted(chi_features, key=lambda item: item[1])
    #     return chi_features

    @staticmethod
    def plot_feature_score(feature_score_pair: [(str, float)], y_title: str, y_limit: float = None, save_path: str = None):
        features = [x[0] for x in feature_score_pair]
        score = [x[1] for x in feature_score_pair]
        x_pos = np.arange(len(features))

        plt.figure(figsize=(20, 5))
        plt.bar(x_pos, score, align='center')
        if y_limit is not None:
            plt.axhline(y=y_limit, color='r', linestyle='-')
        plt.xticks(x_pos, features, rotation=30)
        plt.ylabel(y_title)
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        plt.show()

    @staticmethod
    def plot_feature_score_selected(feature_score_pair: [(str, float)], feature_score_pair_selected: [(str, float)],
                                    y_title: str, y_limit: float = None, title: str = None, save_path: str = None):

        fig, ax = plt.subplots(2, figsize=(20, 10))

        features = [x[0] for x in feature_score_pair]
        score = [x[1] for x in feature_score_pair]
        x_pos = np.arange(len(features))
        ax[0].bar(x_pos, score, align='center')
        if y_limit is not None:
            ax[0].axhline(y=y_limit, color='r', linestyle='-')
        ax[0].set_xticklabels(features, rotation=30)
        ax[0].set_ylabel(y_title)

        features = [x[0] for x in feature_score_pair_selected]
        score = [x[1] for x in feature_score_pair_selected]
        x_pos = np.arange(len(features))
        ax[1].bar(x_pos, score, align='center')
        ax[1].set_xticklabels(features, rotation=30)
        ax[1].set_ylabel(y_title)

        plt.savefig(save_path, dpi=200)
        plt.show()


class ScarEHRAugmentor:
    def __init__(self, ehr_df: pd.DataFrame):
        self.ehr_df = ehr_df
        self.cache_path = 'Data/CachedEHR'

    def perform_smote(self, k_nearest: int = 5) -> pd.DataFrame:
        print('Performing Augmentation')
        try:
            print('Try fetching full distance matrix from cache')
            s_time = time()
            dist_matrix = np.load(os.path.join(self.cache_path, 'full_dist_matrix.npy'), allow_pickle=True)
            e_time = time()
            print(f'Distance matrix fetched from cache in {round(e_time - s_time)} seconds')
        except OSError:
            print('No cached Distance matrix found. Computing from scratch ...')
            dist_matrix = Distance.compute_hvdm_matrix(input_df=self.ehr_df, y_ix=[0], cat_ix=list(range(1, 13)))
            # print(f'Saving Full Distance Matrix at {self.cache_path}')
            # np.save(file=os.path.join(self.cache_path, 'full_dist_matrix'), arr=dist_matrix)

        augmented_list = []

        for index, row in self.ehr_df.iterrows():
            if row['DE'] == 1:
                continue
            neighbour_distances = dist_matrix[index, :]
            neighbour_distances = np.array([x for x in neighbour_distances if not pd.isna(x)], dtype=object)
            sorted_nn = sorted(neighbour_distances, key=lambda item: item[0])
            nn_list = []
            for nn in sorted_nn:
                dist = nn[0]
                neighbor = nn[1]
                if neighbor['DE'] == 0:
                    nn_list.append((dist, neighbor))
                if len(nn_list) == k_nearest:
                    break

            upper_bound = 1
            coin = random.uniform(0, 1)
            if coin >= 0.5:
                upper_bound = 2
            for _ in range(upper_bound):
                ix = random.randint(0, k_nearest-1)
                selected_neighbor = nn_list[ix][1]
                selected_dist = nn_list[ix][0]
                alpha = random.uniform(0, 1)
                augmented_dict = {}
                for attribute_name, value in row.iteritems():
                    if attribute_name == 'DE':
                        augmented_dict['DE'] = 0
                    else:
                        attr_object = EHRAttributeManager.get_attr_object(for_record_name=attribute_name)
                        if attr_object.is_nominal():
                            category_main = value
                            category_neighbor = selected_neighbor[attr_object.record_name]
                            if category_main == category_neighbor:
                                augmented_dict[attr_object.record_name] = category_main
                            else:
                                nn_attr_counter = Counter(
                                    [int(neighbor[1][attr_object.record_name]) for neighbor in nn_list])
                                most_common = nn_attr_counter.most_common(n=1)
                                voted_category = most_common[0][0]
                                augmented_dict[attr_object.record_name] = voted_category
                        else:
                            value_main = value
                            augmented_value = value_main + (alpha * selected_dist)
                            augmented_dict[attr_object.record_name] = augmented_value
                augmented_list.append(augmented_dict)

        augmented_df = pd.DataFrame(augmented_list)
        print('Augmentation Done!')
        return augmented_df


if __name__ == '__main__':

    path_to_ehr = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/Overall.xlsx'
    path_to_mri = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/HCM_MRI_Database_03022022.xlsx'
    path_to_ecg = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/ECGMeta.xlsx'

    # -> Load dataset and impute missing values. Result will be in ehr_df (pd.DataFrame)
    parser = EHRScarParser(path_to_ehr=path_to_ehr, path_to_mri=path_to_mri, path_to_ecg=path_to_ecg)
    categorical_df = parser.get_categorical_ehr_df()
    continuous_df = parser.get_continuous_ehr_df()
    scar_df = parser.get_scar_labels()

    ehr_df = pd.merge(categorical_df, continuous_df, on='Record_ID')
    ehr_df = pd.merge(ehr_df, scar_df, on='Record_ID')

    # -> Feature Selection. Result will be in selected_features (list of str)
    scar_feature_selection = ScarEHRFeatureSelection(ehr_df=ehr_df)
    mi_features_discrete = scar_feature_selection.compute_information_gain(for_discrete_vars=True)
    ttest_features_continuous = scar_feature_selection.compute_welch_t_test()

    scar_feature_selection.plot_feature_score(feature_score_pair=mi_features_discrete, y_limit=0.002, y_title='MI')
    scar_feature_selection.plot_feature_score(feature_score_pair=ttest_features_continuous, y_limit=0.05, y_title='P-value')

    selected_features_discrete = [x[0] for x in mi_features_discrete if x[1] >= 0.002]
    selected_features_continuous = [x[0] for x in ttest_features_continuous if x[1] <= 0.05]
    selected_features = selected_features_discrete + selected_features_continuous

    cols = list(ehr_df)
    cols.insert(1, cols.pop(cols.index('DE')))
    ehr_df = ehr_df.loc[:, cols]
    ehr_df = ehr_df.reindex(columns=['DE'] + selected_features)

    ehr_df = ehr_df.sample(frac=1).reset_index(drop=True)  # -> Do this one time only for now. You will cache the augmentation result

    df_train = ehr_df.iloc[:round(ehr_df.shape[0] * 0.8)]
    df_test = ehr_df.iloc[round(ehr_df.shape[0] * 0.8):]

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    minority_class_shape = df_train.loc[df_train['DE'] == 0].shape
    majority_class_shape = df_train.loc[df_train['DE'] == 1].shape
    print(f'\nOriginal Train Set DE=0: {minority_class_shape} | DE=1: {majority_class_shape}')

    scar_ehr_augmentor = ScarEHRAugmentor(ehr_df=df_train)
    augmented_df = scar_ehr_augmentor.perform_smote()
    df_train_augmented = pd.concat([df_train, augmented_df], axis=0)
    df_train_augmented = df_train_augmented.sample(frac=1).reset_index(drop=True)

    minority_class_shape = df_train_augmented.loc[df_train_augmented['DE'] == 0].shape
    majority_class_shape = df_train_augmented.loc[df_train_augmented['DE'] == 1].shape
    print(f'\nAugmented Train Set DE=0: {minority_class_shape} | DE=1: {majority_class_shape}')

    df_train_augmented.to_excel("Data/CachedEHR/AugmentedTrainSet_V1.xlsx", index=False)
    df_test.to_excel("Data/CachedEHR/TestSet_V1.xlsx", index=False)


    print('Done!')


    #
    # X_train, y_train = df_train_augmented.iloc[:, 1:], df_train_augmented.iloc[:, 0].values
    # X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0].values









