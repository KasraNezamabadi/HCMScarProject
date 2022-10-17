import os
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as P
import scipy
import scipy.stats
from scipy.stats import norm, ttest_ind
from matplotlib import colors
from datetime import datetime
from datetime import timedelta
from time import time
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from distython import HVDM
from dataclasses import dataclass
from collections import Counter
import random
import xml.parsers.expat
import xmltodict
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance
from tslearn.barycenters import softdtw_barycenter

import MuseXMLX
from MuseXMLX import MuseXmlParser
from QTSegmentExtractor import QTSegmentExtractor
from Utility import Util, SignalProcessing
import GlobalPaths


class Distance:
    @staticmethod
    def compute_hvdm_matrix(input_df: pd.DataFrame, y_ix: [int], cat_ix: [int]):
        print(f'--- Computing pair-wise HVDM distance for {input_df.shape[0]} data points')
        ehr_matrix = input_df.values
        # TODO -> THis function is used by two classes. Currently, it assumes that Record_ID is the first column,
        #  which is true for augmentation. But for imputation you need to bring record_id to the first column.
        if list(input_df.columns.values)[0] != 'Record_ID':
            assert 'Record_ID must be the first column in EHR data frame'
        ehr_matrix = ehr_matrix[:, 1:]  # Remove Record_ID which is the first column!
        hvdm_metric = HVDM(X=ehr_matrix, y_ix=y_ix, cat_ix=cat_ix)

        dist_matrix = np.empty(shape=(len(ehr_matrix), len(ehr_matrix)), dtype=object)
        time_log = []
        for i in range(len(ehr_matrix)):
            for j in range(len(ehr_matrix)):
                if j > i:
                    s_time = time()
                    x = ehr_matrix[i]
                    y = ehr_matrix[j]
                    dist = hvdm_metric.hvdm(x=x, y=y)
                    dist_matrix[i, j] = (dist, input_df.iloc[j]['Record_ID'], input_df.iloc[j])
                    e_time = time()
                    if len(time_log) < 3:
                        time_log.append(e_time - s_time)

                    if len(time_log) == 3:
                        elapsed_per_pair = statistics.mean(time_log)
                        total_time = round(len(ehr_matrix) * len(ehr_matrix) * 0.5 * elapsed_per_pair)
                        print(f'ETA = {total_time} seconds')
                        time_log.append(-1)

        for i in range(len(ehr_matrix) - 1, 0, -1):
            for j in range(len(ehr_matrix)):
                if j < i:
                    dist_matrix[i, j] = (dist_matrix[j, i][0], input_df.iloc[j]['Record_ID'], input_df.iloc[j])
        return dist_matrix

    @staticmethod
    def compute_ecg_distance_matrix(ecg_ds: np.ndarray):
        print(f'--- Computing pair-wise ECG distance for {ecg_ds.shape[0]} data points')
        dist_matrix = np.empty(shape=(len(ecg_ds), len(ecg_ds), 4), dtype=object)
        for i in range(len(ecg_ds)):
            for j in range(len(ecg_ds)):
                if j > i:
                    for lead_index in range(4):
                        dist = Util.get_qt_distance(p1=ecg_ds[i], p2=ecg_ds[j], lead_index=lead_index)
                        dist_matrix[i, j, lead_index] = (ecg_ds[j]['pid'], dist, ecg_ds[j]['preprocessed'])

        for i in range(len(ecg_ds) - 1, 0, -1):
            for j in range(len(ecg_ds)):
                if j < i:
                    for lead_index in range(4):
                        dist_matrix[i, j, lead_index] = (
                        ecg_ds[j]['pid'], dist_matrix[j, i, lead_index][1], ecg_ds[j]['preprocessed'])

        return dist_matrix


class EHRECGDistanceObject:
    def __init__(self, other_pid: int, other_ehr_object, other_ecg_object,
                 ehr_dist_norm: float = None,  ehr_dist: float = None,
                 ecg_dist_norm: float = None, ecg_dist: float = None):

        self.ehr_dist_norm = ehr_dist_norm
        self.ecg_dist_norm = ecg_dist_norm
        self.ehr_dist = ehr_dist
        self.ecg_dist = ecg_dist
        self.other_pid = other_pid
        self.other_ehr_object = other_ehr_object
        self.other_ecg_object = other_ecg_object


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


class ECGObject:
    def __init__(self, record_id: int, date: pd.Timestamp, frequency: int, ecg_df: pd.DataFrame):
        self.record_id = record_id
        self.date = date
        self.frequency = frequency
        self.ecg_df = ecg_df
        self.mri_date_diff = None


class EHRBaseParser:
    def __init__(self):
        self.ehr_df = pd.read_excel(GlobalPaths.ehr)
        self.mri_df = pd.read_excel(GlobalPaths.mri)
        self.ehr_df_imputed = pd.DataFrame()

        self.mri_df = self.mri_df.loc[(self.mri_df['DE'] == 0) | (self.mri_df['DE'] == 1)]
        self._join_mri_ehr()
        self._clean_date_attributes()
        print('\nBase Parser Ready!'
              '\n--- Fetch Whole MRI Dataset -> self.get_mri_whole()'
              '\n--- Fetch Whole EHR Dataset -> self.get_ehr_whole()')

    def _clean_date_attributes(self):
        """
        Clean all date-related attributes in `self.ehr_df`, `self.mri_df` using pd.to_datetime.
        If exception is raised either try to convert manually or replace with `np.datetime64('NAT')`

        - If `self.ehr_df`:
            - Step 1 -> Directly convert `Myectomy_date` column to datetime
            - Step 2 -> Loop through `ASA_date` column and try to convert to datetime; if exception is raised, replace
            with NaT.
        - If `self.mri_df`: Loop through `MRI Date` column and manually convert to datetime.
        Lastly, sort all data frames based on `Record_ID` and reset the index.
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

        self.ehr_df = self.ehr_df.sort_values('Record_ID')
        self.ehr_df = self.ehr_df.reset_index(drop=True)
        self.mri_df = self.mri_df.sort_values('Record_ID')
        self.mri_df = self.mri_df.reset_index(drop=True)

    def _join_mri_ehr(self):
        """
        The MRI dataset determines what patients had MRI. These patients must have EHR.
            - Step 1 -> Exclude patients who do not have EHR data in `self.ehr_df`.
            - Step 2 -> Drop the patients found in Step 1 from MRI dataset.
            - Step 3 -> Keep in EHR dataset only the patients who are in MRI dataset.
        """
        index_to_drop = []
        for index, row in self.mri_df.iterrows():
            pid = int(row['Record_ID'])
            # -> Step 1
            ehr_target_df = self.ehr_df.loc[self.ehr_df['Record_ID'] == pid]
            if ehr_target_df.empty:
                index_to_drop.append(index)
                continue
        # -> Step 3
        self.mri_df = self.mri_df.drop(index_to_drop)
        # -> Step 4
        pids = self.mri_df['Record_ID'].values
        self.ehr_df = self.ehr_df[self.ehr_df['Record_ID'].isin(pids)]

    def get_scar_labels(self) -> pd.DataFrame:
        return self.mri_df.reindex(columns=['Record_ID', 'DE'])

    def prepare_ehr_for_imputation(self):
        """
        Prepare EHR dataset (`self.ehr_df`) for KNN imputation.
        - Step 1 -> Remove rows (i.e., patient records) who has more than 60% of attributes as NaN.
        - Step 2 -> Try to convert every single entry in `self.ehr_df` to float. When fails:
            - If the entry is NaN or Datetime: leave it.
            - If the entry has invalid characters: try to manually convert. If not possible, remove from EHR dataset.
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

    def impute_ehr_df(self, k_nearest: int = 5) -> pd.DataFrame:
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
        temp_mri_df = self.mri_df[['Record_ID', 'DE']]
        temp_ehr_df = pd.merge(left=self.ehr_df, right=temp_mri_df, on='Record_ID')
        temp_ehr_df = temp_ehr_df.reset_index(drop=True)
        # record_ids = temp_ehr_df['Record_ID'].values
        # temp_ehr_df = temp_ehr_df.drop(columns=['Record_ID'])
        rearranged_cols = EHRAttributeManager.get_continuous_attrs(include_record_id=False) + \
                          EHRAttributeManager.get_nominal_attrs(include_record_id=False) + \
                          ['DE', 'Record_ID']
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
                        neighbor = nn[2]  # -> contains the EHR row of the neighbor
                        if not pd.isna(neighbor[missing_attr_object.record_name]):
                            nn_list.append(neighbor)
                        if len(nn_list) == k_nearest:
                            break
                    if missing_attr_object.is_nominal():  # -> attr. is nominal: vote among nearest neighbors
                        nn_attr_counter = Counter([int(neighbor[missing_attr_object.record_name]) for neighbor in nn_list])
                        most_common = nn_attr_counter.most_common(n=1)
                        voted_value = most_common[0][0]
                    else:  # -> attr. is continuous: average among nearest neighbors
                        voted_value = statistics.mean([neighbor[missing_attr_object.record_name] for neighbor in nn_list])
                        if missing_attr_object.record_name == 'H_NSVT_N' or missing_attr_object.record_name == 'H_VT/VF_N':
                            voted_value = round(voted_value)
                    temp_ehr_df.loc[index, missing_attr_object.record_name] = voted_value

        # Cache the imputed dataframe for future use
        temp_ehr_df.to_excel(GlobalPaths.cached_imputed_ehr, index=False)
        return temp_ehr_df


class EHRECGParser(EHRBaseParser):
    def __init__(self):
        super(EHRECGParser, self).__init__()
        print('\nLoading ECG Dataset ...')
        try:
            self.ecg_df = pd.read_excel(GlobalPaths.cached_scar_ecg_meta)
            print('--- loaded from cache')
        except FileNotFoundError:
            print('--- No cache found -> parsing muse from scratch')
            self._parse_muse()
            self.ecg_df = pd.read_excel(GlobalPaths.cached_scar_ecg_meta)

        record_ids = set(self.ecg_df['Record_ID'].values)
        self.mri_df = self.mri_df[self.mri_df['Record_ID'].isin(record_ids)]
        self.ehr_df = self.ehr_df[self.ehr_df['Record_ID'].isin(record_ids)]
        # -> Keep in EHR dataset only the attributes indicated in categorical and continuous lists above
        self.ehr_df = self.ehr_df.reindex(columns=EHRAttributeManager.get_all_attrs())

        self.ehr_df = self.ehr_df.sort_values('Record_ID')
        self.ehr_df = self.ehr_df.reset_index(drop=True)
        self.mri_df = self.mri_df.sort_values('Record_ID')
        self.mri_df = self.mri_df.reset_index(drop=True)
        self.ecg_df = self.ecg_df.sort_values('Record_ID')
        self.ecg_df = self.ecg_df.reset_index(drop=True)

        print('\nImputing missing values in EHR dataset ...')
        try:
            self.ehr_df_imputed = pd.read_excel(GlobalPaths.cached_imputed_ehr)
            print('--- loaded from cache')
        except FileNotFoundError:
            print('--- No cache found. Perform Imputation from scratch')
            self.prepare_ehr_for_imputation()
            self.ehr_df_imputed = self.impute_ehr_df()
        self.ehr_df_imputed = self.ehr_df_imputed.reset_index(drop=True)
        self.mri_df = self.mri_df.reset_index(drop=True)

        record_ids = set(self.ehr_df_imputed['Record_ID'].values)
        self.mri_df = self.mri_df[self.mri_df['Record_ID'].isin(record_ids)]
        self.ecg_df = self.ecg_df[self.ecg_df['Record_ID'].isin(record_ids)]

        self.ehr_df = self.ehr_df.sort_values('Record_ID')
        self.ehr_df = self.ehr_df.reset_index(drop=True)
        self.mri_df = self.mri_df.sort_values('Record_ID')
        self.mri_df = self.mri_df.reset_index(drop=True)
        self.ecg_df = self.ecg_df.sort_values('Record_ID')
        self.ecg_df = self.ecg_df.reset_index(drop=True)
        self.ecg_df.to_excel(GlobalPaths.cached_scar_ecg_meta, index=False)
        self.ecg_df.to_csv('Data/ECG/ScarECG/scar_ecg_meta.csv', index=False)

        ecg_names = [f for f in os.listdir(GlobalPaths.ecg) if not f.startswith('.')]
        for ecg_name in ecg_names:
            if '.' in ecg_name and 'meta' not in ecg_name and 'mri' not in ecg_name and 'zip' not in ecg_name:
                record_id = int(ecg_name.split('.')[0])
                if record_id not in record_ids:
                    os.remove(os.path.join(GlobalPaths.ecg, ecg_name))

        print('\nMaking QT Dataset ...')
        try:
            self.qt_dataset = np.load(GlobalPaths.cached_qt_segment_dataset, allow_pickle=True)
            print('Cache found! -> Fetching from cache')
        except OSError:
            print('No Cache Found! -> Parsing Segments From Scratch!')
            self._extract_qt_and_preprocess()

        self.ehr_df_imputed = self.ehr_df_imputed.sort_values(by='Record_ID')
        self.ehr_df_imputed = self.ehr_df_imputed.reset_index(drop=True)
        self.qt_dataset = np.array(sorted(self.qt_dataset, key=lambda item: item['pid']), dtype=dict)

    def remove_ehr_column(self, attribute_name):
        self.ehr_df_imputed = self.ehr_df_imputed.drop(columns=attribute_name)

    def get_categorical_ehr_df(self) -> pd.DataFrame:
        return self.ehr_df_imputed.reindex(columns=EHRAttributeManager.get_nominal_attrs())

    def get_continuous_ehr_df(self) -> pd.DataFrame:
        return self.ehr_df_imputed.reindex(columns=EHRAttributeManager.get_continuous_attrs())

    def _parse_muse(self):
        ecg_meta = []
        ecg_hash_table = {}
        count = 0
        mri_pids = set(self.mri_df['Record_ID'].values)
        xml_names = [f for f in os.listdir(GlobalPaths.muse) if not f.startswith('.')]
        for xml_name in xml_names:
            muse_path = os.path.join(GlobalPaths.muse, xml_name)
            fd = open(muse_path, 'rb')
            muse_dict = xmltodict.parse(fd.read().decode('utf8'))['RestingECG']
            try:
                record_id = int(muse_dict['PatientDemographics']['PatientID'])
                if record_id not in mri_pids:
                    continue
                date_time_str = muse_dict['TestDemographics']['AcquisitionDate'] + ' ' + muse_dict['TestDemographics']['AcquisitionTime']
                date_time = pd.to_datetime(date_time_str)
                frequency = int(muse_dict['Waveform'][1]['SampleBase'])
                temp_csv_path = 'Data/temp_muse_ecg.csv'
                self._translate_measurments(muse_path, temp_csv_path)
                ecg_df = pd.read_csv(filepath_or_buffer=temp_csv_path)
                ecg_df.drop(ecg_df.columns[len(ecg_df.columns) - 1], axis=1, inplace=True)

                count += 1
                if count % 250 == 0:
                    print(f'{count} ECGs processed')
                ecg_object = ECGObject(record_id, date_time, frequency, ecg_df)
                if record_id in ecg_hash_table:
                    ecg_hash_table[record_id].append(ecg_object)
                else:
                    ecg_hash_table[record_id] = [ecg_object]
            except KeyError as error:
                print(error)
                continue

        mri_index_to_drop = []
        for index, row in self.mri_df.iterrows():
            record_id = row['Record_ID']
            closest_ecg_object = None
            if record_id in ecg_hash_table:
                mri_date = pd.to_datetime(row['MRI Date'])
                ecg_object_list = ecg_hash_table[record_id]
                max_timedelta = timedelta(days=360)
                for ecg_object in ecg_object_list:
                    ecg_date = ecg_object.date
                    try:
                        if not self._is_before_myectomy_asa(record_id=record_id, ecg_date=ecg_date):
                            continue
                        diff = abs(mri_date - ecg_date)
                        if diff < max_timedelta:
                            max_timedelta = diff
                            closest_ecg_object = ecg_object
                            closest_ecg_object.mri_date_diff = diff
                    except TypeError as e:
                        print(f'{e} for {record_id}: MRI Date = {mri_date}, ECG Date = {ecg_date}')
            else:
                mri_index_to_drop.append(index)
            if closest_ecg_object is None:
                mri_index_to_drop.append(index)
            else:
                ecg_meta.append([closest_ecg_object.record_id, closest_ecg_object.record_id, closest_ecg_object.frequency,
                                 closest_ecg_object.date, closest_ecg_object.mri_date_diff])
                save_path = f'Data/ECG/ScarECG/{record_id}.csv'
                closest_ecg_object.ecg_df.to_csv(path_or_buf=save_path, index=False)

        ecg_meta_df = pd.DataFrame(ecg_meta, columns=['ECG ID', 'Record_ID', 'Sample Base', 'Acquisition Date', 'MRI Date Difference'])
        ecg_meta_df.to_excel(GlobalPaths.cached_scar_ecg_meta, index=False)
        ecg_meta_df.to_csv('Data/ECG/ScarECG/scar_ecg_meta.csv', index=False)

    def _is_before_myectomy_asa(self, record_id: int, ecg_date: pd.Timestamp) -> bool:
        ehr_target_df = self.ehr_df.loc[self.ehr_df['Record_ID'] == record_id]
        myectomy_date = ehr_target_df['Myectomy_date'].values[0]
        asa_date = ehr_target_df['ASA_date'].values[0]

        if not np.isnat(myectomy_date) and ecg_date > myectomy_date:
            return False
        if asa_date is not None and asa_date != 0 and ecg_date > asa_date:
            return False
        return True

    @staticmethod
    def _translate_measurments(path_muse, path_save):
        museXMLParser = MuseXmlParser()
        expat_parser = xml.parsers.expat.ParserCreate()

        museXMLX = MuseXMLX
        museXMLX.g_Parser = museXMLParser
        expat_parser.StartElementHandler = museXMLX.start_element
        expat_parser.EndElementHandler = museXMLX.end_element
        expat_parser.CharacterDataHandler = museXMLX.char_data

        expat_parser.ParseFile(open(path_muse, 'rb'))
        museXMLParser.makeZcg()
        museXMLParser.writeCSV(path_save)

    def _extract_qt_and_preprocess(self, segment_length: int = 96):
        extractor = QTSegmentExtractor(verbose=True)
        extracted_segments_dict = extractor.extract_segments()
        print(f'---> {len(extracted_segments_dict)} patients processed')
        print('\nPreprocessing segments ...')
        patient_dataset = []
        for pid, segment_dict in extracted_segments_dict.items():
            segments = np.array(segment_dict['segments'])
            de = self.mri_df.loc[self.mri_df['Record_ID'] == pid]['DE'].values[0]
            preprocessed_segments = []
            for segment in segments:
                segment_resampled = TimeSeriesResampler(sz=segment_length).fit_transform(segment)
                segment_resampled = np.reshape(segment_resampled,
                                               (segment_resampled.shape[0], segment_resampled.shape[1]))
                segment_standardized = TimeSeriesScalerMeanVariance().fit_transform(segment_resampled)
                segment_standardized = np.reshape(segment_standardized,
                                                  (segment_standardized.shape[0], segment_standardized.shape[1]))
                preprocessed_segments.append(segment_standardized)
            extracted_segments_dict[pid]['preprocessed'] = preprocessed_segments
            extracted_segments_dict[pid]['de'] = de
            extracted_segments_dict[pid]['pid'] = pid
            patient_dataset.append(extracted_segments_dict[pid])

        patient_dataset = np.array(patient_dataset, dtype=object)
        np.random.shuffle(patient_dataset)
        self.qt_dataset = patient_dataset

        # -> Report:
        p_num_de_0 = 0
        p_num_de_1 = 0
        seg_num_de_0 = 0
        seg_num_de_1 = 0
        for patient_object in patient_dataset:
            if patient_object['de'] == 0:
                p_num_de_0 += 1
                seg_num_de_0 += len(patient_object['preprocessed'])
            else:
                p_num_de_1 += 1
                seg_num_de_1 += len(patient_object['preprocessed'])

        print('Preprocessing Done!  STATS:')
        print(f'---> Class 0: {seg_num_de_0} segments from {p_num_de_0} patients')
        print(f'---> Class 1: {seg_num_de_1} segments from {p_num_de_1} patients\n')

        print(f'Saving Dataset at {GlobalPaths.qt_segment_dataset}')
        np.save(file=os.path.join(GlobalPaths.qt_segment_dataset, 'qt_segment_dataset'), arr=patient_dataset)


class EHRFeatureSelection:
    def __init__(self, ehr_df: pd.DataFrame):
        self.ehr_df = ehr_df

    def get_top_nominal_features(self, min_info_gain: float = 0.002):
        all_features = EHRAttributeManager.get_nominal_attrs(include_record_id=False)
        ig_list = []
        for feature in all_features:
            ig = self.compute_single_information_gain(attribute_name=feature)
            ig_list.append((feature, ig))

        ig_list = sorted(ig_list, key=lambda item: item[1], reverse=True)
        result = [ig_tup[0] for ig_tup in ig_list if ig_tup[1] >= min_info_gain]
        return result

    def get_top_continuous_features(self):
        all_features = EHRAttributeManager.get_continuous_attrs(include_record_id=False)
        p_value_list = []
        for feature in all_features:
            p_value = self.compute_single_welch_t_test(attribute_name=feature)
            p_value_list.append((feature, p_value))

        p_value_list = sorted(p_value_list, key=lambda item: item[1])
        result = [p_value_tup[0] for p_value_tup in p_value_list if p_value_tup[1] <= 0.05]
        return result

    def plot_nominal_hist(self, attribute_name: str):
        target_df = self.ehr_df[['Record_ID', 'DE', attribute_name]]
        df_0 = target_df.loc[target_df['DE'] == 0]
        df_1 = target_df.loc[target_df['DE'] == 1]
        a = np.unique(df_0[attribute_name].values)
        b = np.unique(df_1[attribute_name].values)
        n_bins = max([len(a), len(b)])

        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True, figsize=(10, 5))
        N_0, bins_0, patches_0 = axs[0].hist(df_0[attribute_name].values, bins=n_bins)
        N_1, bins_1, patches_1 = axs[1].hist(df_1[attribute_name].values, bins=n_bins)
        fracs = N_0 / N_0.max()
        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())
        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches_0):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        fracs = N_1 / N_1.max()
        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())
        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches_1):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        mi = self.compute_single_information_gain(attribute_name=attribute_name)

        attr_obj = EHRAttributeManager.get_attr_object(for_record_name=attribute_name)
        fig.suptitle(f"Attribute = {attribute_name} ({attr_obj.description})\n Information Gain = {round(mi[0][1], 6)}", fontsize=15)
        axs[0].set_xlabel('No Scar (DE=0)', fontsize=13)
        axs[0].set_ylabel('# of Patients', fontsize=13)
        axs[1].set_xlabel('Scar (DE=1)', fontsize=13)
        axs[1].set_ylabel('# of Patients', fontsize=13)
        if '/' in attribute_name:
            attribute_name = attribute_name.replace('/', '_')
        plt.savefig(f'FeatureEngineering/EHR/Plots/{attribute_name}.png', dpi=200)
        plt.show()

    def plot_continuous_hist(self, attribute_name: str):
        target_df = self.ehr_df[['Record_ID', 'DE', attribute_name]]
        df_0 = target_df.loc[target_df['DE'] == 0]
        df_1 = target_df.loc[target_df['DE'] == 1]
        mu_0 = df_0[attribute_name].mean()
        sigma_0 = df_0[attribute_name].std()
        mu_1 = df_1[attribute_name].mean()
        sigma_1 = df_1[attribute_name].std()
        n_bins = 50

        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True, figsize=(10, 5))

        N_0, bins_0, patches_0 = axs[0].hist(df_0[attribute_name].values, bins=n_bins, density=True, histtype='stepfilled')
        N_1, bins_1, patches_1 = axs[1].hist(df_1[attribute_name].values, bins=n_bins, density=True, histtype='stepfilled')

        P.setp(patches_0, 'facecolor', 'g', 'alpha', 0.75)
        P.setp(patches_1, 'facecolor', 'r', 'alpha', 0.75)

        axs[0].plot(bins_0, norm.pdf(bins_0, mu_0, sigma_0), 'b', linewidth=3.5)
        axs[1].plot(bins_1, norm.pdf(bins_1, mu_1, sigma_1), 'b', linewidth=3.5)

    def plot_kde(self, attribute_name: str):
        target_df = self.ehr_df[['Record_ID', 'DE', attribute_name]]
        df_0 = target_df.loc[target_df['DE'] == 0]
        df_1 = target_df.loc[target_df['DE'] == 1]
        min_all = min([df_0[attribute_name].min(), df_1[attribute_name].min()])
        max_all = max([df_0[attribute_name].max(), df_1[attribute_name].max()])
        xs = np.linspace(min_all, max_all, 200)

        density_0 = scipy.stats.gaussian_kde(df_0[attribute_name].values, bw_method='silverman')
        density_1 = scipy.stats.gaussian_kde(df_1[attribute_name].values, bw_method='silverman')

        density_0.covariance_factor = lambda: density_0.factor / 1.5
        density_0._compute_covariance()

        density_1.covariance_factor = lambda: density_1.factor / 1.5
        density_1._compute_covariance()

        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True, figsize=(10, 5))
        axs[0].plot(xs, density_0(xs), 'g', linewidth=2.5)
        axs[1].plot(xs, density_1(xs), 'r', linewidth=2.5)

        p_value = self.compute_single_welch_t_test(attribute_name=attribute_name)
        attr_obj = EHRAttributeManager.get_attr_object(for_record_name=attribute_name)
        fig.suptitle(f"Attribute = {attribute_name} ({attr_obj.description})\n P-value = {round(p_value, 5)}",
                     fontsize=15)
        axs[0].set_xlabel('No Scar (DE=0)', fontsize=13)
        axs[0].set_ylabel('Gaussian KDE', fontsize=13)
        axs[1].set_xlabel('Scar (DE=1)', fontsize=13)
        if '/' in attribute_name:
            attribute_name = attribute_name.replace('/', '_')
        plt.savefig(f'FeatureEngineering/EHR/Plots/{attribute_name}.png', dpi=200)
        plt.show()
        v = 9

    @staticmethod
    def _get_x_y(df: pd.DataFrame):
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

    def compute_single_information_gain(self, attribute_name: str):
        x, y, feature_labels = self._get_x_y(df=self.ehr_df.reindex(columns=[attribute_name, 'DE', 'Record_ID']))
        mi = mutual_info_classif(X=x, y=y, discrete_features=True)
        return mi[0]

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

    def compute_single_welch_t_test(self, attribute_name: str):
        continuous_df = self.ehr_df.reindex(columns=[attribute_name, 'DE', 'Record_ID'])
        continuous_df_class_0 = continuous_df.loc[continuous_df['DE'] == 0]
        continuous_df_class_1 = continuous_df.loc[continuous_df['DE'] == 1]
        continuous_df_class_0 = continuous_df_class_0.drop(columns=['DE', 'Record_ID'])
        continuous_df_class_1 = continuous_df_class_1.drop(columns=['DE', 'Record_ID'])
        feature_labels = continuous_df_class_0.columns.values
        sample_0 = continuous_df_class_0[attribute_name].values
        sample_1 = continuous_df_class_1[attribute_name].values
        t_value, p_value = ttest_ind(a=sample_0, b=sample_1, equal_var=False)
        return p_value

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


class ScarAugmentor:
    def __init__(self, ehr_df: pd.DataFrame, ecg_ds: np.ndarray):
        if list(ehr_df.columns.values)[0] != 'Record_ID':
            assert 'Record_ID must be the first column in EHR data frame'
        self.ehr_df = ehr_df.sort_values(by='Record_ID')
        self.ehr_df = self.ehr_df.reset_index(drop=True)
        if len(ecg_ds) > 0:
            self.ecg_ds = np.array(sorted(ecg_ds, key=lambda item: item['pid']), dtype=dict)

    def smote_ehr(self, k_nearest: int = 5):
        y_ix = []
        cat_ix = []
        nominal_set = set(EHRAttributeManager.get_nominal_attrs(include_record_id=False))
        col_list = list(self.ehr_df.columns.values)
        for col_index in range(len(col_list)):
            attribute_name = col_list[col_index]
            if attribute_name == 'DE':
                y_ix.append(col_index - 1)
                # Why -1? -> Record_id will be removed when computing distance,
                # so all indexes must be shifted one to the left (record_id is the first column!).
            elif attribute_name in nominal_set:
                cat_ix.append(col_index - 1)

        ehr_dist_matrix = Distance.compute_hvdm_matrix(input_df=self.ehr_df, y_ix=y_ix, cat_ix=cat_ix)

        ehr_augmented_list = []
        for i in range(len(ehr_dist_matrix)):
            row = self.ehr_df.iloc[i]
            if row['DE'] == 1:
                continue
            neighbour_distances = []
            for j in range(len(ehr_dist_matrix)):
                ecg_dist_neighbor_over_leads = []
                ehr_dist_neighbor_over_leads = []
                if i != j:
                    dist = ehr_dist_matrix[i, j][0]
                    n_object = ehr_dist_matrix[i, j][2]
                    neighbour_distances.append((dist, n_object))
                else:
                    neighbour_distances.append(None)

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

            ix = random.randint(0, k_nearest - 1)
            selected_neighbor = nn_list[ix][1]
            alpha = random.uniform(0, 1)
            augmented_dict = {}
            for attribute_name, value in row.iteritems():
                if attribute_name == 'DE':
                    augmented_dict['DE'] = 0
                elif attribute_name == 'Record_ID':
                    augmented_dict['Record_ID'] = 1000000 + value
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
                    else:  # -> Attribute is continuous
                        value_main = value
                        value_neighbor = selected_neighbor[attr_object.record_name]
                        augmented_value = (alpha * value_main) + ((1 - alpha) * value_neighbor)
                        augmented_dict[attr_object.record_name] = augmented_value
            ehr_augmented_list.append(augmented_dict)

        return pd.DataFrame(ehr_augmented_list)


    def smote_ehr_ecg(self, k_nearest: int = 5, dist_mode: str = 'both'):
        '''
        To augment ECG part of sample s:
            Step 1 -> Randomly select a neighbor s' from K nearest neighbors.
            For each lead:
                Step 2 -> Randomly select a QT segment from s and another QT segment from s'
                Step 3 -> Average the QT segments (weighted uniformly) = Generated QT segment
            Step 4 -> Stack up the generated QT segments
        '''
        if dist_mode == 'both' and self.ehr_df.shape[0] != self.ecg_ds.shape[0]:
            assert 'EHR and ECG Dataset must have same number of data points for `both` mode augmentation'
        print(f'\nPerforming SMOTE augmentation with {dist_mode} mode on {self.ehr_df.shape[0]} data points')
        ehr_dist_matrix = None
        ecg_dist_matrix = None
        if dist_mode == 'ehr' or dist_mode == 'both':

            y_ix = []
            cat_ix = []
            nominal_set = set(EHRAttributeManager.get_nominal_attrs(include_record_id=False))
            col_list = list(self.ehr_df.columns.values)
            for col_index in range(len(col_list)):
                attribute_name = col_list[col_index]
                if attribute_name == 'DE':
                    y_ix.append(col_index-1)
                    # Why -1? -> Record_id will be removed when computing distance,
                    # so all indexes must be shifted one to the left (record_id is the first column!).
                elif attribute_name in nominal_set:
                    cat_ix.append(col_index-1)

            ehr_dist_matrix = Distance.compute_hvdm_matrix(input_df=self.ehr_df, y_ix=y_ix, cat_ix=cat_ix)
        if dist_mode == 'ecg' or dist_mode == 'both':
            ecg_dist_matrix = Distance.compute_ecg_distance_matrix(self.ecg_ds)

        if ehr_dist_matrix is not None:
            for i in range(len(ehr_dist_matrix)):
                dist_row = ehr_dist_matrix[i, :]
                dist_row = [dist[0] for dist in dist_row if dist is not None]
                max_dist = max(dist_row)
                min_dist = min(dist_row)
                for j in range(len(ehr_dist_matrix[i, :])):
                    dist_obj = ehr_dist_matrix[i, j]
                    if dist_obj is not None:
                        dist = dist_obj[0]
                        dist_norm = (dist - min_dist) / (max_dist - min_dist)
                        new_tuple = (dist_norm, dist, dist_obj[1], dist_obj[2])
                        ehr_dist_matrix[i, j] = new_tuple

        if ecg_dist_matrix is not None:
            for i in range(len(ecg_dist_matrix)):
                for k in range(4):
                    dist_row = ecg_dist_matrix[i, :, k]
                    dist_row = [dist[1] for dist in dist_row if dist is not None]
                    max_dist = max(dist_row)
                    min_dist = min(dist_row)
                    for j in range(len(ecg_dist_matrix[i, :, k])):
                        dist_obj = ecg_dist_matrix[i, j, k]
                        if dist_obj is not None:
                            dist = dist_obj[1]
                            dist_norm = (dist - min_dist) / (max_dist - min_dist)
                            new_tuple = (dist_norm, dist, dist_obj[0], dist_obj[2])
                            ecg_dist_matrix[i, j, k] = new_tuple

        merged_dist_matrix = np.empty(shape=(len(self.ecg_ds), len(self.ecg_ds), 4), dtype=object)
        for i in range(len(self.ehr_df)):
            for j in range(len(self.ehr_df)):
                if i != j:
                    for lead_index in range(4):

                        ehr_dist = None
                        ehr_dist_norm = None
                        ecg_dist = None
                        ecg_dist_norm = None

                        other_pid = self.ecg_ds[j]['pid']
                        other_ehr_object = self.ehr_df.iloc[j]
                        other_ecg_object = self.ecg_ds[j]

                        if ehr_dist_matrix is not None:
                            ehr_dist_norm = ehr_dist_matrix[i, j][0]
                            ehr_dist = ehr_dist_matrix[i, j][1]

                        if ecg_dist_matrix is not None:
                            ecg_dist_norm = ecg_dist_matrix[i, j, lead_index][0]
                            ecg_dist = ecg_dist_matrix[i, j, lead_index][1]

                        ehr_ecg_distance_object = EHRECGDistanceObject(other_pid=other_pid,
                                                                       ehr_dist_norm=ehr_dist_norm, ehr_dist=ehr_dist,
                                                                       ecg_dist_norm=ecg_dist_norm, ecg_dist=ecg_dist,
                                                                       other_ehr_object=other_ehr_object,
                                                                       other_ecg_object=other_ecg_object)
                        merged_dist_matrix[i, j, lead_index] = ehr_ecg_distance_object

        # -> Augment EHR Part
        ehr_augmented_list = []
        for i in range(len(merged_dist_matrix)):
            row = self.ehr_df.iloc[i]
            if row['DE'] == 1:
                continue
            neighbour_distances = []
            for j in range(len(merged_dist_matrix)):
                ecg_dist_neighbor_over_leads = []
                ehr_dist_neighbor_over_leads = []
                if i != j:
                    for k in range(4):
                        ehr_ecg_distance = merged_dist_matrix[i, j, k]
                        ehr_dist_norm = ehr_ecg_distance.ehr_dist_norm
                        ecg_dist_norm = ehr_ecg_distance.ecg_dist_norm
                        if ehr_dist_norm is not None:
                            ehr_dist_neighbor_over_leads.append(ehr_dist_norm)
                        if ecg_dist_norm is not None:
                            ecg_dist_neighbor_over_leads.append(ecg_dist_norm)
                    mean_ehr_dist_among_leads = 0
                    mean_ecg_dist_among_leads = 0
                    if len(ecg_dist_neighbor_over_leads) > 0:
                        mean_ecg_dist_among_leads = statistics.mean(ecg_dist_neighbor_over_leads)
                    if len(ehr_dist_neighbor_over_leads) > 0:
                        mean_ehr_dist_among_leads = statistics.mean(ehr_dist_neighbor_over_leads)
                    neighbour_distances.append(
                        (mean_ehr_dist_among_leads + mean_ecg_dist_among_leads, self.ehr_df.iloc[j]))
                else:
                    neighbour_distances.append(None)

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

            ix = random.randint(0, k_nearest - 1)
            selected_neighbor = nn_list[ix][1]
            alpha = random.uniform(0, 1)
            augmented_dict = {}
            for attribute_name, value in row.iteritems():
                if attribute_name == 'DE':
                    augmented_dict['DE'] = 0
                elif attribute_name == 'Record_ID':
                    augmented_dict['Record_ID'] = 1000000 + value
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
                    else:  # -> Attribute is continuous
                        value_main = value
                        value_neighbor = selected_neighbor[attr_object.record_name]
                        augmented_value = (alpha * value_main) + ((1 - alpha) * value_neighbor)
                        augmented_dict[attr_object.record_name] = augmented_value
            ehr_augmented_list.append(augmented_dict)

        # Augment ECG Part -> Each lead is augmented separately.
        ecg_augmented_list = []
        for i in range(len(merged_dist_matrix)):
            row_ehr = self.ehr_df.iloc[i]
            if row_ehr['DE'] == 1:
                continue
            p_object = self.ecg_ds[i]
            new_segment = []
            for lead_index in range(4):  # -> Augment each lead separately
                merged_dist_row = merged_dist_matrix[i, :, lead_index]
                neighbour_distances = []
                for j in range(len(merged_dist_row)):
                    if i != j:
                        ehr_ecg_distance = merged_dist_row[j]
                        ehr_dist_norm = ehr_ecg_distance.ehr_dist_norm
                        ecg_dist_norm = ehr_ecg_distance.ecg_dist_norm
                        dist = 0
                        if ehr_dist_norm is not None:
                            dist += ehr_dist_norm
                        if ecg_dist_norm is not None:
                            dist += ecg_dist_norm
                        # -> Append distance from i to all j neighbors for lead_index
                        neighbour_distances.append(
                            (dist, ehr_ecg_distance.other_ehr_object, ehr_ecg_distance.other_ecg_object))
                neighbour_distances = np.array([x for x in neighbour_distances if not pd.isna(x)], dtype=object)
                sorted_nn = sorted(neighbour_distances, key=lambda item: item[0])
                nn_list = []
                for nn in sorted_nn:
                    dist = nn[0]
                    neighbor_ehr = nn[1]
                    neighbor_ecg = nn[2]
                    if neighbor_ehr['DE'] == 0:
                        nn_list.append((dist, neighbor_ecg))
                    if len(nn_list) == k_nearest:
                        break

                selected_neighbor = nn_list[random.randint(0, len(nn_list) - 1)]
                selected_neighbor_qt_segments = selected_neighbor[1]['preprocessed']
                neighbor_segments = selected_neighbor_qt_segments[
                    random.randint(0, len(selected_neighbor_qt_segments) - 1)]
                neighbor_segment = neighbor_segments[lead_index, :]

                main_segments = p_object['preprocessed']
                main_segment = main_segments[random.randint(0, len(main_segments) - 1)]
                main_segment = main_segment[lead_index, :]

                generated_segment = softdtw_barycenter(X=[main_segment, neighbor_segment], gamma=0.8)
                generated_segment = np.reshape(generated_segment, generated_segment.shape[0])
                generated_segment = SignalProcessing.smooth_two_third(generated_segment)
                fig, ax = plt.subplots(3)
                ax[0].plot(main_segment)
                ax[1].plot(neighbor_segment)
                ax[2].plot(generated_segment)
                plt.show()
                new_segment.append(generated_segment)
                v = 9

                # nearest_mean_segments = []
                # for other_p_object in nn_list:
                #     segments = np.array(other_p_object[1]['preprocessed'])
                #     segments = segments[:, lead_index, :]  # -> will be of shape m x 96, where m is the number of hb
                #     mean_segment = softdtw_barycenter(segments, gamma=0.1, max_iter=10, tol=1e-3)
                #     mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
                #     mean_segment = SignalProcessing.smooth_two_third(mean_segment)
                #     nearest_mean_segments.append(mean_segment)
                #
                # segments = np.array(p_object['preprocessed'])
                # mean_segment = softdtw_barycenter(segments[:, lead_index, :], gamma=0.1, max_iter=10, tol=1e-3)
                # mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
                # mean_segment = SignalProcessing.smooth_two_third(mean_segment)
                # main_segment = mean_segment
                #
                # new_lead_segments = []
                # for neighbor_segment in nearest_mean_segments:
                #     temp_for_mean = [main_segment, neighbor_segment]
                #     interpolate_1 = softdtw_barycenter(X=temp_for_mean, gamma=1, weights=[0.5, 0.5])
                #     interpolate_2 = softdtw_barycenter(X=temp_for_mean, gamma=1, weights=[0.25, 0.75])
                #     interpolate_3 = softdtw_barycenter(X=temp_for_mean, gamma=1, weights=[0.75, 0.25])
                #     interpolate_1 = np.reshape(interpolate_1, interpolate_1.shape[0])
                #     interpolate_2 = np.reshape(interpolate_2, interpolate_2.shape[0])
                #     interpolate_3 = np.reshape(interpolate_3, interpolate_3.shape[0])
                #     interpolate_1 = SignalProcessing.smooth_two_third(interpolate_1)
                #     interpolate_2 = SignalProcessing.smooth_two_third(interpolate_2)
                #     interpolate_3 = SignalProcessing.smooth_two_third(interpolate_3)
                #     new_lead_segments.append(interpolate_1)
                #     new_lead_segments.append(interpolate_2)
                #     new_lead_segments.append(interpolate_3)
                # new_lead_segments = np.array(new_lead_segments)
                # new_segment.append(new_lead_segments)

            new_segments = np.array(new_segment)
            results = []
            for i in range(new_segments.shape[1]):
                temp = []
                for lead in range(4):
                    segment = new_segments[lead, i, :]
                    temp.append(segment)
                temp = np.array(temp)
                temp = np.transpose(temp)
                results.append(temp)
            results = np.array(results)
            ecg_augmented_list.append({'pid': p_object['pid'] + 1000000,
                                       'de': p_object['de'],
                                       'preprocessed': results})

        return pd.DataFrame(ehr_augmented_list), np.array(ecg_augmented_list)


if __name__ == '__main__':

    parser = EHRECGParser()
    feature_selector = EHRFeatureSelection(ehr_df=parser.ehr_df_imputed)
    # pair = feature_selector.compute_information_gain(for_discrete_vars=True)
    # feature_selector.plot_feature_score(feature_score_pair=pair, y_title='Information Gain')
    result = feature_selector.compute_welch_t_test()
    feature_selector.plot_feature_score(feature_score_pair=result, y_title='p-value', y_limit=0.05)
    # top_nominal_features = feature_selector.get_top_nominal_features()
    # top_continuous_features = feature_selector.get_top_continuous_features()
    # vv = ScarAugmentor(ehr_df=parser.ehr_df_imputed, ecg_ds=parser.qt_dataset)
    v = 9
    # v = parser.qt_dataset
    # vv = parser.ehr_df_imputed
#
#     print('Done!')
    # feature_explorer = EHRFeatureSelection(ehr_df=parser.ehr_df_imputed)
    # for attribute_name in EHRAttributeManager.get_nominal_attrs(include_record_id=False):
    #     feature_explorer.plot_hist(attribute_name=attribute_name)
    # for attribute_name in EHRAttributeManager.get_continuous_attrs(include_record_id=False):
    #     feature_explorer.plot_continuous_hist(attribute_name=attribute_name)

    # for attribute_name in EHRAttributeManager.get_continuous_attrs(include_record_id=False):
    #     feature_explorer.plot_kde(attribute_name=attribute_name)

    # path_to_ehr = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/Overall.xlsx'
    # path_to_mri = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/HCM_MRI_Database_03022022.xlsx'
    # path_to_ecg = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/ECGMeta.xlsx'
    #
    # # -> Load dataset and impute missing values. Result will be in ehr_df (pd.DataFrame)
    # parser = EHRScarParser(path_to_ehr=path_to_ehr, path_to_mri=path_to_mri, path_to_ecg=path_to_ecg)
    # categorical_df = parser.get_categorical_ehr_df()
    # continuous_df = parser.get_continuous_ehr_df()
    # scar_df = parser.get_scar_labels()
    #
    # ehr_df = pd.merge(categorical_df, continuous_df, on='Record_ID')
    # ehr_df = pd.merge(ehr_df, scar_df, on='Record_ID')
    #
    # # -> Feature Selection. Result will be in selected_features (list of str)
    # scar_feature_selection = ScarEHRFeatureSelection(ehr_df=ehr_df)
    # mi_features_discrete = scar_feature_selection.compute_information_gain(for_discrete_vars=True)
    # ttest_features_continuous = scar_feature_selection.compute_welch_t_test()
    #
    # scar_feature_selection.plot_feature_score(feature_score_pair=mi_features_discrete, y_limit=0.002, y_title='MI')
    # scar_feature_selection.plot_feature_score(feature_score_pair=ttest_features_continuous, y_limit=0.05, y_title='P-value')
    #
    # selected_features_discrete = [x[0] for x in mi_features_discrete if x[1] >= 0.002]
    # selected_features_continuous = [x[0] for x in ttest_features_continuous if x[1] <= 0.05]
    # selected_features = selected_features_discrete + selected_features_continuous
    #
    # cols = list(ehr_df)
    # cols.insert(1, cols.pop(cols.index('DE')))
    # ehr_df = ehr_df.loc[:, cols]
    # ehr_df = ehr_df.reindex(columns=['DE'] + selected_features)
    #
    # ehr_df = ehr_df.sample(frac=1).reset_index(drop=True)  # -> Do this one time only for now. You will cache the augmentation result
    #
    # df_train = ehr_df.iloc[:round(ehr_df.shape[0] * 0.8)]
    # df_test = ehr_df.iloc[round(ehr_df.shape[0] * 0.8):]
    #
    # df_train = df_train.reset_index(drop=True)
    # df_test = df_test.reset_index(drop=True)
    #
    # minority_class_shape = df_train.loc[df_train['DE'] == 0].shape
    # majority_class_shape = df_train.loc[df_train['DE'] == 1].shape
    # print(f'\nOriginal Train Set DE=0: {minority_class_shape} | DE=1: {majority_class_shape}')
    #
    # scar_ehr_augmentor = ScarEHRAugmentor(ehr_df=df_train)
    # augmented_df = scar_ehr_augmentor.perform_smote()
    # df_train_augmented = pd.concat([df_train, augmented_df], axis=0)
    # df_train_augmented = df_train_augmented.sample(frac=1).reset_index(drop=True)
    #
    # minority_class_shape = df_train_augmented.loc[df_train_augmented['DE'] == 0].shape
    # majority_class_shape = df_train_augmented.loc[df_train_augmented['DE'] == 1].shape
    # print(f'\nAugmented Train Set DE=0: {minority_class_shape} | DE=1: {majority_class_shape}')
    #
    # df_train_augmented.to_excel("Data/CachedEHR/AugmentedTrainSet_V1.xlsx", index=False)
    # df_test.to_excel("Data/CachedEHR/TestSet_V1.xlsx", index=False)
    #
    #
    # print('Done!')


    #
    # X_train, y_train = df_train_augmented.iloc[:, 1:], df_train_augmented.iloc[:, 0].values
    # X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0].values









