import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from QTSegmentExtractor import QTSegmentExtractor
import GlobalPaths
from Utility import Util
from scipy.stats import entropy, skew, kurtosis
from hurst import compute_Hc
from fQRS import parse_QRS_complex


def plot_ecg(ecg: pd.DataFrame, pid: int = None, ecg_id: int = None):
    fig, ax = plt.subplots(12, figsize=(12, 15))
    for i in range(12):
        ax[i].plot(ecg.values[:, i])
        ax[i].set_title(Util.get_lead_name(i))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

    if pid is not None and ecg_id is not None:
        fig.suptitle(f'PID={pid} ECGID={ecg_id}')
    plt.show()


def plot_qt_segments(segments: [np.ndarray], pid: int = None, ecg_id: int = None, x: [int] = None):
    fig, ax = plt.subplots(nrows=12, ncols=10, figsize=(12, 15))
    n = min([10, len(segments)])
    for i in range(12):
        for j in range(n):
            qt_segment = segments[j][i]
            ax[i][j].plot(qt_segment)
            ax[i][j].axvline(x=x[j], color='red')
            if j == 0:
                ax[i][j].set_title(Util.get_lead_name(i))
    for i in range(12):
        for j in range(10):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            ax[i][j].spines['bottom'].set_visible(False)
            ax[i][j].spines['left'].set_visible(False)

    if pid is not None and ecg_id is not None:
        fig.suptitle(f'PID={pid} ECGID={ecg_id}')
    plt.show()


def get_ecg_scar_dataset() -> pd.DataFrame:
    mri_loc_df = pd.read_excel(GlobalPaths.scar_location)
    mri_meta_df = pd.read_excel(GlobalPaths.mri)
    mri_meta_df = mri_meta_df[['Record_ID', 'Scar tissue %']]
    mri_loc_df = mri_loc_df[['Record_ID', 'MRI Date'] + [col for col in mri_loc_df.columns if
                                                         'Basal' in col or 'Mid' in col or 'Apical' in col or 'Apex' in col]]
    dataset = pd.merge(left=mri_meta_df, right=mri_loc_df, how='inner', on=['Record_ID'])
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # Clean MRI Date.
    need_correction = 0
    for index, row in dataset.iterrows():
        try:
            dataset.iat[index, 2] = pd.to_datetime(row['MRI Date'])
        except:
            date_str = str(row['MRI Date'])
            if '.' in date_str:
                date_str = date_str.replace('.', '')
            if ',' in date_str:
                mri_date = pd.to_datetime(date_str.split(',')[0])
            else:
                mri_date = pd.to_datetime(date_str.split(' ')[0])
            need_correction += 1
            dataset.iat[index, 2] = mri_date
    dataset['MRI Date'] = pd.to_datetime(dataset['MRI Date'])

    ecg_df = pd.read_excel('Data/overall_ecg_feature_uncertain_scar_location.xlsx')

    dataset = pd.merge(left=ecg_df, right=dataset[['Record_ID', 'Scar tissue %']], how='inner', on=['Record_ID'])
    dataset.reset_index(drop=True, inplace=True)
    mri_diff_days = []
    entire_lv_scar = []
    for index, row in dataset.iterrows():
        ecg_date = pd.to_datetime(row['ECG Date'])
        mri_date = pd.to_datetime(row['MRI Date'])
        diff = ecg_date - mri_date
        mri_diff_days.append(diff.days)
        entire_lv_scar.append(sum(row[['Basal', 'Mid', 'Apical', 'Apex']].values))

    dataset['MRI Diff'] = mri_diff_days
    dataset['LV Scar'] = entire_lv_scar
    return dataset


# dataset = get_ecg_scar_dataset()

extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg_loc,
                                   ann_dir_path=GlobalPaths.pla_annotation_loc,
                                   metadata_path=GlobalPaths.ecg_meta_loc,
                                   verbose=True)

extract_dict = extractor.extract_segments(debug=10)
for pid, ecg_list in extract_dict.items():
    for ecg_dict in ecg_list:
        frequency = ecg_dict['frequency']
        # plot_ecg(ecg_dict['ecg_denoised'])

        mean_qt_dist = round(statistics.mean(ecg_dict['qt_distances']) * (1/frequency), 4)
        std_qt_dist = round(statistics.stdev(ecg_dict['qt_distances']) * (1 / frequency), 4)
        voted_qrs_offsets = [0] * len(ecg_dict['segments'])
        k = 0
        for qt_segment_12_lead in ecg_dict['segments']:
            candidate_qrs_offsets = []
            for lead in range(12):
                qt_segment = qt_segment_12_lead[lead, :]
                try:
                    qrs, qrs_offset = parse_QRS_complex(qt_segment)
                except:
                    continue
                candidate_qrs_offsets.append(qrs_offset)

            # Vote among 12 qrs offsets -> threshold for vicinity = 20ms. Any two QRS offsets farther than 20ms from
            # each other are not in each other's consensus set.
            threshold = round(frequency / 50)
            consensus = []
            for i in range(len(candidate_qrs_offsets)):
                c = 0
                offset_current = candidate_qrs_offsets[i]
                for j in range(len(candidate_qrs_offsets)):
                    if i != j:
                        offset_neighbor = candidate_qrs_offsets[j]
                        if abs(offset_current - offset_neighbor) <= threshold:
                            c += 1
                consensus.append(c)
            # Final QRS offset is the weighted average of all lead QRS offsets whose consensus is more than 4.
            offsets = []
            weights = []
            for i in range(len(consensus)):
                if consensus[i] >= 5:
                    weights.append(consensus[i])
                    offsets.append(candidate_qrs_offsets[i])
            qrs_offset = round(np.average(offsets, weights=weights)) + 1
            voted_qrs_offsets[k] = qrs_offset
            k += 1
        plot_qt_segments(ecg_dict['segments'], x=voted_qrs_offsets)
        # Part 2
        temp = list(zip(voted_qrs_offsets, ecg_dict['segments']))
        consensus_segment = []
        for i in range(len(ecg_dict['segments'])):
            c = 0
            current_qrs_offset = temp[i][0]
            for j in range(len(ecg_dict['segments'])):
                if i != j:
                    neighbor_qrs_offset = temp[j][0]
                    if abs(current_qrs_offset - neighbor_qrs_offset) <= 2:
                        c += 1
            consensus_segment.append((c, temp[i]))


        v = 9

        v = 9
        signal = ecg_dict['ecg_denoised'][Util.get_lead_name(lead)].values
        sk = skew(signal)
        ku = kurtosis(signal)

rand_pids = np.random.choice(pids, size= 10, replace=False)
for pid in rand_pids:
    count = 0
    for ecg_dict in extract_dict[pid]:
        plot_ecg(ecg_dict['ecg_denoised'])
        plot_qt_segments(ecg_dict['segments'])
        count += 1
        if count > 2:
            break
        v = 9