import statistics
import random
import pandas as pd
import numpy as np
from collections import Counter
from tslearn.barycenters import softdtw_barycenter
from Utility import SignalProcessing
from EHRDataManagement import EHRScarParser, ScarEHRFeatureSelection, EHRDistance, EHRAttributeManager
from ECGDataManagement import QTSegmentParser, ECGDistance


path_to_ehr = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/Overall.xlsx'
path_to_mri = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/HCM_MRI_Database_03022022.xlsx'
path_to_ecg = '/Users/kasra/OneDrive - University of Delaware - o365/UCSF Data/ECGMeta.xlsx'


def get_ehr_df() -> pd.DataFrame:
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
    selected_features_discrete = [x[0] for x in mi_features_discrete if x[1] >= 0.002]
    selected_features_continuous = [x[0] for x in ttest_features_continuous if x[1] <= 0.05]
    selected_features = selected_features_discrete + selected_features_continuous

    cols = list(ehr_df)
    cols.insert(1, cols.pop(cols.index('DE')))
    ehr_df = ehr_df.loc[:, cols]
    ehr_df = ehr_df.reindex(columns=['Record_ID', 'DE'] + selected_features)
    return ehr_df


def get_ecg_df(size: int = 640):
    ecg_parser = QTSegmentParser(size=size)
    return ecg_parser.patient_dataset


def merge_ehr_ecg(ehr_df: pd.DataFrame, ecg_ds: np.ndarray):
    ehr_pids = set(ehr_df['Record_ID'].values)
    temp_ecg_ds = []
    ecg_pids = set()
    for p_dict in ecg_ds:
        if p_dict['pid'] in ehr_pids:
            ecg_pids.add(p_dict['pid'])
            temp_ecg_ds.append(p_dict)

    ecg_ds = np.array(temp_ecg_ds, dtype=dict)
    ehr_df = ehr_df.loc[ehr_df['Record_ID'].isin(ecg_pids)]

    ehr_df = ehr_df.sort_values(by='Record_ID')
    ehr_df = ehr_df.reset_index(drop=True)
    ecg_ds = np.array(sorted(ecg_ds, key=lambda item: item['pid']), dtype=dict)

    return ehr_df, ecg_ds


def smote_ehr_ecg(ehr_df: pd.DataFrame, ecg_ds: np.ndarray, k_nearest: int = 5, dist_mode: str = 'both'):
    ehr_dist_matrix = None
    ecg_dist_matrix = None
    if dist_mode == 'ehr' or dist_mode == 'both':
        ehr_dist_matrix = EHRDistance.compute_hvdm_matrix(input_df=ehr_df, y_ix=[0], cat_ix=list(range(1, 13)))
    if dist_mode == 'ecg' or dist_mode == 'both':
        ecg_dist_matrix = ECGDistance.compute_distance_matrix(ecg_ds)

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

    merged_dist_matrix = np.empty(shape=(len(ecg_ds), len(ecg_ds), 4), dtype=object)
    for i in range(len(ehr_df)):
        for j in range(len(ehr_df)):
            if i != j:
                for lead_index in range(4):

                    ehr_dist = None
                    ehr_dist_norm = None
                    ecg_dist = None
                    ecg_dist_norm = None

                    other_pid = ecg_ds[j]['pid']
                    other_ehr_object = ehr_df.iloc[j]
                    other_ecg_object = ecg_ds[j]

                    if ehr_dist_matrix is not None:
                        ehr_dist_norm = ehr_dist_matrix[i, j][0]
                        ehr_dist = ehr_dist_matrix[i, j][1]

                    if ecg_dist_matrix is not None:
                        ecg_dist_norm = ecg_dist_matrix[i, j, lead_index][0]
                        ecg_dist = ecg_dist_matrix[i, j, lead_index][1]

                    ehr_ecg_distance_object = EHRECGDistance(other_pid=other_pid,
                                                             ehr_dist_norm=ehr_dist_norm, ehr_dist=ehr_dist,
                                                             ecg_dist_norm=ecg_dist_norm, ecg_dist=ecg_dist,
                                                             other_ehr_object=other_ehr_object, other_ecg_object=other_ecg_object)
                    merged_dist_matrix[i, j, lead_index] = ehr_ecg_distance_object

    # -> Augment EHR Part
    ehr_augmented_list = []
    for i in range(len(merged_dist_matrix)):
        row = ehr_df.iloc[i]
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
                neighbour_distances.append((mean_ehr_dist_among_leads + mean_ecg_dist_among_leads, ehr_df.iloc[j]))
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
                    augmented_value = (alpha * value_main) + ((1-alpha) * value_neighbor)
                    augmented_dict[attr_object.record_name] = augmented_value
        ehr_augmented_list.append(augmented_dict)

    # Augment ECG Part -> Each lead is augmented separately.
    ecg_augmented_list = []
    for i in range(len(merged_dist_matrix)):
        row_ehr = ehr_df.iloc[i]
        if row_ehr['DE'] == 1:
            continue
        p_object = ecg_ds[i]
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
                    # -> Append i distance from all j neighbors for lead_index
                    neighbour_distances.append((dist, ehr_ecg_distance.other_ehr_object, ehr_ecg_distance.other_ecg_object))
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

            nearest_mean_segments = []
            for other_p_object in nn_list:
                segments = np.array(other_p_object[1]['preprocessed'])
                segments = segments[:, lead_index, :]  # -> will be of shape m x 96, where m is the number of hb
                mean_segment = softdtw_barycenter(segments, gamma=0.1, max_iter=10, tol=1e-3)
                mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
                mean_segment = SignalProcessing.smooth_two_third(mean_segment)
                nearest_mean_segments.append(mean_segment)

            segments = np.array(p_object['preprocessed'])
            mean_segment = softdtw_barycenter(segments[:, lead_index, :], gamma=0.1, max_iter=10, tol=1e-3)
            mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
            mean_segment = SignalProcessing.smooth_two_third(mean_segment)
            main_segment = mean_segment

            new_lead_segments = []
            for neighbor_segment in nearest_mean_segments:
                temp_for_mean = [main_segment, neighbor_segment]
                interpolate_1 = softdtw_barycenter(X=temp_for_mean, gamma=1, weights=[0.5, 0.5])
                interpolate_2 = softdtw_barycenter(X=temp_for_mean, gamma=1, weights=[0.25, 0.75])
                interpolate_3 = softdtw_barycenter(X=temp_for_mean, gamma=1, weights=[0.75, 0.25])
                interpolate_1 = np.reshape(interpolate_1, interpolate_1.shape[0])
                interpolate_2 = np.reshape(interpolate_2, interpolate_2.shape[0])
                interpolate_3 = np.reshape(interpolate_3, interpolate_3.shape[0])
                interpolate_1 = SignalProcessing.smooth_two_third(interpolate_1)
                interpolate_2 = SignalProcessing.smooth_two_third(interpolate_2)
                interpolate_3 = SignalProcessing.smooth_two_third(interpolate_3)
                new_lead_segments.append(interpolate_1)
                new_lead_segments.append(interpolate_2)
                new_lead_segments.append(interpolate_3)
            new_lead_segments = np.array(new_lead_segments)
            new_segment.append(new_lead_segments)

            v = 9

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

    return ehr_augmented_list, ecg_augmented_list


class EHRECGDistance:
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


if __name__ == '__main__':
    ehr_df = get_ehr_df()
    ecg_ds = get_ecg_df(size=50)

    ehr_df, ecg_ds = merge_ehr_ecg(ehr_df, ecg_ds)
    smote_ehr_ecg(ehr_df, ecg_ds, dist_mode='ehr')



    vv = 9





