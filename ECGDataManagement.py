import os

import numpy
import numpy as np
import pandas as pd
from QTSegmentExtractor import QTSegmentExtractor
from Utility import Loader, Util, NoOffsetException, SignalProcessing
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance
from sklearn.model_selection import KFold
from tslearn.metrics import dtw
from tslearn.barycenters import softdtw_barycenter
import random
import statistics
from time import time
from datetime import datetime, timedelta


class ECGDistance:
    @staticmethod
    def compute_distance_matrix(ecg_ds: np.ndarray):
        dist_matrix = np.empty(shape=(len(ecg_ds), len(ecg_ds), 4), dtype=object)
        for i in range(len(ecg_ds)):
            for j in range(len(ecg_ds)):
                if j > i:
                    for lead_index in range(4):
                        dist = Util.get_qt_distance(p1=ecg_ds[i], p2=ecg_ds[j], lead_index=lead_index)
                        dist_matrix[i, j, lead_index] = (ecg_ds[j]['pid'], dist, ecg_ds[j]['preprocessed'])

        print('Copying lower triangle')
        for i in range(len(ecg_ds) - 1, 0, -1):
            for j in range(len(ecg_ds)):
                if j < i:
                    for lead_index in range(4):
                        dist_matrix[i, j, lead_index] = (ecg_ds[j]['pid'], dist_matrix[j, i, lead_index][1], ecg_ds[j]['preprocessed'])

        return dist_matrix


class QTSegmentParser:
    def __init__(self, size: int, segment_length: int = 96):
        self.num_ecgs = size
        self.segment_length = segment_length
        self.leads = [1, 2, 5, 7]
        self.segment_name = 'Vent Segment'
        self.patient_dataset = None
        self.path_to_save = 'Data/Dataset'
        try:
            self.patient_dataset = np.load(os.path.join(self.path_to_save, f'patient_dataset_{self.num_ecgs}.npy'),
                                           allow_pickle=True)
            print('ECG Dataset Fetched from Cache!')
        except OSError:
            print('No Cache Found -> Parsing Segments From Scratch!')
            self._extract_and_preprocess()

    def _extract_and_preprocess(self):
        print(f'Extracting QT segments from {self.num_ecgs} patients ...')
        extractor = QTSegmentExtractor(verbose=False)
        extracted_segments_dict = extractor.extract_segments(num_ecgs=self.num_ecgs)
        print(f'---> {len(extracted_segments_dict)} patients processed')
        print('\nPreprocessing segments ...')
        scar_df = pd.read_excel('Data/scar_dataset.xlsx')
        patient_dataset = []
        for pid, segment_dict in extracted_segments_dict.items():
            # if pid != 10338:
            #     continue
            segments = np.array(segment_dict['segments'])
            de = scar_df.loc[scar_df['Record ID'] == pid]['DE'].values[0]
            preprocessed_segments = []
            for segment in segments:
                segment_resampled = TimeSeriesResampler(sz=self.segment_length).fit_transform(segment)
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
        self.patient_dataset = patient_dataset
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

        print(f'Saving Dataset at {self.path_to_save}')
        np.save(file=os.path.join(self.path_to_save, f'patient_dataset_{self.num_ecgs}'), arr=patient_dataset)


class DatasetLoader:
    def __init__(self, size: int, n_folds: int, segment_length: int = 96):
        self.num_ecgs = size
        self.segment_length = segment_length
        self.leads = [1, 2, 5, 7]
        self.segment_name = 'Vent Segment'
        self.patient_dataset = None
        self.path_to_save = 'Data/Dataset'
        self.n_folds = n_folds
        self.patient_dataset, self.cached_result = self._fetch_from_cache()
        if self.cached_result is None or self.patient_dataset is None:
            print('No Cached Data Found -> Fetching From Scratch')
            self._extract_and_preprocess()

    def _fetch_from_cache(self):
        all_fold_data = []
        patient_dataset = []
        for fold_index in range(self.n_folds):
            try:
                patient_dataset = np.load(os.path.join(self.path_to_save, f'patient_dataset_{self.num_ecgs}.npy'),
                                          allow_pickle=True)
                train_x = np.load(os.path.join(self.path_to_save, f'Train_X_Fold_{fold_index + 1}_{self.num_ecgs}.npy'),
                                  allow_pickle=True)
                train_y = np.load(os.path.join(self.path_to_save, f'Train_Y_Fold_{fold_index + 1}_{self.num_ecgs}.npy'),
                                  allow_pickle=True)
                test_x = np.load(os.path.join(self.path_to_save, f'Test_X_Fold_{fold_index + 1}_{self.num_ecgs}.npy'),
                                 allow_pickle=True)
                test_y = np.load(os.path.join(self.path_to_save, f'Test_Y_Fold_{fold_index + 1}_{self.num_ecgs}.npy'),
                                 allow_pickle=True)

                train_s0_len = len(train_x[train_y == 0])
                train_s1_len = len(train_x[train_y == 1])
                test_s0_len = len(test_x[test_y == 0])
                test_s1_len = len(test_x[test_y == 1])
                print(f'Cached Fold {fold_index+1}')
                print(f'---> Train set:\n'
                      f'        |class 0| = {train_s0_len} segments \n'
                      f'        |class 1| = {train_s1_len} segments')
                print(f'---> Test set:\n'
                      f'        |class 0| = {test_s0_len} segments\n'
                      f'        |class 1| = {test_s1_len} segments\n')

                fold_dict = {'Train_X': train_x,
                             'Test_X': test_x,
                             'Train_Y': train_y,
                             'Test_Y': test_y}
                all_fold_data.append(fold_dict)
            except OSError:
                return None, None

        return patient_dataset, all_fold_data

    def _extract_and_preprocess(self):
        print(f'Extracting QT segments from {self.num_ecgs} patients ...')
        extractor = QTSegmentExtractor(verbose=False)
        extracted_segments_dict = extractor.extract_segments(num_ecgs=self.num_ecgs)
        print(f'---> {len(extracted_segments_dict)} patients processed')
        print('\nPreprocessing segments ...')
        scar_df = pd.read_excel('Data/scar_dataset.xlsx')
        patient_dataset = []
        for pid, segment_dict in extracted_segments_dict.items():
            # if pid != 10338:
            #     continue
            segments = np.array(segment_dict['segments'])
            de = scar_df.loc[scar_df['Record ID'] == pid]['DE'].values[0]
            preprocessed_segments = []
            for segment in segments:
                segment_resampled = TimeSeriesResampler(sz=self.segment_length).fit_transform(segment)
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
        self.patient_dataset = patient_dataset
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

        print(f'Saving Dataset at {self.path_to_save}')
        np.save(file=os.path.join(self.path_to_save, f'patient_dataset_{self.num_ecgs}'), arr=patient_dataset)

    def load_fold(self, train_index, test_index, fold_index: int, augment: bool = False):
        if self.cached_result is not None:
            print('Fetching fold from cache')
            return self.cached_result[fold_index]
        else:
            patient_set_train, patient_set_test = self.patient_dataset[train_index], self.patient_dataset[test_index]
            set_train = []
            set_test = []
            for patient_object in patient_set_train:
                pid = patient_object['pid']
                de = patient_object['de']
                segments = patient_object['preprocessed']
                for segment in segments:
                    segment = np.transpose(segment)
                    set_train.append((pid, de, segment))
            set_train = np.array(set_train, dtype=object)

            for patient_object in patient_set_test:
                pid = patient_object['pid']
                de = patient_object['de']
                segments = patient_object['preprocessed']
                for segment in segments:
                    segment = np.transpose(segment)
                    set_test.append((pid, de, segment))
            set_test = np.array(set_test, dtype=object)

            np.random.shuffle(set_train)
            np.random.shuffle(set_test)

            train_pids = set(set_train[:, 0])
            train_y = set_train[:, 1]
            train_x = np.array([np.array(x[2]) for x in set_train])
            test_pids = set(set_test[:, 0])
            test_y = set_test[:, 1]
            test_x = np.array([np.array(x[2]) for x in set_test])

            train_y = np.asarray(train_y).astype('float32')
            train_x = np.asarray(train_x).astype('float32')

            test_y = np.asarray(test_y).astype('float32')
            test_x = np.asarray(test_x).astype('float32')

            augmented_x_minor = []
            if augment:
                # Augmenting minority class
                size_minor = len(train_x[train_y == 0])
                size_major = len(train_x[train_y == 1])

                minority_patient_set = [p_dict for p_dict in patient_set_train if p_dict['de'] == 0]
                augmented_x_minor = self._augment_smote_v2(patient_train_set=minority_patient_set)
                augmented_y_minor = np.zeros(shape=len(augmented_x_minor))
                train_x = np.concatenate((train_x, augmented_x_minor), axis=0)
                train_y = np.concatenate((train_y, augmented_y_minor), axis=0)

                # Augment Majority Class
                # majority_patient_set = [p_dict for p_dict in patient_set_train if p_dict['de'] == 1]
                # augmented_x_major = self._augment_smote(patient_train_set=majority_patient_set)
                # sample_size = len(augmented_x_minor) + size_minor - size_major
                # if sample_size < 0:
                #     sample_size = 1
                # print(f'Selecting {sample_size} data points from augmented majority')
                # idx = np.random.choice(np.arange(augmented_x_major.shape[0]), size=sample_size)
                # augmented_x_major = augmented_x_major[idx]
                # augmented_y_major = np.ones(shape=len(augmented_x_major))
                # train_x = np.concatenate((train_x, augmented_x_major), axis=0)
                # train_y = np.concatenate((train_y, augmented_y_major), axis=0)

                c = list(zip(train_x, train_y))
                random.shuffle(c)
                train_x, train_y = zip(*c)
                train_x = np.array(train_x)
                train_y = np.array(train_y)

            print(f'Fold Exported')
            train_p0_len = len(set([x[0] for x in set_train if x[1] == 0]))
            train_p1_len = len(set([x[0] for x in set_train if x[1] == 1]))
            test_p0_len = len(set([x[0] for x in set_test if x[1] == 0]))
            test_p1_len = len(set([x[0] for x in set_test if x[1] == 1]))
            train_s0_len = len(train_x[train_y == 0])
            train_s1_len = len(train_x[train_y == 1])
            test_s0_len = len(test_x[test_y == 0])
            test_s1_len = len(test_x[test_y == 1])
            print(f'---> Train set:\n'
                  f'        |class 0| = {train_s0_len} segments ({len(augmented_x_minor)} augmented) from {train_p0_len} patients\n'
                  f'        |class 1| = {train_s1_len} segments from {train_p1_len} patients')
            print(f'---> Test set:\n'
                  f'        |class 0| = {test_s0_len} segments from {test_p0_len} patients\n'
                  f'        |class 1| = {test_s1_len} segments from {test_p1_len} patients\n')

            print(f'Saving fold to {self.path_to_save}')

            np.save(file=os.path.join(self.path_to_save, f'Train_X_Fold_{fold_index + 1}_{self.num_ecgs}'),
                    arr=train_x)
            np.save(file=os.path.join(self.path_to_save, f'Train_Y_Fold_{fold_index + 1}_{self.num_ecgs}'),
                    arr=train_y)
            np.save(file=os.path.join(self.path_to_save, f'Test_X_Fold_{fold_index + 1}_{self.num_ecgs}'),
                    arr=test_x)
            np.save(file=os.path.join(self.path_to_save, f'Test_Y_Fold_{fold_index + 1}_{self.num_ecgs}'),
                    arr=test_y)

            return {'Train_X': train_x,
                    'Test_X': test_x,
                    'Train_Y': train_y,
                    'Test_Y': test_y,
                    'Train_PIDs': train_pids,
                    'Test_PIDs': test_pids}

    def _augment_smote(self, patient_train_set):
        #  An idea: in this code you find the nearest patients based on their ECG patterns.
        #  Maybe you can find them based on their EHR (or other parameters)  ;)

        # Algorithm:
        # NOTE -> This augmentation is performed on each lead separately. The generated QT segments are stacked up on
        #         each other to create a new 96 x 4 segment.
        # Step 1 -> For each patient, find their QT distance with all other patients.
        #   How to find QT distance between p1 and p2 -> for each segment in p1, find the closest segment in p2
        #   based on DTW and store their distance. Return the mean of all distances.
        # Step 2 -> Sort the distant array and select the k nearest patients.
        # Step 3 -> Represent each neighbor patient and also the main patient by the mean of their QT segments.
        # Store them in nearest_mean_segments array.
        #   Example: if k = 5 -> nearest_mean_segments will have 6 segments of length 96
        # Step 4 -> Generate one segment by averaging the nearest_mean_segments with uniform weights.
        # Step 5 -> Let's |nearest_mean_segments| = q. Generate q segments by averaging the nearest_mean_segments with
        # following weights: rotate shift [0.5, 1-0.5/k, 1-0.5/k, 1-0.5/k ...]
        # Final Note -> k + 2 segments are generated from each patient.

        print(f'Performing augmentation on {len(patient_train_set)} patients')
        time_log = []
        k_nearest = 5

        augmented_set = []
        for patient in patient_train_set:
            new_segment = []
            time_start = time()

            for lead_index in range(4):
                patient_per_lead_dist = []
                for other_patient in patient_train_set:
                    if patient['pid'] != other_patient['pid']:
                        dist = self._get_distance(p1=patient, p2=other_patient, lead_index=lead_index)
                        patient_per_lead_dist.append((other_patient['pid'], dist, other_patient['preprocessed']))
                patient_per_lead_dist = sorted(patient_per_lead_dist, key=lambda item: item[1])
                nearest_patients = patient_per_lead_dist[:k_nearest]
                nearest_mean_segments = []

                for selected_patient in nearest_patients:
                    segments = np.array(selected_patient[2])
                    segments = segments[:, lead_index, :]  # -> will be of shape m x 96, where m is the number of hb
                    mean_segment = softdtw_barycenter(segments, gamma=0.1, max_iter=10, tol=1e-3)
                    mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
                    nearest_mean_segments.append(mean_segment)

                    # fig, ax = plt.subplots(len(segments) + 1, figsize=(5, 10))
                    # ax[0].plot(mean_segment, color='r')
                    # for i in range(len(segments)):
                    #     segment = segments[i]
                    #     ax[i+1].plot(segment)
                    # plt.show()

                segments = np.array(patient['preprocessed'])
                mean_segment = softdtw_barycenter(segments[:, lead_index, :], gamma=0.1, max_iter=10, tol=1e-3)
                mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
                nearest_mean_segments.append(mean_segment)
                nearest_mean_segments = np.array(nearest_mean_segments)

                weights = [0.5]
                for _ in range(k_nearest):
                    weights.append((1 - 0.5)/k_nearest)

                weights_rotated = [weights]
                for _ in range(k_nearest):
                    current_weights = weights_rotated[-1]
                    new_weights = np.roll(current_weights, shift=1)
                    weights_rotated.append(new_weights)
                weights_rotated = np.array(weights_rotated)

                new_lead_segments = []
                new_lead_segment_1 = softdtw_barycenter(nearest_mean_segments, gamma=3)
                new_lead_segment_1 = np.reshape(new_lead_segment_1, new_lead_segment_1.shape[0])
                new_lead_segments.append(new_lead_segment_1)
                for weights in weights_rotated:
                    new_lead_segment = softdtw_barycenter(X=nearest_mean_segments, gamma=3, weights=weights)
                    new_lead_segment = np.reshape(new_lead_segment, new_lead_segment.shape[0])
                    new_lead_segments.append(new_lead_segment)
                new_lead_segments = np.array(new_lead_segments)
                new_segment.append(new_lead_segments)
                v = 9
                # fig, ax = plt.subplots(nrows=len(new_lead_segments), ncols=2, figsize=(5, 8))
                # for i in range(len(new_lead_segments)):
                #     if i < len(nearest_mean_segments):
                #         ax[i][0].plot(nearest_mean_segments[i])
                #     ax[i][1].plot(new_lead_segments[i], color='r')

                # fig, ax = plt.subplots(len(nearest_mean_segments) + 1, figsize=(5, 8))
                # ax[0].plot(new_lead_segment_1, color='r')
                # for i in range(len(nearest_mean_segments)):
                #     segment = nearest_mean_segments[i]
                #     ax[i + 1].plot(segment)
                # plt.show()

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
            augmented_set.extend(results)
            time_end = time()
            if len(time_log) < 3:
                time_log.append(time_end - time_start)
            if len(time_log) == 3:
                elapsed_time = statistics.mean(time_log)
                print(f'---Average time elapsed per patient : {round(elapsed_time, 1)} seconds')
                print(f'---ETA = {round(elapsed_time * len(patient_train_set) - (elapsed_time * 3))} seconds')
                time_log.append(-1)
        augmented_set = np.array(augmented_set)
        # fig, ax = plt.subplots(nrows=4, ncols=7, figsize=(10, 8))
        # for i in range(len(results)):
        #     for row in range(4):
        #         ax[row][i].plot(results[i, row, :])
        # plt.show()
        return augmented_set

    def _augment_smote_v2(self, patient_train_set):
        #  An idea: in this code you find the nearest patients based on their ECG patterns.
        #  Maybe you can find them based on their EHR (or other parameters)  ;)

        # Algorithm:
        # NOTE -> This augmentation is performed on each lead separately. The generated QT segments are stacked up on
        #         each other to create a new 96 x 4 segment.
        # Step 1 -> For each patient, find their QT distance with all other patients.
        #   How to find QT distance between p1 and p2 -> for each segment in p1, find the closest segment in p2
        #   based on DTW and store their distance. Return the mean of all distances.
        # Step 2 -> Sort the distant array and select the k nearest patients.
        # Step 3 -> Represent each neighbor patient and also the main patient by the mean of their QT segments.
        # Store them in nearest_mean_segments array.
        #   Example: if k = 5 -> nearest_mean_segments will have 6 segments of length 96
        # Step 4 -> Generate one segment by averaging the nearest_mean_segments with uniform weights.
        # Step 5 -> Let's |nearest_mean_segments| = q. Generate q segments by averaging the nearest_mean_segments with
        # following weights: rotate shift [0.5, 1-0.5/k, 1-0.5/k, 1-0.5/k ...]
        # Final Note -> k + 2 segments are generated from each patient.

        print(f'Performing augmentation on {len(patient_train_set)} patients')
        time_log = []
        k_nearest = 5

        augmented_set = []
        for patient in patient_train_set:
            new_segment = []
            time_start = time()

            for lead_index in range(4):
                patient_per_lead_dist = []
                for other_patient in patient_train_set:

                    if patient['pid'] != other_patient['pid']:
                        dist = self._get_distance(p1=patient, p2=other_patient, lead_index=lead_index)
                        patient_per_lead_dist.append((other_patient['pid'], dist, other_patient['preprocessed']))
                patient_per_lead_dist = sorted(patient_per_lead_dist, key=lambda item: item[1])
                nearest_patients = patient_per_lead_dist[:k_nearest]

                nearest_mean_segments = []
                for selected_patient in nearest_patients:
                    segments = np.array(selected_patient[2])
                    segments = segments[:, lead_index, :]  # -> will be of shape m x 96, where m is the number of hb
                    mean_segment = softdtw_barycenter(segments, gamma=0.1, max_iter=10, tol=1e-3)
                    mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
                    mean_segment = self._smooth_two_third(mean_segment)
                    nearest_mean_segments.append(mean_segment)

                segments = np.array(patient['preprocessed'])
                mean_segment = softdtw_barycenter(segments[:, lead_index, :], gamma=0.1, max_iter=10, tol=1e-3)
                mean_segment = np.reshape(mean_segment, mean_segment.shape[0])
                mean_segment = self._smooth_two_third(mean_segment)
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
                    interpolate_1 = self._smooth_two_third(interpolate_1)
                    interpolate_2 = self._smooth_two_third(interpolate_2)
                    interpolate_3 = self._smooth_two_third(interpolate_3)
                    new_lead_segments.append(interpolate_1)
                    new_lead_segments.append(interpolate_2)
                    new_lead_segments.append(interpolate_3)
                    # fig, ax = plt.subplots(5, figsize=(4, 7))
                    # ax[0].plot(main_segment, color='r')
                    # ax[1].plot(neighbor_segment, color='g')
                    # ax[2].plot(interpolate_1, color='b')
                    # ax[3].plot(interpolate_2, color='b')
                    # ax[4].plot(interpolate_3, color='b')
                    # for i in range(5):
                    #     ax[i].set_xticks([])
                    #     ax[i].set_yticks([])
                    # plt.show()
                    v = 9
                new_lead_segments = np.array(new_lead_segments)
                new_segment.append(new_lead_segments)

                v = 9
                # fig, ax = plt.subplots(nrows=len(new_lead_segments), ncols=2, figsize=(5, 8))
                # for i in range(len(new_lead_segments)):
                #     if i < len(nearest_mean_segments):
                #         ax[i][0].plot(nearest_mean_segments[i])
                #     ax[i][1].plot(new_lead_segments[i], color='r')

                # fig, ax = plt.subplots(len(nearest_mean_segments) + 1, figsize=(5, 8))
                # ax[0].plot(new_lead_segment_1, color='r')
                # for i in range(len(nearest_mean_segments)):
                #     segment = nearest_mean_segments[i]
                #     ax[i + 1].plot(segment)
                # plt.show()

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
            augmented_set.extend(results)
            time_end = time()
            if len(time_log) < 2:
                time_log.append(time_end - time_start)
            if len(time_log) == 2:
                elapsed_time = statistics.mean(time_log)
                print(f'--- Average time per patient : {round(elapsed_time, 1)} seconds')
                eta = round(elapsed_time * len(patient_train_set) - (elapsed_time * 3))
                finish_time = datetime.now() + timedelta(seconds=eta)
                eta = timedelta(seconds=eta)
                print(f'--- ETA = {finish_time.hour}:{finish_time.minute}:{finish_time.second} (after {eta})\n')
                time_log.append(-1)
        augmented_set = np.array(augmented_set)
        return augmented_set


            # fig, ax = plt.subplots(nrows=4, ncols=7, figsize=(10, 8))
            # for i in range(len(results)):
            #     for row in range(4):
            #         ax[row][i].plot(results[i, row, :])
            # plt.show()

    @staticmethod
    def _smooth_two_third(interpolate_1):
        temp = interpolate_1[round(len(interpolate_1) / 3):]
        smooth_part = SignalProcessing.smooth(x=interpolate_1[round(len(interpolate_1) / 3):], window_len=5, window='hamming')
        padded_length = len(smooth_part) - len(temp)
        left_pad = round(padded_length/2)
        right_pad = padded_length - left_pad
        smooth_part = smooth_part[left_pad:-right_pad]
        interpolate_1 = np.concatenate([interpolate_1[:round(len(interpolate_1) / 3)], smooth_part])
        return interpolate_1

    @staticmethod
    def _get_distance(p1: numpy.ndarray, p2: numpy.ndarray, lead_index: int):
        p1_segments = p1['preprocessed']
        p2_segments = p2['preprocessed']

        if len(p1_segments) > 2 and len(p2_segments) > 2:

            p1_segments = np.array(p1_segments)
            p1_lead_segments = p1_segments[:, lead_index, :]
            p1_lead_segments = p1_lead_segments[np.random.choice(p1_lead_segments.shape[0], size=3, replace=False), :]

            p2_segments = np.array(p2_segments)
            p2_lead_segments = p2_segments[:, lead_index, :]
            p2_lead_segments = p2_lead_segments[np.random.choice(p2_lead_segments.shape[0], size=3, replace=False), :]

            dists = []
            for segment in p1_lead_segments:
                temp_dists = []
                for segment_other in p2_lead_segments:
                    dist = dtw(s1=segment, s2=segment_other)
                    temp_dists.append(dist)
                dists.append(min(temp_dists))
            return statistics.mean(dists)
        else:
            dists = []
            for segment in p1_segments:
                temp_dists = []
                for segment_other in p2_segments:
                    dist = dtw(s1=segment[lead_index], s2=segment_other[lead_index])
                    temp_dists.append(dist)
                dists.append(min(temp_dists))
            return statistics.mean(dists)



if __name__ == '__main__':
    n_folds = 5
    loader = DatasetLoader(size=640, n_folds=n_folds)
    kf = KFold(n_splits=n_folds)
    fold_index = 0
    for train_index, test_index in kf.split(loader.patient_dataset):
        fold_dict = loader.load_fold(train_index, test_index, fold_index=fold_index, augment=True)
        fold_index += 1

