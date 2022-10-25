import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utility import Loader, Util, NoOffsetException
import GlobalPaths


class QTSegmentExtractor:
    def __init__(self, ecg_dir_path: str, ann_dir_path: str, metadata_path: str, verbose: bool = True):
        self.ecg_dir_path = ecg_dir_path
        self.ann_dir_path = ann_dir_path
        self.metadata_path = metadata_path

        self.ecg_ids, self.pids, self.frequency_list = Loader.metadata(metadata_path=metadata_path)
        self.column_names = ['P Start', 'P End', 'QRS Start', 'QRS End', 'T Start', 'T End']
        # self.selected_leads = [1, 2, 5, 7]
        self.selected_leads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.verbose = verbose

    def extract_segments(self):
        results = {}
        for i in range(len(self.ecg_ids)):
            ecg_id = self.ecg_ids[i]
            pid = self.pids[i]
            frequency = self.frequency_list[i]
            segment_dict = self._parse_annotation(ecg_id=ecg_id, pid=pid, frequency=frequency)

            if pid in segment_dict:
                p_results_dict = segment_dict[pid]
                results[pid] = p_results_dict
            else:
                print(f'Skipping {pid}')
            if self.verbose and i != 0 and i % 50 == 0:
                print(f'--- {round(i/len(self.ecg_ids) * 100)}% Done')

        return results

    # def extract_qrs(self):
    #     results = {}
    #     for i in range(len(self.ecg_ids)):
    #         ecg_id = self.ecg_ids[i]
    #         pid = self.pids[i]
    #         frequency = self.frequency_list[i]
    #         segment_dict = self._parse_annotation(ecg_id=ecg_id, pid=pid, frequency=frequency)
    #         if pid in segment_dict:
    #             p_results_dict = segment_dict[pid]
    #             results[pid] = p_results_dict
    #         else:
    #             print(f'Skipping {pid}')
    #         if self.verbose and i != 0 and i % 50 == 0:
    #             print(f'--- {round(i/len(self.ecg_ids) * 100)}% Done')
    #
    #     return results

    def _parse_annotation(self, ecg_id: int, pid: int, frequency: int):
        results = {}
        ecg = Loader.ecg(path=os.path.join(self.ecg_dir_path, str(ecg_id) + '.csv'), frequency=frequency)
        try:
            ann = Loader.pla_annotation(ann_folder_path=self.ann_dir_path, ecg_id=ecg_id)
        except FileNotFoundError:
            # print('No annotation found for ECG = ' + str(ecg_id))
            return {}

        for index, row in ann.iterrows():
            lead_ecg = ecg[Util.get_lead_name(index=index)].values
            for col_name in self.column_names:
                lead_bound_candidates = row[col_name]
                corrected_bound = self._find_correct_bound(lead_ecg, lead_bound_candidates)
                ann.at[index, col_name] = corrected_bound

        consensus_threshold_qrs = round(frequency * 0.0625)
        consensus_threshold_t = round(consensus_threshold_qrs * 1.5)

        qrs_peaks = ann['QRS Peak'].values
        qrs_ons = ann['QRS Start'].values
        qrs_ends = ann['QRS End'].values
        t_ends = ann['T End'].values
        _, qrs_peaks = self._vote_among_leads(annotations=qrs_peaks, consensus_threshold=consensus_threshold_qrs)
        _, qrs_ons = self._vote_among_leads(annotations=qrs_ons, consensus_threshold=consensus_threshold_qrs)
        _, qrs_ends = self._vote_among_leads(annotations=qrs_ends, consensus_threshold=consensus_threshold_qrs)
        _, t_ends = self._vote_among_leads(annotations=t_ends, consensus_threshold=consensus_threshold_t)

        sum_hb_intervals = 0
        count_hb = 0
        for lead_qrs_peak in qrs_peaks:
            for i in range(len(lead_qrs_peak) - 1):
                hb_interval = lead_qrs_peak[i+1] - lead_qrs_peak[i]
                if hb_interval < 2.03 * frequency:  # -> It was 2, but because of PID 10844 I changed it to 2.03
                    sum_hb_intervals += hb_interval
                    count_hb += 1
        if count_hb == 0:
            return {}
        hb_interval = round(sum_hb_intervals/count_hb)

        count_qt = 0
        all_lead_bounds = []

        for lead in self.selected_leads:
            qt_segments = []
            qt_bounds = []

            lead_qrs_onsets = qrs_ons[lead]
            lead_t_offsets = t_ends[lead]

            for qrs_onset in lead_qrs_onsets:
                try:
                    offset = Util.get_offset(onset=qrs_onset, offset_list=lead_t_offsets,
                                             frequency=frequency,
                                             wave='QT',
                                             hb_interval=hb_interval)
                    qt_segment = list(ecg.iloc[qrs_onset:offset + 1, lead].values)
                    qt_segments.append(qt_segment)
                    qt_bounds.append([qrs_onset, offset])
                    count_qt += 1
                except NoOffsetException:
                    v = 9
                    pass
            all_lead_bounds.append(qt_bounds)
        outlier_bounds = {'I': [], 'II': [], 'III': [],
                          'aVR': [], 'aVL': [], 'aVF': [],
                          'V1': [], 'V2': [], 'V3': [], 'V4': [], 'V5': [], 'V6': []}
        is_not_equal, min_hb, min_hb_index = self._hb_not_equal(all_lead_bounds)
        if is_not_equal:
            for lead_index in range(len(self.selected_leads)):
                for bound in all_lead_bounds[lead_index]:
                    for second_lead_index in range(len(self.selected_leads)):
                        if second_lead_index != lead_index:
                            # Find the closest onset/offset in the other lead
                            max_dist_onset = 10000
                            max_dist_offset = 10000
                            closest_bound = None
                            for other_lead_bound in all_lead_bounds[second_lead_index]:
                                onset_diff = abs(other_lead_bound[0] - bound[0])
                                offset_diff = abs(other_lead_bound[1] - bound[1])
                                if onset_diff < max_dist_onset and offset_diff < max_dist_offset:
                                    max_dist_onset = onset_diff
                                    max_dist_offset = offset_diff
                                    closest_bound = other_lead_bound
                            if closest_bound is None:
                                continue
                            onset_diff = abs(closest_bound[0] - bound[0])
                            offset_diff = abs(closest_bound[1] - bound[1])
                            # If the closest onset/offset in other lead is beyond threshold, bound is outlier
                            if (onset_diff > round(0.3 * frequency) or offset_diff > round(0.3 * frequency)) and abs(bound[1] - bound[0]) > round(0.1 * frequency):
                                lead_name = Util.get_lead_name(self.selected_leads[lead_index])
                                interval_already_exists = False
                                for out_lead_name, outlier_list in outlier_bounds.items():
                                    for y in outlier_list:
                                        shared_interval = self._get_shared_interval(x=bound, y=y)
                                        if shared_interval > 0:
                                            interval_already_exists = True
                                            break
                                if not interval_already_exists:
                                    outlier_bounds[lead_name].append(bound)

        if not is_not_equal:
            lead_index = 0
            bounds_selected = []
            for bound in all_lead_bounds[lead_index]:
                closest_bounds = [bound]
                for second_lead_index in range(len(self.selected_leads)):
                    if second_lead_index != lead_index:
                        # Find the closest onset/offset in the other lead
                        max_dist_onset = 10000
                        max_dist_offset = 10000
                        closest_bound = None
                        for other_lead_bound in all_lead_bounds[second_lead_index]:
                            onset_diff = abs(other_lead_bound[0] - bound[0])
                            offset_diff = abs(other_lead_bound[1] - bound[1])
                            if onset_diff < max_dist_onset and offset_diff < max_dist_offset:
                                max_dist_onset = onset_diff
                                max_dist_offset = offset_diff
                                closest_bound = other_lead_bound
                        closest_bounds.append(closest_bound)
                # Now you have 4 onset/offset pais -> find the widest one
                onset = closest_bounds[0][0]
                offset = closest_bounds[0][1]
                for closest_bound in closest_bounds:
                    if closest_bound[0] < onset:
                        onset = closest_bound[0]
                    if closest_bound[1] > offset:
                        offset = closest_bound[1]
                if offset - onset > round(0.9 * hb_interval):  # check if selected QT interval is more than 1 second
                    onset = closest_bounds[0][0]
                    offset = closest_bounds[0][1]
                    for closest_bound in closest_bounds:
                        if closest_bound[0] < onset:
                            onset = closest_bound[0]
                        if closest_bound[1] < offset:
                            offset = closest_bound[1]
                interval = offset - onset
                if interval <= round(0.9 * hb_interval):
                    bounds_selected.append([onset, offset])
                    # qt_segments = []
                    # for lead_id in selected_leads:
                    #     lead_qt_segment = np.array(ecg.iloc[onset:offset + 1, lead_id].values)
                    #     qt_segments.append(lead_qt_segment)
                    # qt_segments = np.array(qt_segments)
                    # if pid in results:
                    #     results[pid].append(qt_segments)
                    # else:
                    #     results[pid] = [qt_segments]
        else:
            bounds_selected = []
            for bound in all_lead_bounds[min_hb_index]:
                closest_bounds = [bound]
                for second_lead_index in range(len(self.selected_leads)):
                    if second_lead_index != min_hb_index:
                        # Find the closest onset/offset in the other lead
                        max_dist_onset = 10000
                        max_dist_offset = 10000
                        closest_bound = None
                        for other_lead_bound in all_lead_bounds[second_lead_index]:
                            onset_diff = abs(other_lead_bound[0] - bound[0])
                            offset_diff = abs(other_lead_bound[1] - bound[1])
                            if onset_diff < max_dist_onset and offset_diff < max_dist_offset:
                                max_dist_onset = onset_diff
                                max_dist_offset = offset_diff
                                closest_bound = other_lead_bound
                        if closest_bound is not None:
                            closest_bounds.append(closest_bound)
                # Now you have 4 onset/offset pais -> find the widest one
                onset = closest_bounds[0][0]
                offset = closest_bounds[0][1]
                for closest_bound in closest_bounds:
                    if closest_bound[0] < onset:
                        onset = closest_bound[0]
                    if closest_bound[1] > offset:
                        offset = closest_bound[1]
                if offset - onset > round(0.9 * hb_interval):  # check if selected QT interval is more than 1 second
                    onset = closest_bounds[0][0]
                    offset = closest_bounds[0][1]
                    for closest_bound in closest_bounds:
                        if closest_bound[0] < onset:
                            onset = closest_bound[0]
                        if closest_bound[1] < offset:
                            offset = closest_bound[1]
                interval = offset - onset
                if interval <= round(0.9 * hb_interval):
                    bounds_selected.append([onset, offset])

            selected_outlier_bound_list = []
            for outlier_bound_list in outlier_bounds.values():
                for outlier_bound in outlier_bound_list:
                    for y in bounds_selected:
                        if self._get_shared_interval(x=outlier_bound, y=y) == 0:
                            selected_outlier_bound_list.append(outlier_bound)
            bounds_selected.extend(selected_outlier_bound_list)

        final_bounds = []
        threshold = round(0.1 * hb_interval)
        for bound in bounds_selected:
            has_identical = False
            for final_bound in final_bounds:

                if bound[0] - threshold < final_bound[0] < bound[0] + threshold or \
                        bound[1] - threshold < final_bound[1] < bound[1] + threshold:
                    has_identical = True
                    break
            if not has_identical:
                final_bounds.append(bound)

        bounds_selected = final_bounds
        if len(bounds_selected) > 0:
            sum_intervals = 0
            num_intervals = 0
            for bound in bounds_selected:
                interval = bound[1] - bound[0]
                sum_intervals += interval
                num_intervals += 1

            mean_interval = round(sum_intervals/num_intervals)
            final_bound_list = []
            for bound in bounds_selected:
                new_bound = [bound[0], bound[0] + mean_interval]
                if new_bound[1] < ecg.shape[0]:
                    final_bound_list.append(new_bound)

            final_bound_list = sorted(final_bound_list, key=lambda item: item[0])

            qt_distances = []
            for i in range(len(final_bound_list) - 1):
                bound_current = final_bound_list[i]
                bound_next = final_bound_list[i+1]
                dist = bound_next[0] - bound_current[1]
                qt_distances.append(dist)

            bound_index = 0
            for bound in final_bound_list:
                can_extract = True
                if bound_index < len(qt_distances) and qt_distances[bound_index] < round((1/12)*hb_interval):
                    can_extract = False
                bound_index += 1

                if can_extract:
                    onset = bound[0]
                    offset = bound[1]
                    qt_segments = []
                    for lead_id in self.selected_leads:
                        lead_qt_segment = np.array(ecg.iloc[onset:offset + 1, lead_id].values)
                        qt_segments.append(lead_qt_segment)
                    qt_segments = np.array(qt_segments)
                    if pid in results:
                        results[pid].append(qt_segments)
                    else:
                        results[pid] = [qt_segments]

            if pid in results:
                segments = results[pid]
                results[pid] = {'ecg_denoised': ecg,
                                'segments': segments,
                                'hb_interval': hb_interval,
                                'qt_distances': qt_distances,
                                'frequency': frequency,
                                'ecg_id': ecg_id}
            else:
                print(f'No segments extracted from ECG = {ecg_id}, PID = {pid}')
            return results
        else:
            return {}

    @staticmethod
    def is_identical(ts1: [float], ts2: [float]):
        if len(ts1) != len(ts2):
            return False
        for i in range(len(ts1)):
            if ts1[i] != ts2[i]:
                return False
        return True

    @staticmethod
    def _get_shared_interval(x, y):
        if y[0] > x[1] or y[1] < x[0]:
            return 0
        z = max(x[0], y[0])
        t = min(x[1], y[1])
        return t - z + 1

    @staticmethod
    def _hb_not_equal(all_lead_bounds):
        min_hb = 1000
        min_hb_index = -1
        not_equal_found = False
        prev_n_segments = 0
        index = 0
        for item in all_lead_bounds:
            if len(item) < min_hb and len(item) != 0:
                min_hb = len(item)
                min_hb_index = index
            index += 1

            n_segments = len(item)
            if prev_n_segments == 0:
                prev_n_segments = n_segments
            else:
                if prev_n_segments != n_segments:
                    not_equal_found = True
                    break
        return not_equal_found, min_hb, min_hb_index

    @staticmethod
    def _find_correct_bound(lead_ecg, lead_ann):
        result = []
        for item in lead_ann:
            if type(item) is list:
                volt1 = lead_ecg[item[0]]
                volt2 = lead_ecg[item[1]]
                if abs(volt1) < abs(volt2):
                    result.append(item[0])
                else:
                    result.append(item[1])
            else:
                result.append(item)
        return result

    @staticmethod
    def _vote_among_leads(annotations, consensus_threshold: int):
        total = []
        for i in range(len(annotations)):
            ann_with_score = []
            for j in range(len(annotations[i])):
                score = 0
                ann_item = annotations[i][j]
                for k in range(len(annotations)):
                    if k == i:
                        continue
                    lead_ann = annotations[k]
                    if len(lead_ann) > 0:
                        min_diff = Util.get_min_distance(ann_item, lead_ann)
                        if min_diff < consensus_threshold:
                            score += 1
                ann_with_score.append((ann_item, score))
            total.append(ann_with_score)

        for i in range(len(total)):
            for j in range(len(total[i])):
                ann_item = total[i][j]
                score = ann_item[1]
                if score >= 8:
                    continue
                else:
                    high_conf_matches = []
                    for k in range(len(total)):
                        if k == i:
                            continue
                        lead_ann = total[k]
                        if len(lead_ann) > 0:
                            match_ann = Util.get_closest_ann(ann_source=ann_item[0], input_list=lead_ann)
                            if match_ann[1] >= 8 and match_ann[1] != 12:
                                high_conf_matches.append(match_ann)
                    if len(high_conf_matches) == 0:
                        # total[i][j] = (ann_item[0], -2)
                        pass
                    else:
                        consensus_ann = round(
                            sum(ann_match for ann_match, score_match in high_conf_matches) / len(high_conf_matches))
                        if j - 1 >= 0 and abs(total[i][j - 1][0] - consensus_ann) <= consensus_threshold:
                            total[i][j] = (-1, -1)
                        elif j + 1 < len(total[i]) and abs(total[i][j + 1][0] - consensus_ann) <= consensus_threshold:
                            total[i][j] = (-1, -1)
                        else:
                            total[i][j] = (consensus_ann, 12)
        # Union across 12 leads
        total_new = []
        for i in range(len(total)):
            row_ann = []
            for j in range(len(total[i])):
                ann_item = total[i][j]
                score = ann_item[1]
                if score == -1:
                    continue
                else:
                    row_ann.append(ann_item)
            total_new.append(row_ann)

        for i in range(len(total_new)):
            total_new[i].sort(key=lambda item: item[0])

        total_new_no_score = []
        for lead_ann in total_new:
            row_temp = []
            for item in lead_ann:
                row_temp.append(item[0])
            total_new_no_score.append(row_temp)

        return total_new, total_new_no_score