import os
import pandas as pd
from scipy import signal
from mat4py import loadmat
import numpy as np
from numpy.linalg import norm
import statistics as stat
from functools import reduce
from tslearn.metrics import dtw
import GlobalPaths

import GlobalPaths

path_to_ecg_files = 'Data/ECG'
path_to_ecg_meta = 'Data/scar_dataset.xlsx'
path_to_pla_ann = 'Data/PLAAnnotation'


class WaveBoundaryException(Exception):
    def __init__(self, message):
        super().__init__(message)


class VectorNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)


class NoOffsetException(WaveBoundaryException):
    def __init__(self, onset_index: int, closest_dist: int):
        self.onset_index = onset_index
        self.closest_dist = closest_dist
        self.message = 'No Offset for Onset'
        super(NoOffsetException, self).__init__(self.message)


class Util:

    @staticmethod
    def get_offset(onset: int, offset_list: [int], frequency: int, wave: str, hb_interval: int = None):
        threshold = round(0.5 * frequency)
        if wave == 'T' or wave == 'P':
            threshold = round(1 * frequency)
        elif wave == 'QT':
            threshold = round(0.9 * hb_interval)

        closest_index = -1
        if onset >= max(offset_list):
            raise NoOffsetException(onset_index=onset, closest_dist=-1)
        for index in range(len(offset_list)):
            diff = abs(offset_list[index] - onset)
            if diff < threshold and onset < offset_list[index]:
                threshold = diff
                closest_index = index
        if closest_index == -1:
            closest_dist = min([x-onset for x in offset_list if x > onset])
            raise NoOffsetException(onset_index=onset, closest_dist=closest_dist)
        return offset_list[closest_index]

    @staticmethod
    def get_min_distance(target: int, input_list: [int]):
        temp = [abs(x - target) for x in input_list]
        return min(temp)

    @staticmethod
    def get_closest_ann(ann_source: int, input_list: [(int, int)]):
        min_diff = 5000
        closest_match = ()
        for item in input_list:
            if min_diff > abs(ann_source - item[0]):
                min_diff = abs(ann_source - item[0])
                closest_match = item

        return closest_match

    @staticmethod
    def get_ecg_list() -> [int]:
        ann_names = [f for f in os.listdir(path_to_pla_ann) if not f.startswith('.')]
        ecg_ids = []
        for ann_name in ann_names:
            ecg_ids.append(int(ann_name.split('_')[0]))
        ecg_ids = list(set(ecg_ids))
        return ecg_ids

    @staticmethod
    def get_lead_name(index: int):
        names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'e']
        return names[index]

    @staticmethod
    def get_lead_id(lead_name: str):
        names = {'I':0, 'II':1, 'III':2, 'aVR':3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11, 'e':12}
        return names[lead_name]

    @staticmethod
    def get_qt_distance(p1: np.ndarray, p2: np.ndarray, lead_index: int):
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
            return stat.mean(dists)
        else:
            dists = []
            for segment in p1_segments:
                temp_dists = []
                for segment_other in p2_segments:
                    dist = dtw(s1=segment[lead_index], s2=segment_other[lead_index])
                    temp_dists.append(dist)
                dists.append(min(temp_dists))
            return stat.mean(dists)


class Loader:

    @staticmethod
    def get_ecg_pid_pair_list():
        meta = pd.read_excel(GlobalPaths.cached_scar_ecg_meta)
        ecg_ids = list(meta['ECG ID'].values)
        pid_list = list(meta['Record_ID'].values)
        frequency_list = list(meta['Sample Base'].values)
        return ecg_ids, pid_list, frequency_list

    @staticmethod
    def fast_get_ecg(ecg_id: int, frequency: int, denoise=True) -> pd.DataFrame:
        path = os.path.join(GlobalPaths.ecg, str(ecg_id) + '.csv')
        ecg = pd.read_csv(filepath_or_buffer=path,
                          header=None, skiprows=1,
                          names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
        if denoise:
            for lead in ecg:
                lead_ecg = ecg[lead]
                filtered = SignalProcessing.filter(sig=lead_ecg, frequency=frequency)
                ecg[lead] = filtered
        return ecg

    # @staticmethod
    # def get_ecg(ecg_id: int, denoise=True) -> [pd.DataFrame, int]:
    #     meta = pd.read_excel(path_to_ecg_meta)
    #     try:
    #         frequency = int(meta.loc[meta['First ECG'] == ecg_id-1]['Frequency'].values[0])
    #         pid = int(meta.loc[meta['First ECG'] == ecg_id - 1]['Record ID'].values[0])
    #         path = os.path.join(path_to_ecg_files, str(ecg_id) + '.csv')
    #         ecg = pd.read_csv(filepath_or_buffer=path,
    #                           header=None, skiprows=1,
    #                           names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'e'])
    #         ecg.drop(ecg.columns[len(ecg.columns) - 1], axis=1, inplace=True)
    #
    #         if denoise:
    #             for lead in ecg:
    #                 lead_ecg = ecg[lead]
    #                 filtered = SignalProcessing.filter(sig=lead_ecg, frequency=frequency)
    #                 ecg[lead] = filtered
    #         return [ecg, frequency, pid]
    #     except KeyError:
    #         assert False, 'Could not find meta data for ECG ' + str(ecg_id)
    #     except IndexError:
    #         assert False, 'ECG ' + str(ecg_id) + ' does not have sample frequency.'

    @staticmethod
    def _correct_annotations(ann_list):
        result = []
        for lead_ann in ann_list:
            if type(lead_ann) is not list:
                lead_ann = [lead_ann]
            lead_result = []
            for ann in lead_ann:
                if type(ann) is list:
                    if len(ann) == 0:
                        continue
                    lead_result.append([ann[0] - 1, ann[1] - 1])
                else:
                    lead_result.append(ann-1)
            result.append(lead_result)
        return result


    @staticmethod
    def get_annotations(ecg_id: int) -> [pd.DataFrame]:
        p_ann_name = str(ecg_id) + '_P.mat'
        qrs_ann_name = str(ecg_id) + '_QRS.mat'
        t_ann_name = str(ecg_id) + '_T.mat'
        p_ann = loadmat(os.path.join(GlobalPaths.pla_annotation, p_ann_name))
        qrs_ann = loadmat(os.path.join(GlobalPaths.pla_annotation, qrs_ann_name))
        t_ann = loadmat(os.path.join(GlobalPaths.pla_annotation, t_ann_name))

        p_start = Loader._correct_annotations(p_ann['P_anns']['onset'])
        p_peak = Loader._correct_annotations(p_ann['P_anns']['peak'])
        p_end = Loader._correct_annotations(p_ann['P_anns']['offset'])

        qrs_start = Loader._correct_annotations(qrs_ann['QRS_anns']['onset'])
        qrs_peak = Loader._correct_annotations(qrs_ann['QRS_anns']['peak'])
        qrs_fragmented = Loader._correct_annotations(qrs_ann['QRS_anns']['fragmented'])
        qrs_end = Loader._correct_annotations(qrs_ann['QRS_anns']['offset'])

        t_start = Loader._correct_annotations(t_ann['T_anns']['onset'])
        t_peak = Loader._correct_annotations(t_ann['T_anns']['peak'])
        t_end = Loader._correct_annotations(t_ann['T_anns']['offset'])

        final_ann = []
        for lead in range(12):
            row = [p_start[lead], p_peak[lead], p_end[lead],
                   qrs_start[lead], qrs_peak[lead], qrs_end[lead], qrs_fragmented[lead],
                   t_start[lead], t_peak[lead], t_end[lead]]
            final_ann.append(row)

        final_df = pd.DataFrame(data=final_ann)
        final_df.columns = ['P Start', 'P Peak', 'P End',
                            'QRS Start', 'QRS Peak', 'QRS End', 'QRS Fragmented',
                            'T Start', 'T Peak', 'T End']

        return final_df

    # @staticmethod
    # def get_gt_fqrs(ecg_id: int) -> [bool, bool, bool]:
    #     fqrs_gt = pd.read_excel(path_to_fqrs_gt)
    #     fqrs_gt = fqrs_gt.loc[fqrs_gt['ECG ID'] == ecg_id]
    #     if fqrs_gt.empty:
    #         raise KeyError('ECG ' + str(ecg_id) + ' not in ground-truth set')
    #     fqrs = fqrs_gt.iloc[:, 3:15].values[0]
    #     rsr = fqrs_gt.iloc[:, 15:27].values[0]
    #     jwave = fqrs_gt.iloc[:, 27:].values[0]
    #     has_fqrs = False
    #     has_rsr = False
    #     has_j = False
    #     if sum(fqrs) > 0:
    #         has_fqrs = True
    #     if sum(rsr) > 0:
    #         has_rsr = True
    #     if sum(jwave) > 0:
    #         has_j = True
    #
    #     return [has_fqrs, has_rsr, has_j]


class SignalProcessing:
    @staticmethod
    def filter(sig: [int], frequency: int, ):
        nyq = frequency / 2  # Nyquist Frequency
        Wp = [1/nyq, 100/nyq]  # Pass band (Normalised)
        Ws = [0.5 /nyq, 110/nyq]  # Stop band (Normalised)
        order, Wn = signal.cheb2ord(Wp, Ws, gpass=10, gstop=30)
        b, a = signal.cheby2(order, 30, Wn, btype='bandpass', output='ba')
        filtered = signal.filtfilt(b=b, a=a, x=sig)
        return filtered

    @staticmethod
    def smooth_two_third(interpolate_1):
        temp = interpolate_1[round(len(interpolate_1) / 3):]
        smooth_part = SignalProcessing.smooth(x=interpolate_1[round(len(interpolate_1) / 3):], window_len=5, window='hamming')
        padded_length = len(smooth_part) - len(temp)
        left_pad = round(padded_length/2)
        right_pad = padded_length - left_pad
        smooth_part = smooth_part[left_pad:-right_pad]
        interpolate_1 = np.concatenate([interpolate_1[:round(len(interpolate_1) / 3)], smooth_part])
        return interpolate_1

    @staticmethod
    def smooth(x, window_len=3, window='hanning'):

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    @staticmethod
    def get_peaks(segment: [float]):
        amp_baseline = stat.mean([segment[0], segment[-1]])
        max_point = (list(segment).index(max(segment)), max(segment))
        min_point = (list(segment).index(min(segment)), min(segment))
        max_base_div = max_point[1] - amp_baseline
        global_extremum = max_point[0]
        # if abs(min_point[1]) > abs(max_point[1]):
        #     max_base_div = amp_baseline - min_point[1]
        #     global_extremum = min_point[0]

        second_level_maxima, bb = signal.find_peaks(x=segment, prominence=max_base_div/2)
        third_level_maxima, bb = signal.find_peaks(x=segment, prominence=(max_base_div / 3, max_base_div / 2))
        fourth_level_maxima, bb = signal.find_peaks(x=segment, prominence=(max_base_div / 4, max_base_div / 3))
        fifth_level_maxima, bb = signal.find_peaks(x=segment, prominence=(max_base_div / 5, max_base_div / 4))
        sixth_level_maxima, bb = signal.find_peaks(x=segment, prominence=(max_base_div / 10, max_base_div / 5))
        seventh_level_maxima, bb = signal.find_peaks(x=segment, prominence=(max_base_div / 20, max_base_div / 10))
        last_level_maxima, bb = signal.find_peaks(x=segment, prominence=(max_base_div / 30, max_base_div / 20))

        set1 = set([global_extremum])
        set2 = set(second_level_maxima)
        set3 = set(third_level_maxima)
        set4 = set(fourth_level_maxima)
        set5 = set(fifth_level_maxima)
        set6 = set(sixth_level_maxima)
        set7 = set(seventh_level_maxima)
        set8 = set(last_level_maxima)
        set2 = set2 - set1
        set3 = set3 - set1
        set4 = set4 - set1
        set5 = set5 - set1
        set6 = set6 - set1
        set7 = set7 - set1

        return [list(set1), list(set2), list(set3), list(set4), list(set5), list(set6), list(set7), list(set8)]

    def get_significant_points(self, segment: [int], threshold: float):
        significant_points = []
        p_start = (0, segment[0])
        p_end = (len(segment) - 1, segment[-1])
        p_start = np.asarray(p_start)
        p_end = np.asarray(p_end)

        max_distance = 0
        max_index = -1
        for i in range(len(segment)):
            point = (i, segment[i])
            point = np.asarray(point)
            distance = norm(np.cross(p_end - p_start, p_start - point)) / norm(p_end - p_start)
            if distance > max_distance:
                max_distance = distance
                max_index = i

        if max_distance > threshold:
            significant_points.append(max_index)

            left_segment = segment[:max_index]
            right_segment = segment[max_index+1:]

            significant_points_left = self.get_significant_points(segment=left_segment, threshold=threshold)
            significant_points_right = self.get_significant_points(segment=right_segment, threshold=threshold)
            if significant_points_right is not None:
                significant_points_right = [x + max_index for x in significant_points_right]

            if significant_points_left is not None:
                significant_points.extend(significant_points_left)
            if significant_points_right is not None:
                significant_points.extend(significant_points_right)
            return significant_points


'''
/*******************************************************************************
 * Copyright (C) 2018 Francois Petitjean
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
'''

def performDBA(series, n_iterations=10):
    n_series = len(series)
    max_length = reduce(max, map(len, series))

    cost_mat = np.zeros((max_length, max_length))
    delta_mat = np.zeros((max_length, max_length))
    path_mat = np.zeros((max_length, max_length), dtype=np.int8)

    medoid_ind = approximate_medoid_index(series, cost_mat, delta_mat)
    center = series[medoid_ind]

    for i in range(0, n_iterations):
        center = DBA_update(center, series, cost_mat, path_mat, delta_mat)

    return center


def approximate_medoid_index(series,cost_mat,delta_mat):
    if len(series)<=50:
        indices = range(0,len(series))
    else:
        indices = np.random.choice(range(0,len(series)),50,replace=False)

    medoid_ind = -1
    best_ss = 1e20
    for index_candidate in indices:
        candidate = series[index_candidate]
        ss = sum_of_squares(candidate,series,cost_mat,delta_mat)
        if(medoid_ind==-1 or ss<best_ss):
            best_ss = ss
            medoid_ind = index_candidate
    return medoid_ind

def sum_of_squares(s,series,cost_mat,delta_mat):
    return sum(map(lambda t:squared_DTW(s,t,cost_mat,delta_mat),series))

def DTW(s,t,cost_mat,delta_mat):
    return np.sqrt(squared_DTW(s,t,cost_mat,delta_mat))

def squared_DTW(s,t,cost_mat,delta_mat):
    s_len = len(s)
    t_len = len(t)
    length = len(s)
    fill_delta_mat_dtw(s, t, delta_mat)
    cost_mat[0, 0] = delta_mat[0, 0]
    for i in range(1, s_len):
        cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]

    for j in range(1, t_len):
        cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]

    for i in range(1, s_len):
        for j in range(1, t_len):
            diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
            if(diag <=left):
                if(diag<=top):
                    res = diag
                else:
                    res = top
            else:
                if(left<=top):
                    res = left
                else:
                    res = top
            cost_mat[i, j] = res+delta_mat[i, j]
    return cost_mat[s_len-1,t_len-1]

def fill_delta_mat_dtw(center, s, delta_mat):
    slim = delta_mat[:len(center),:len(s)]
    np.subtract.outer(center, s,out=slim)
    np.square(slim, out=slim)


def DBA_update(center, series, cost_mat, path_mat, delta_mat):
    options_argmin = [(-1, -1), (0, -1), (-1, 0)]
    updated_center = np.zeros(center.shape)
    n_elements = np.array(np.zeros(center.shape), dtype=int)
    center_length = len(center)
    for s in series:
        s_len = len(s)
        fill_delta_mat_dtw(center, s, delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        path_mat[0, 0] = -1

        for i in range(1, center_length):
            cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]
            path_mat[i, 0] = 2

        for j in range(1, s_len):
            cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]
            path_mat[0, j] = 1

        for i in range(1, center_length):
            for j in range(1, s_len):
                diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
                if(diag <=left):
                    if(diag<=top):
                        res = diag
                        path_mat[i,j] = 0
                    else:
                        res = top
                        path_mat[i,j] = 2
                else:
                    if(left<=top):
                        res = left
                        path_mat[i,j] = 1
                    else:
                        res = top
                        path_mat[i,j] = 2

                cost_mat[i, j] = res+delta_mat[i, j]

        i = center_length-1
        j = s_len-1

        while(path_mat[i, j] != -1):
            updated_center[i] += s[j]
            n_elements[i] += 1
            move = options_argmin[path_mat[i, j]]
            i += move[0]
            j += move[1]
        assert(i == 0 and j == 0)
        updated_center[i] += s[j]
        n_elements[i] += 1

    return np.divide(updated_center, n_elements)

def main():
    #generating synthetic data
    n_series = 20
    length = 200

    series = list()
    padding_length=30
    indices = range(0, length-padding_length)
    main_profile_gen = np.array([np.sin(2*np.pi*j/len(indices)) for j in indices])
    randomizer = lambda j:np.random.normal(j,0.02)
    randomizer_fun = np.vectorize(randomizer)
    for i in range(0,n_series):
        n_pad_left = np.random.randint(0,padding_length)
        #adding zero at the start or at the end to shif the profile
        series_i = np.pad(main_profile_gen,(n_pad_left,padding_length-n_pad_left),mode='constant',constant_values=0)
        #chop some of the end to prove it can work with multiple lengths
        l = np.random.randint(length-20,length+1)
        series_i = series_i[:l]
        #randomize a bit
        series_i = randomizer_fun(series_i)

        series.append(series_i)
    series = np.array(series)

    #plotting the synthetic data
    for s in series:
        plt.plot(range(0,len(s)), s)
    plt.draw()

    #calculating average series with DBA
    average_series = performDBA(series)

    #plotting the average series
    plt.figure()
    plt.plot(range(0,len(average_series)), average_series)
    plt.show()

if __name__== "__main__":
    main()