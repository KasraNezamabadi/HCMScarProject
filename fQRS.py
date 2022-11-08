import random
import statistics

import numpy as np
import pandas as pd
import scipy
import collections
import scipy.stats.mstats_basic
from scipy import signal
from random import shuffle
from random import choice
from scipy.stats import norm, ttest_ind
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
from QTSegmentExtractor import QTSegmentExtractor
import GlobalPaths
from Utility import Util, SignalProcessing
from sklearn.metrics import classification_report
from scipy.stats import linregress
from matplotlib.pyplot import figure
import warnings
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from math import sin, cos, radians
from DataManagement import EHRFeatureSelection
from heapq import nsmallest
from imblearn.over_sampling import SMOTENC, ADASYN
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

import tensorflow
from tensorflow import keras

# import tensorflow

# from tensorflow import keras

# import keras
# from keras.layers import Dense

warnings.filterwarnings("ignore")

notch_prom_threshold = 0.05
non_terminal_notch_prom = 0.1
wave_prom_threshold = 0.05


class Wave:
    def __init__(self, peak_index: int, prominence: float, width: float, amp: float = None):
        self.peak_index = peak_index
        self.prominence = prominence
        self.width = width
        self.amp = amp


class QRSComplex:
    def __init__(self, segment_raw: [float], segment_norm: [float], interpret: dict, base_amp: float, width: int = None,
                 duration: float = None, energy: float = None):
        self.segment_raw = segment_raw
        self.segment_norm = segment_norm
        self.interpret = interpret
        self.base_amp = base_amp
        self.width = width
        self.duration = duration
        self.energy = energy

        for wave_name in self.interpret:
            self.get_wave(name=wave_name)

    # get_wave returns the actual prominence of waves and notches as seen in the original signal (not the normed sig).
    def get_wave(self, name: str):  # Returns None or [] if the wave does not exist.
        if name == 'notches':
            notches = []
            for notch in self.interpret[name]:
                prom = signal.peak_prominences(x=self.segment_raw, peaks=[notch.peak_index])[0][0]
                notches.append(Wave(peak_index=notch.peak_index,
                                    prominence=prom,
                                    width=notch.width,
                                    amp=self.segment_raw[notch.peak_index]))
            return notches
        elif name == 'terminal_notches':
            notches = []
            s = self.get_wave(name='S')
            for notch in self.interpret['notches']:
                if notch.peak_index > round(0.6 * len(self.segment_norm)) or (
                        s is not None and notch.peak_index > s.peak_index):
                    prom = signal.peak_prominences(x=self.segment_raw, peaks=[notch.peak_index])[0][0]
                    notches.append(Wave(peak_index=notch.peak_index,
                                        prominence=prom,
                                        width=notch.width,
                                        amp=self.segment_raw[notch.peak_index]))
            return notches
        elif name == 'non_terminal_notches':
            notches = []
            s = self.get_wave(name='S')
            for notch in self.interpret['notches']:
                if notch.prominence > non_terminal_notch_prom and \
                        ((s is None and notch.peak_index < round(0.6 * len(self.segment_norm))) or (
                                s is not None and notch.peak_index < s.peak_index)):
                    prom = signal.peak_prominences(x=self.segment_raw, peaks=[notch.peak_index])[0][0]
                    notches.append(Wave(peak_index=notch.peak_index,
                                        prominence=prom,
                                        width=notch.width,
                                        amp=self.segment_raw[notch.peak_index]))
            return notches
        else:
            peak_index = self.interpret[name]
            if peak_index == -1:
                return None
            segment = self.segment_raw
            if name == 'Q' or name == 'S':
                segment = [-1 * x for x in self.segment_norm]
            prominence = signal.peak_prominences(x=segment, peaks=[peak_index])[0][0]
            if prominence == 0:
                raise AssertionError(f'Interpreted peak {name} = {self.interpret[name]} is not a local minimum/maximum'
                                     f' (prominence = {prominence})')

            width = signal.peak_widths(x=segment, peaks=[peak_index], rel_height=1)[0][0]
            amp = self.segment_raw[peak_index]
            return Wave(peak_index=peak_index, prominence=prominence, width=width, amp=amp)


class Twave:
    def __init__(self, segment_raw: [float], segment_norm: [float], primary_wave: Wave, st_line,
                 biphasic_wave: Wave = None, onset: int = None, energy: float = None):
        self.segment_raw = segment_raw
        self.segment_norm = segment_norm
        self.primary_wave = primary_wave
        self.biphasic_wave = biphasic_wave
        self.st_line = st_line
        self.onset = onset
        self.energy = energy


def cross_baseline(segment: [float], base_amp: float, peak_index: int) -> (bool, int, int):
    left_leg_index = 0
    right_leg_index = len(segment) - 1
    prev_left_amp = segment[peak_index]
    for i in range(peak_index - 1, -1, -1):
        current_amp = segment[i]
        if current_amp > prev_left_amp:
            left_leg_index = i + 1
            break
        prev_left_amp = current_amp

    prev_right_amp = segment[peak_index]
    for i in range(peak_index + 1, len(segment), +1):
        current_amp = segment[i]
        if current_amp > prev_right_amp:
            right_leg_index = i - 1
            break
        prev_right_amp = current_amp

    peak_amp = segment[peak_index]
    left_amp = segment[left_leg_index]
    right_amp = segment[right_leg_index]

    if peak_amp > base_amp > left_amp and right_amp < base_amp:
        return True, left_leg_index, right_leg_index
    return False, left_leg_index, right_leg_index


def get_global_peak(segment: [float]):
    max_point = (list(segment).index(max(segment)), max(segment))
    min_point = (list(segment).index(min(segment)), min(segment))
    global_extremum = max_point[0]
    if abs(min_point[1]) > abs(max_point[1]):
        global_extremum = min_point[0]
    return global_extremum


def identify_j_wave(qrs_segment_norm: [float], qrs_segment_orig: [float], r_index: int, s_index: int, notches: [Wave],
                    base_amp: float):
    # Identifies J-wave, if present, and returns its Wave object. If J-wave is not present, it returns None.
    # J-wave: a positive wave to the right of R-wave that:
    #   1. Must also happen after S-wave (if present).
    #   2. Must be close to R-wave (or S-wave, if present) -> not farther than 1/4 QRS segment.
    #   3. Must have the amplitude less than half the R-wave amp.
    #   4. QRS must end after the J-wave -> its right leg must meet the baseline.
    #   5. In cases where multiple waves meet the above criteria, choose the one closest to R-wave (or S-wave).
    #       5.1. In such cases, ignore all the notches after the J-wave (they are noise).
    if r_index == -1 and s_index == -1:
        raise ValueError('At least one of the R or S peak indexes must be provided for J-wave identification')
    ref_index = r_index
    if s_index != -1:
        ref_index = s_index

    target_notches = [w for w in notches if w.peak_index > ref_index]
    for notch in target_notches:
        if qrs_segment_norm[notch.peak_index] > base_amp:  # First check
            v = 9


def interpret_qrs_peaks(segment_norm: [float], segment_orig: [float], base_amp: float, gmax_index: int):
    result = {'Q': -1, 'R': -1, 'S': -1, 'J': -1, 'notches': []}

    significant_positive_waves = find_positive_waves(segment=segment_norm,
                                                     min_prominence=notch_prom_threshold,
                                                     global_max=gmax_index)

    left_waves = [x for x in significant_positive_waves if x.peak_index < gmax_index]
    right_waves = [x for x in significant_positive_waves if x.peak_index > gmax_index]

    if segment_orig[gmax_index] > 0:
        # R-wave is always the global maximum.
        result['R'] = gmax_index

        if len(left_waves) > 0:
            # All the positive waves to the left of R-wave, no matter if they cross the baseline or not, are categorized
            # as notch. These can then further be post-processed based on their prominence and whether they coss the
            # baseline.
            result['notches'].extend(left_waves)

        if len(right_waves) > 0:
            # All the positive waves to the left of R-wave, no matter if they cross the baseline or not, are categorized
            # as notch. These can then further be post-processed to find a possible J-wave.
            result['notches'].extend(right_waves)
    else:
        if len(left_waves) == 0:
            result['Q'] = gmax_index
        else:
            waves_above_base = [x for x in left_waves if segment_norm[x.peak_index] > base_amp]
            if len(waves_above_base) > 0:
                r_wave = max(waves_above_base, key=lambda x: x.prominence)
                result['R'] = r_wave.peak_index
                result['S'] = gmax_index
                result['notches'].extend([x for x in left_waves if x.peak_index != r_wave.peak_index])
            else:
                result['Q'] = gmax_index
                result['notches'].extend(left_waves)

        if len(right_waves) > 0:
            waves_above_base = [x for x in right_waves if segment_norm[x.peak_index] > base_amp]
            if len(waves_above_base) > 0:
                r_wave = max(waves_above_base, key=lambda x: x.prominence)
                if result['R'] == -1:
                    result['R'] = r_wave.peak_index
                else:
                    result['notches'].append(r_wave)
                result['notches'].extend([x for x in right_waves if x.peak_index != r_wave.peak_index])
            else:
                result['notches'].extend(right_waves)
    return result


def identify_qs(segment_norm: [float], r_index: int, base_amp: float):
    result = {'Q': -1, 'S': -1}
    onset = np.asarray((0, segment_norm[0]))
    offset = np.asarray((len(segment_norm) - 1, segment_norm[-1]))
    r_peak = np.asarray((r_index, segment_norm[r_index]))

    inverted_segment = [-1 * x for x in segment_norm]
    valley_indexes, _ = signal.find_peaks(x=inverted_segment)
    valley_prominences = signal.peak_prominences(x=inverted_segment, peaks=valley_indexes)[0]
    valley_widths = signal.peak_widths(x=inverted_segment, peaks=valley_indexes, rel_height=1)[0]

    all_negative_waves = []
    for i in range(len(valley_indexes)):
        wave = Wave(peak_index=valley_indexes[i], prominence=valley_prominences[i], width=valley_widths[i])
        all_negative_waves.append(wave)

    significant_negative_waves = []
    for wave in all_negative_waves:
        if wave.prominence > wave_prom_threshold:
            significant_negative_waves.append(wave)
    significant_negative_waves = sorted(significant_negative_waves, key=lambda x: x.peak_index)

    left_max_distance = 0
    right_max_distance = 0
    candidate_q_wave = None
    candidate_s_wave = None
    for wave in significant_negative_waves:
        point = np.asarray((wave.peak_index, segment_norm[wave.peak_index]))
        if wave.peak_index < r_index:
            l_dist = np.linalg.norm(np.cross(r_peak - onset, onset - point)) / np.linalg.norm(r_peak - onset)
            if l_dist > left_max_distance:
                left_max_distance = l_dist
                candidate_q_wave = wave
        else:
            r_dist = np.linalg.norm(np.cross(offset - r_peak, r_peak - point)) / np.linalg.norm(offset - r_peak)
            if r_dist > right_max_distance:
                right_max_distance = r_dist
                candidate_s_wave = wave

    if candidate_q_wave is not None and segment_norm[candidate_q_wave.peak_index] < min(base_amp, 0):
        result['Q'] = candidate_q_wave.peak_index

    if candidate_s_wave is not None and segment_norm[candidate_s_wave.peak_index] < min(base_amp, 0):
        result['S'] = candidate_s_wave.peak_index

    # plt.plot(segment_norm)
    # plt.show()

    return result


def identify_qrs_offset(qrs_segment_norm: [float], qrs: QRSComplex) -> int:
    q = qrs.get_wave(name='Q')
    r = qrs.get_wave(name='R')
    s = qrs.get_wave(name='S')
    notches = qrs.get_wave(name='notches')
    last_peak_index = 0
    if q is not None and q.peak_index > last_peak_index:
        last_peak_index = q.peak_index
    if r is not None and r.peak_index > last_peak_index:
        last_peak_index = r.peak_index
    if s is not None and s.peak_index > last_peak_index:
        last_peak_index = s.peak_index
    for notch in notches:
        if notch.peak_index > last_peak_index:
            last_peak_index = notch.peak_index

    qrs_base_amp = statistics.mean([qrs_segment_norm[0], qrs_segment_norm[-1]])
    base_deviation_threshold = abs(qrs_segment_norm[last_peak_index] - qrs_base_amp) / 2

    if qrs_segment_norm[last_peak_index] < qrs_base_amp:
        _, _, last_peak_index = cross_baseline(qrs_segment_norm, qrs_base_amp, last_peak_index)

    qrs_offset = -1
    max_dist = 0

    p_start = np.asarray((last_peak_index, qrs_segment_norm[last_peak_index]))
    p_end = np.asarray((len(qrs_segment_norm) - 1, qrs_segment_norm[-1]))

    for i in range(last_peak_index + 1, len(qrs_segment_norm)):
        point = np.asarray((i, qrs_segment_norm[i]))
        distance = np.linalg.norm(np.cross(p_end - p_start, p_start - point)) / np.linalg.norm(p_end - p_start)
        if distance > max_dist:
            max_dist = distance
            qrs_offset = i

    return qrs_offset


def find_global_extremum(segment_orig: [float]):
    peak_indexes, _ = signal.find_peaks(x=segment_orig)
    peak_prominences = signal.peak_prominences(x=segment_orig, peaks=peak_indexes)[0]
    peak_widths = signal.peak_widths(x=segment_orig, peaks=peak_indexes, rel_height=1)[0]

    segment_inverted = [-1 * x for x in segment_orig]
    valley_indexes, _ = signal.find_peaks(x=segment_inverted)
    valley_prominences = signal.peak_prominences(x=segment_inverted, peaks=valley_indexes)[0]
    valley_widths = signal.peak_widths(x=segment_inverted, peaks=valley_indexes)[0]

    extrema_idx = np.concatenate((peak_indexes, valley_indexes))
    extrema_prominences = np.concatenate((peak_prominences, valley_prominences))
    extrema_widths = np.concatenate((peak_widths, valley_widths))

    extrema = zip(extrema_idx, extrema_prominences, extrema_widths)
    extrema = [ext for ext in extrema if ext[0] > 2]
    if len(extrema) > 0:
        extremum_idx_with_max_prom = max(extrema, key=lambda x: x[1])
        return Wave(peak_index=extremum_idx_with_max_prom[0],
                    prominence=extremum_idx_with_max_prom[1],
                    width=extremum_idx_with_max_prom[2],
                    amp=segment_orig[extremum_idx_with_max_prom[0]])
    return None


def find_t_peak(t_segment_norm: [float], t_segment_orig: [float]):
    t_waves = []  # Single T-wave or biphasic T-wave.

    smooth_w_len = round(len(t_segment_norm) / 10)
    t_segment_norm_smooth = SignalProcessing.smooth(x=t_segment_norm, window_len=smooth_w_len, window='flat')
    t_segment_orig_smooth = SignalProcessing.smooth(x=t_segment_orig, window_len=smooth_w_len, window='flat')

    t_segment_norm = t_segment_norm_smooth[
                     round(smooth_w_len / 2) - 1: len(t_segment_norm_smooth) - round(smooth_w_len / 2)]
    t_segment_orig = t_segment_orig_smooth[
                     round(smooth_w_len / 2) - 1: len(t_segment_orig_smooth) - round(smooth_w_len / 2)]

    t_wave = find_global_extremum(segment_orig=t_segment_orig)
    if t_wave is None:
        raise AssertionError('could not identify global T peak.')
    t_waves.append(t_wave)

    if t_wave.amp > 0:
        inverted_t_segment = [-1 * x for x in t_segment_norm]
    else:
        inverted_t_segment = t_segment_norm

    valley_indexes, _ = signal.find_peaks(x=inverted_t_segment)
    valley_prominences = signal.peak_prominences(x=inverted_t_segment, peaks=valley_indexes)[0]
    valley_widths = signal.peak_widths(x=inverted_t_segment, peaks=valley_indexes, rel_height=1)[0]
    if len(valley_indexes) > 0:
        valleys = zip(valley_indexes, valley_prominences, valley_widths)
        valley_max = max(valleys, key=lambda x: x[1])
        if t_wave.prominence - 0.2 * t_wave.prominence < valley_max[1] < t_wave.prominence + 0.2 * t_wave.prominence:
            t2_wave = Wave(peak_index=valley_max[0], prominence=valley_max[1], width=valley_max[2],
                           amp=t_segment_orig[valley_max[0]])
            t_waves.append(t2_wave)

    left_most_t_peak = t_waves[0].peak_index
    if len(t_waves) > 1 and t_waves[1].peak_index < left_most_t_peak:
        left_most_t_peak = t_waves[1].peak_index

    p_start = np.asarray((0, t_segment_norm[0]))
    p_end = np.asarray((left_most_t_peak, t_segment_norm[left_most_t_peak]))

    max_dist = 0
    t_onset = -1
    for i in range(1, left_most_t_peak):
        point = np.asarray((i, t_segment_norm[i]))
        distance = np.linalg.norm(np.cross(p_end - p_start, p_start - point)) / np.linalg.norm(p_end - p_start)
        if distance > max_dist:
            max_dist = distance
            t_onset = i

    if t_onset == -1:
        plt.plot(t_segment_orig)
        plt.show()
        raise AssertionError('could not identify T onset.')

    t_width = len(t_segment_orig) - t_onset + 1
    t_waves[0].width = t_width

    x = np.array(list(range(t_onset)))
    y = t_segment_orig[x]
    st_line = linregress(x=x, y=y)
    # plt.plot(t_segment_orig)
    # plt.plot(x, st_line.intercept + st_line.slope * x, 'r')
    # plt.show()
    v = 9
    return t_waves, st_line, t_onset


def plot_qrs_waves(qrs: QRSComplex, ax=None, i: int = None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
        i = 0
    ax[i].plot(qrs.segment_norm)

    q = qrs.get_wave(name='Q')
    r = qrs.get_wave(name='R')
    s = qrs.get_wave(name='S')
    notches = qrs.get_wave(name='notches')

    if q is not None:
        ax[i].scatter(x=q.peak_index, y=qrs.segment_norm[q.peak_index], color='y')
        ax[i].annotate(f'Q-{round(q.prominence, 2)}', (q.peak_index, qrs.segment_norm[q.peak_index]))

    if r is not None:
        ax[i].scatter(x=r.peak_index, y=qrs.segment_norm[r.peak_index], color='r')
        ax[i].annotate(f'R-{round(r.prominence, 2)}', (r.peak_index, qrs.segment_norm[r.peak_index]))

    if s is not None:
        ax[i].scatter(x=s.peak_index, y=qrs.segment_norm[s.peak_index], color='g')
        ax[i].annotate(f'S-{round(s.prominence, 2)}', (s.peak_index, qrs.segment_norm[s.peak_index]))

    for notch in notches:
        ax[i].scatter(x=notch.peak_index, y=qrs.segment_norm[notch.peak_index], color='b')
        ax[i].annotate(f'N-{round(notch.prominence, 2)}', (notch.peak_index, qrs.segment_norm[notch.peak_index]))


def get_gt_ann(ann_df: pd.DataFrame, pid: int, lead_name: str):
    tail = f'_Lead_{lead_name}'
    # heads = ['fQRS', 'RSR\'_RSR\'S\'', 'J_wave']
    heads = ['fQRS', 'RSR\'_RSR\'S\'']
    result = []
    for head in heads:
        col_name = head + tail
        result.append(ann_df.loc[ann_df['Record_ID'] == pid][col_name].values[0])

    if sum(result) > 0:
        return 1
    return 0


def normalize(segment: [float]):
    segment_norm = np.array(TimeSeriesScalerMinMax(value_range=(-1, 1)).fit_transform([segment]))
    return np.reshape(segment_norm, (segment_norm.shape[0], segment_norm.shape[1])).ravel()


def find_positive_waves(segment: [float], min_prominence: float = 0.0, global_max: int = None) -> [Wave]:
    result = []
    all_peak_indexes, _ = signal.find_peaks(x=segment)
    all_peak_prominences = signal.peak_prominences(x=segment, peaks=all_peak_indexes)[0]
    all_peak_widths = signal.peak_widths(x=segment, peaks=all_peak_indexes, rel_height=1)[0]
    for i in range(len(all_peak_indexes)):
        if all_peak_prominences[i] > min_prominence:
            if global_max is not None and all_peak_indexes[i] != global_max:
                wave = Wave(peak_index=all_peak_indexes[i], prominence=all_peak_prominences[i],
                            width=all_peak_widths[i])
                result.append(wave)
    return sorted(result, key=lambda x: x.peak_index)


def get_baseline_prominence(qt_segments: list, lead_index: int):
    for qt_segment in qt_segments:
        lead_qt_segment = qt_segment[lead_index]
        lead_qt_segment_norm = normalize(segment=lead_qt_segment)
        t_segment = lead_qt_segment_norm[round(len(lead_qt_segment_norm) / 3):]
        t_segment_waves = find_positive_waves(segment=t_segment)
        # if len(t_segment_waves) > 2:
        #     return round(sorted(t_segment_waves, key=lambda x: x.prominence, reverse=True)[2].prominence, 2)
        # else:
        #     return 0
        t_segment_waves = sorted(t_segment_waves, key=lambda x: x.prominence, reverse=True)[2:]
        t_segment_waves = [x for x in t_segment_waves if x.prominence >= 0.01]
        if len(t_segment_waves) > 2:
            return round(statistics.mean([x.prominence for x in t_segment_waves]), 2)
        elif len(t_segment_waves) > 0:
            return round(max([x.prominence for x in t_segment_waves]), 2)
        else:
            return 0


def identify_qrs_waves(qrs_segment_orig: [float], qrs_segment_norm: [float]):
    gmax_index = get_global_peak(qrs_segment_orig)  # Global extremum index.

    # Step 1: Compute the baseline: mean of first and last endpoints of the QRS segment.
    # TODO: It is physiologically more reasonable to compute the baseline from the P-wave segment: [T_end, P_start].
    qrs_base_amp = statistics.mean([qrs_segment_norm[0], qrs_segment_norm[-1]])

    # Step 2: Identify and interpret all positive waves and significant notches in the QRS segment.
    interpret_result = interpret_qrs_peaks(segment_norm=qrs_segment_norm,
                                           segment_orig=qrs_segment_orig,
                                           base_amp=qrs_base_amp,
                                           gmax_index=gmax_index)

    # R-wave is always identified by the interpret_qrs_peaks function (because it is a positive wave).
    # If interpret_result says R=-1, it means the QRS complex does not have any R-wave.
    # Step 3: Identify Q and S waves, if R-wave has been identified.
    # The `interpret_qrs_peaks` function does not necessarily identify the Q and S waves when the R-wave
    # exists (remember: its job is to identify and interpret `positive` waves).
    if interpret_result['R'] != -1:
        qs_dict = identify_qs(segment_norm=qrs_segment_norm, r_index=interpret_result['R'], base_amp=qrs_base_amp)
        if interpret_result['Q'] == -1:
            interpret_result['Q'] = qs_dict['Q']
        if interpret_result['S'] == -1:
            interpret_result['S'] = qs_dict['S']

    return QRSComplex(segment_raw=qrs_segment_orig, segment_norm=qrs_segment_norm, interpret=interpret_result,
                      base_amp=qrs_base_amp)


def process_website_ecgs():
    gt_ann = pd.read_excel(GlobalPaths.fqrs_annotation)
    extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.website_ecg,
                                   ann_dir_path=GlobalPaths.website_pla_annotation,
                                   metadata_path=GlobalPaths.website_ecg_meta,
                                   verbose=True)
    extracted_segments_dict = extractor.extract_segments()

    pids = list(extracted_segments_dict.keys())

    # for _ in range(10):
    #     ecg = extracted_segments_dict[random.choice(pids)]['ecg_denoised']
    #     ecg_augmented = vcg_augmentation(np.transpose(ecg.values))
    #     plt.figure("Original ECG", figsize=(15, 5))
    #     plt.plot(ecg['V6'].values)
    #     plt.figure("Augmented ECG", figsize=(15, 5))
    #     plt.plot(ecg_augmented['V6'].values)
    #     plt.show()
    #     v = 9

    report_fp = []
    report_fn = []
    report_per_pid = {}
    for lead_index in range(12):  # Identify fQRS and evaluate per lead.
        lead_name = Util.get_lead_name(index=lead_index)
        result_auto = []
        result_gt = []
        for pid in pids:
            ecg_id = extracted_segments_dict[pid]['ecg_id']
            qt_segments = extracted_segments_dict[pid]['segments']
            fqrs_lead = []
            qrs_object_list = []
            base_prominence = get_baseline_prominence(qt_segments=qt_segments, lead_index=lead_index)
            # Each PID in each lead can have several heartbeats; thus, several QT segments.
            for qt_segment in qt_segments:
                lead_qt_segment = qt_segment[lead_index]
                # Assumption: The QRS complex occurs within the first one-third of the QT interval.
                qrs_segment = lead_qt_segment[:round(len(lead_qt_segment) / 3)]
                # Sometimes the QRS segment is incorrectly identified: monotonically increasing or decreasing. Skip.
                if (qrs_segment[0] == min(qrs_segment) and qrs_segment[-1] == max(qrs_segment)) or \
                        qrs_segment[-1] == min(qrs_segment) and qrs_segment[0] == max(qrs_segment):
                    continue

                gmax_index = get_global_peak(qrs_segment)  # Global extremum index.
                # Step 2: Normalize the QT segment (not just the QRS segment) into [-1, 1]. It is needed to maintain a
                # fixed prominence threshold when identifying notches.
                qrs_segment_norm = normalize(segment=lead_qt_segment)[:len(qrs_segment)]
                # Step 3: Compute the baseline: mean of first and last endpoints of the QRS segment.
                # TODO: It is physiologically more reasonable to compute the baseline from the P-wave segment:
                # [T_end, P_start].
                qrs_base_amp = statistics.mean([qrs_segment_norm[0], qrs_segment_norm[-1]])
                # Step 4: Identify and interpret all positive waves and significant notches in the QRS segment.
                interpret_result = interpret_qrs_peaks(segment_norm=qrs_segment_norm,
                                                       segment_orig=qrs_segment,
                                                       base_amp=qrs_base_amp,
                                                       gmax_index=gmax_index)

                # R-wave is always identified by the interpret_qrs_peaks function (because it is a positive wave).
                # If interpret_result says R=-1, it means the QRS complex does not have any R-wave.

                # Step 5: Identify Q and S waves, if R-wave has been identified.
                # The `interpret_qrs_peaks` function does not necessarily identify the Q and S waves when the R-wave
                # exists (remember: its job is to identify and interpret `positive` waves).
                if interpret_result['R'] != -1:
                    qs_dict = identify_qs(segment_norm=qrs_segment_norm, r_index=interpret_result['R'],
                                          base_amp=qrs_base_amp)
                    if interpret_result['Q'] == -1:
                        interpret_result['Q'] = qs_dict['Q']
                    if interpret_result['S'] == -1:
                        interpret_result['S'] = qs_dict['S']

                # Step 6: Get rid of notches in the terminal portion of the QRS complex.
                all_notches = interpret_result['notches']
                terminal_notches = []
                non_terminal_notches = []
                for notch in all_notches:
                    if notch.peak_index < round(0.5 * len(qrs_segment)):
                        non_terminal_notches.append(notch)
                    else:
                        terminal_notches.append(notch)
                interpret_result['notches'] = non_terminal_notches

                try:
                    qrs = QRSComplex(segment_raw=qrs_segment, segment_norm=qrs_segment_norm, interpret=interpret_result,
                                     base_amp=qrs_base_amp)
                    qrs_object_list.append(qrs)
                except AssertionError as err:
                    print(err)
                    continue

                if len(non_terminal_notches) == 0:
                    fqrs_lead.append(0)
                else:
                    fqrs_lead.append(1)

            if sum(fqrs_lead) > 0:
                result_auto.append(1)
                if pid in report_per_pid:
                    report_per_pid[pid].append(lead_name)
                else:
                    report_per_pid[pid] = [lead_name]
            else:
                result_auto.append(0)

            has_fqrs = get_gt_ann(gt_ann, pid=pid, lead_name=lead_name)
            result_gt.append(has_fqrs)

            # Visualize FP/FN detection and save the errors for reporting.
            if (sum(fqrs_lead) > 0 and has_fqrs == 0) or (sum(fqrs_lead) == 0 and has_fqrs == 1):
                # fig, axes = plt.subplots(nrows=1, ncols=len(qrs_object_list), figsize=(10, 4))
                # for j in range(len(qrs_object_list)):
                #     plot_qrs_waves(qrs_object_list[j], ax=axes, i=j)
                # for ax in axes:
                #     ax.set_xticks([])
                #     ax.set_yticks([])
                # error_type = 'FP'
                # if sum(fqrs_lead) == 0 and has_fqrs == 1:
                #     error_type = 'FN'
                # fig.suptitle(f'{error_type} Detection: PID={pid} ECG={ecg_id}-{lead_name}', fontsize=16)
                # fig_name = f'Data/fQRSImages/{error_type}_{ecg_id}_{lead_name}.png'
                # plt.savefig(fig_name)
                # plt.show()
                row = [pid, ecg_id, lead_name]
                if sum(fqrs_lead) > 0 and has_fqrs == 0:
                    report_fp.append(row)
                else:
                    report_fn.append(row)

        print(f'\n--- Lead {lead_name} ---:')
        print(classification_report(result_gt, result_auto, target_names=['Non-fQRS', 'fQRS']))
        v = 0

    cols = ['Record_ID', 'ECG_ID', 'Disagreed Lead']
    df_fp = pd.DataFrame(report_fp, columns=cols)
    df_fn = pd.DataFrame(report_fn, columns=cols)
    report_fp = []
    report_fn = []
    for ecg_id in set(df_fp['ECG_ID'].values):
        target_df = df_fp.loc[df_fp['ECG_ID'] == ecg_id]
        disagreed_leads = list(target_df['Disagreed Lead'].values)
        record_id = target_df['Record_ID'].values[0]
        report_fp.append([record_id, ecg_id, disagreed_leads])

    for ecg_id in set(df_fn['ECG_ID'].values):
        target_df = df_fn.loc[df_fn['ECG_ID'] == ecg_id]
        disagreed_leads = list(target_df['Disagreed Lead'].values)
        record_id = target_df['Record_ID'].values[0]
        report_fn.append([record_id, ecg_id, disagreed_leads])

    final_report_fp = []
    final_report_fn = []
    for row in report_fp:
        row.append(report_per_pid[row[0]])
        final_report_fp.append(row)
    for row in report_fn:
        if row[0] in report_per_pid:
            row.append(report_per_pid[row[0]])
        else:
            row.append([])
        final_report_fn.append(row)
    cols = ['Record_ID', 'ECG_ID', 'Disagreed Leads', 'All fQRS leads identified by method']
    df_fp = pd.DataFrame(final_report_fp, columns=cols)
    df_fn = pd.DataFrame(final_report_fn, columns=cols)
    df_fp.to_csv('Data/fQRS_FP.csv', index=False)
    df_fn.to_csv('Data/fQRS_FN.csv', index=False)
    print('Reports Saved!')


def process_t_waves():
    extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.website_ecg,
                                   ann_dir_path=GlobalPaths.website_pla_annotation,
                                   metadata_path=GlobalPaths.website_ecg_meta,
                                   verbose=True)
    extracted_segments_dict = extractor.extract_segments()
    pids = list(extracted_segments_dict.keys())
    t_segment_ratio = 3.5
    for lead_index in range(12):  # Identify fQRS and evaluate per lead.
        lead_name = Util.get_lead_name(index=lead_index)
        for pid in pids:
            ecg_id = extracted_segments_dict[pid]['ecg_id']
            qt_segments = extracted_segments_dict[pid]['segments']
            # Each PID in each lead can have several heartbeats; thus, several QT segments.
            for qt_segment in qt_segments:
                lead_qt_segment = qt_segment[lead_index]
                # Assumption: The QRS complex occurs within the first one-third of the QT interval.
                t_segment = lead_qt_segment[round(len(lead_qt_segment) / t_segment_ratio):]

                # Step 2: Normalize the QT segment (not just the QRS segment) into [-1, 1]. It is needed to maintain a
                # fixed prominence threshold when identifying notches.
                t_segment_norm = normalize(segment=lead_qt_segment)[round(len(lead_qt_segment) / t_segment_ratio):]
                t_waves, st_line, t_onset = find_t_peak(t_segment_norm=t_segment_norm, t_segment_orig=t_segment)

                lead_qt_segment = normalize(lead_qt_segment)
                plt.plot(lead_qt_segment)
                offset = round(len(lead_qt_segment) / t_segment_ratio)
                x = t_waves[0].peak_index + offset
                plt.scatter(x=x, y=lead_qt_segment[x], color='b')
                if len(t_waves) > 1:
                    x = t_waves[1].peak_index + offset
                    plt.scatter(x=x, y=lead_qt_segment[x], color='y')

                x = np.array(list(range(t_onset)))
                plt.plot(x + offset, st_line.intercept + st_line.slope * x, 'r')
                t_onset = t_onset + offset
                plt.scatter(x=t_onset, y=lead_qt_segment[t_onset], color='g')
                plt.axvline(x=round(len(lead_qt_segment) / t_segment_ratio))
                plt.title(f'{ecg_id}-{lead_name}: '
                          f'Prominence = {round(t_waves[0].prominence, 2)} | '
                          f' Width = {round(t_waves[0].width, 2)} | '
                          f' Slope={round(st_line.slope, 3)} | '
                          f' R={round(st_line.rvalue, 3)}')
                plt.show()
                v = 9


def extract_features(extracted_segments_dict: dict, pid: int, lead_index: int):
    ecg_id = extracted_segments_dict[pid]['ecg_id']
    frequency = extracted_segments_dict[pid]['frequency']
    lead_name = Util.get_lead_name(index=lead_index)
    lead_qt_segments = [x[lead_index, :] for x in extracted_segments_dict[pid]['segments']]

    # Buffer vars for feature extraction from lead.
    lead_qrs_list = []
    lead_t_list = []
    # lead_qrs_segment_norm = []
    # lead_t_segment_norm = []

    # Each lead has several QT segments. Loop over all of them and identify QRS and T-wave in each.
    for qt_segment in lead_qt_segments:
        qrs_segment = qt_segment[:round(len(qt_segment) / 3)]
        t_segment = qt_segment[round(len(qt_segment) / 3):]

        # Normalize segments.
        qt_segment_norm = normalize(segment=qt_segment)
        qrs_segment_norm = qt_segment_norm[:round(len(qt_segment) / 3)]
        t_segment_norm = qt_segment_norm[round(len(qt_segment) / 3):]

        # Sometimes the QRS segment is incorrectly identified: monotonically increasing or decreasing. Skip.
        if (qrs_segment[0] == min(qrs_segment) and qrs_segment[-1] == max(qrs_segment)) or \
                qrs_segment[-1] == min(qrs_segment) and qrs_segment[0] == max(qrs_segment):
            continue

        try:
            qrs = identify_qrs_waves(qrs_segment_orig=qrs_segment, qrs_segment_norm=qrs_segment_norm)
            qrs_offset = identify_qrs_offset(qrs_segment_norm, qrs)
            if qrs_offset < 1:
                continue
            qrs.width = qrs_offset
            qrs.duration = round((1 / frequency) * qrs_offset, 4)
            # Calculate energy of signal: https://matel.p.lodz.pl/wee/i12zet/Signal%20energy%20and%20power.pdf
            qrs.energy = scipy.integrate.simpson(y=[abs(y) ** 2 for y in qrs_segment[:qrs_offset + 1]])
            # plot_qrs_waves(qrs)
            # plt.show()
            v = 9
        except AssertionError:
            continue

        try:
            t_waves, st_line, t_onset = find_t_peak(t_segment_norm=t_segment_norm, t_segment_orig=t_segment)
        except AssertionError:
            continue
        biphasic_wave = None
        if len(t_waves) > 1:
            biphasic_wave = t_waves[1]
        twave = Twave(segment_raw=t_segment, segment_norm=t_segment_norm, primary_wave=t_waves[0], st_line=st_line,
                      biphasic_wave=biphasic_wave, onset=t_onset)
        twave.energy = scipy.integrate.simpson(y=[abs(y) ** 2 for y in qt_segment[qrs_offset + 1:]])
        # x = np.array(list(range(round(len(qt_segment) / 3), round(len(qt_segment) / 3) + 20)))
        # figure(figsize=(3, 6), dpi=80)
        # plt.plot(qt_segment)
        # plt.scatter(x=twave.primary_wave.peak_index + t_offset, y=qt_segment[twave.primary_wave.peak_index + t_offset],
        #             color='r')
        # plt.scatter(x=twave.onset + t_offset, y=qt_segment[twave.onset + t_offset],
        #             color='b')
        # plt.plot(x, st_line.intercept + st_line.slope * x, 'r')
        # plt.title(f'Slope={round(twave.st_line.slope, 3)}')
        # plt.show()
        v = 9

        lead_qrs_list.append(qrs)
        lead_t_list.append(twave)

    lead_feature_vector = {'Q': 0, 'R': 0, 'S': 0, 'QRS_duration': 0,
                           'terminal_notches': 0, 'non_terminal_notches': 0, 'cross_baseline': False,
                           'max_prominence': 0,
                           'T': 0, 'T2': False, 't_prominence': 0, 't_width': 0}

    # Step 1: Extract Q, R, and S amps and QRS prominence:
    # QRS prominence = max prominence among Q, R, and S.
    q_list = []
    r_list = []
    s_list = []
    qrs_prominence = 0
    for qrs in lead_qrs_list:
        q = qrs.get_wave(name='Q')
        r = qrs.get_wave(name='R')
        s = qrs.get_wave(name='S')
        if q is not None:
            q_list.append(q)
            if q.prominence > qrs_prominence:
                qrs_prominence = q.prominence
        if r is not None:
            r_list.append(r)
            if r.prominence > qrs_prominence:
                qrs_prominence = r.prominence
        if s is not None:
            s_list.append(s)
            if s.prominence > qrs_prominence:
                qrs_prominence = s.prominence

    lead_feature_vector['QRS_prominence'] = qrs_prominence
    if len(q_list) > 1:
        lead_feature_vector['Q'] = statistics.mean([x.amp for x in q_list])
    elif len(q_list) == 1:
        lead_feature_vector['Q'] = q_list[0].amp

    if len(r_list) > 1:
        lead_feature_vector['R'] = statistics.mean([x.amp for x in r_list])
    elif len(r_list) == 1:
        lead_feature_vector['R'] = r_list[0].amp

    if len(s_list) > 1:
        lead_feature_vector['S'] = statistics.mean([x.amp for x in s_list])
    elif len(s_list) == 1:
        lead_feature_vector['S'] = s_list[0].amp

    lead_feature_vector['QRS_duration'] = statistics.mean([qrs.width for qrs in lead_qrs_list]) * (1 / frequency)
    lead_feature_vector['QRS_energy'] = statistics.mean([qrs.energy for qrs in lead_qrs_list])

    # Step 2: Handle notches.
    lead_feature_vector['terminal_notches'] = max(
        [len(qrs_notches) for qrs_notches in [qrs.get_wave(name='terminal_notches') for qrs in lead_qrs_list]])
    lead_feature_vector['non_terminal_notches'] = max(
        [len(qrs_notches) for qrs_notches in [qrs.get_wave(name='non_terminal_notches') for qrs in lead_qrs_list]])

    non_terminal_prominence = 0
    for qrs_notches in [qrs.get_wave(name='non_terminal_notches') for qrs in lead_qrs_list]:
        if len(qrs_notches) > 0:
            temp_max_prominence = max([notch.prominence for notch in qrs_notches])
            if temp_max_prominence > non_terminal_prominence:
                non_terminal_prominence = temp_max_prominence
    lead_feature_vector['non_terminal_prominence'] = non_terminal_prominence

    terminal_prominence = 0
    for qrs_notches in [qrs.get_wave(name='terminal_notches') for qrs in lead_qrs_list]:
        if len(qrs_notches) > 0:
            temp_max_prominence = max([notch.prominence for notch in qrs_notches])
            if temp_max_prominence > terminal_prominence:
                terminal_prominence = temp_max_prominence
    lead_feature_vector['terminal_prominence'] = terminal_prominence

    terminal_notch_cross_list = []
    non_terminal_notch_cross_list = []
    for qrs in lead_qrs_list:
        terminal_notches = qrs.get_wave(name='terminal_notches')
        non_terminal_notches = qrs.get_wave(name='non_terminal_notches')
        terminal_has_crossed = 0
        non_terminal_has_crossed = 0
        for notch in terminal_notches:
            temp, _, _ = cross_baseline(segment=qrs.segment_norm, base_amp=qrs.base_amp, peak_index=notch.peak_index)
            if temp:
                terminal_has_crossed = 1
                break
        terminal_notch_cross_list.append(terminal_has_crossed)
        for notch in non_terminal_notches:
            temp, _, _ = cross_baseline(segment=qrs.segment_norm, base_amp=qrs.base_amp, peak_index=notch.peak_index)
            if temp:
                non_terminal_has_crossed = 1
                break
        non_terminal_notch_cross_list.append(non_terminal_has_crossed)

    lead_feature_vector['terminal_has_crossed'] = int(any(terminal_notch_cross_list))
    lead_feature_vector['non_terminal_has_crossed'] = int(any(non_terminal_notch_cross_list))

    # Step 3: Handle T-wave.
    # TODO: A wave's width must be proportional to its sampling frequency. -> Done!
    # TODO: Slope of the line intercepting QRS_end and left leg of T-wave. -> Done!
    lead_feature_vector['T'] = statistics.mean([t.primary_wave.amp for t in lead_t_list])
    lead_feature_vector['t_prominence'] = statistics.mean([t.primary_wave.prominence for t in lead_t_list])
    lead_feature_vector['t_duration'] = statistics.mean([t.primary_wave.width for t in lead_t_list]) * (1 / frequency)
    lead_feature_vector['T2'] = any([True if t.biphasic_wave is not None else False for t in lead_t_list])
    lead_feature_vector['st_slope'] = statistics.mean([t.st_line.slope for t in lead_t_list])
    lead_feature_vector['st_slope_min'] = min([t.st_line.slope for t in lead_t_list])
    lead_feature_vector['st_slope_max'] = max([t.st_line.slope for t in lead_t_list])
    lead_feature_vector['st_rvalue'] = statistics.mean([t.st_line.rvalue for t in lead_t_list])
    lead_feature_vector['t_energy'] = statistics.mean([t.energy for t in lead_t_list])

    # for key in lead_feature_vector:
    #     if isinstance(lead_feature_vector[key], float):
    #         if abs(lead_feature_vector[key]) > 1:
    #             lead_feature_vector[key] = round(lead_feature_vector[key], 1)
    #         else:
    #             lead_feature_vector[key] = round(lead_feature_vector[key], 3)
    #
    # lead_signal = extracted_segments_dict[pid]['ecg_denoised'][lead_name].values
    # plt.figure(figsize=(30, 19))
    # plt.plot(lead_signal)
    # title = f"{pid} - {ecg_id} - {lead_name}\n " \
    #         f"Q={lead_feature_vector['Q']}   R={lead_feature_vector['R']}    S={lead_feature_vector['S']} QRS_dur={lead_feature_vector['QRS_duration']} QRS_energy={lead_feature_vector['QRS_energy']:,}\n" \
    #         f"Non-TerminalNotchCount={lead_feature_vector['non_terminal_notches']}   TerminalNotchCount={lead_feature_vector['terminal_notches']}\n" \
    #         f"QRSProm={lead_feature_vector['QRS_prominence']}    Non-TerminalProm={lead_feature_vector['non_terminal_prominence']}   TerminalProm={lead_feature_vector['terminal_prominence']}\n" \
    #         f"Non-TerminalHasCrossed?={lead_feature_vector['non_terminal_has_crossed']}  TerminalHasCrossed?={lead_feature_vector['terminal_has_crossed']}\n" \
    #         f"T={lead_feature_vector['T']}   T2={lead_feature_vector['T2']}  T_Prom={lead_feature_vector['t_prominence']}    T_dur={lead_feature_vector['t_duration']} T_energy={lead_feature_vector['t_energy']:,}\n" \
    #         f"ST_slope={lead_feature_vector['st_slope']},    Slope_max={lead_feature_vector['st_slope_max']},    Slope_min={lead_feature_vector['st_slope_min']}  ST_rvalue={lead_feature_vector['st_rvalue']}"
    # plt.title(title, fontsize=20)
    # plt.xticks(np.arange(0, len(lead_signal) + 1, 50))
    # plt.gca().xaxis.grid(True)
    # plt.show()
    v = 9
    return lead_feature_vector


def process_website_ecg_for_ml():
    extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.website_ecg,
                                   ann_dir_path=GlobalPaths.website_pla_annotation,
                                   metadata_path=GlobalPaths.website_ecg_meta,
                                   verbose=True)
    extracted_segments_dict = extractor.extract_segments()
    pids = list(extracted_segments_dict.keys())

    # f = extract_features(extracted_segments_dict, pid=10650, lead_index=Util.get_lead_id(lead_name='aVR'))

    for _ in range(20):
        pid = random.choice(pids)
        lead_index = random.choice(range(12))
        print(f'PID = {pid}, Lead = {lead_index}')
        f = extract_features(extracted_segments_dict, pid=pid, lead_index=lead_index)

    # for pid in pids:
    #     for lead_index in range(12):
    #         f = extract_features(extracted_segments_dict, pid=pid, lead_index=lead_index)


# Returns a DataFrame with 'Record_ID', 'Basal', 'Mid', 'Apical', 'Apex'.
def get_scar_location_dataset() -> pd.DataFrame:
    scar_columns = ['Record_ID', 'HCM type', 'Basal A', 'Basal S', 'Basal I', 'Basal L', 'Mid A', 'Mid S', 'Mid I',
                    'Mid L', 'Apical A', 'Apical S', 'Apical I', 'Apical L', 'Apex']
    scar_loc_df = pd.read_excel(GlobalPaths.scar_location)[scar_columns]
    scar_loc_df = scar_loc_df[scar_loc_df[scar_columns[2]].notna()]

    # hypertrophy_df = scar_loc_df[['Record_ID', 'HCM type']]
    basal_df = scar_loc_df[[col for col in scar_loc_df.columns if 'Basal' in col] + ['Record_ID']]
    mid_df = scar_loc_df[[col for col in scar_loc_df.columns if 'Mid' in col] + ['Record_ID']]
    apical_df = scar_loc_df[[col for col in scar_loc_df.columns if 'Apical' in col] + ['Record_ID']]
    apex_df = scar_loc_df[[col for col in scar_loc_df.columns if 'Apex' in col] + ['Record_ID']]

    scar_loc_4_areas = []
    for _, row in scar_loc_df.iterrows():
        pid = row['Record_ID']
        basal = int(any(basal_df.loc[basal_df['Record_ID'] == pid].iloc[:, :-1].values[0]))
        mid = int(any(mid_df.loc[mid_df['Record_ID'] == pid].iloc[:, :-1].values[0]))
        apical = int(any(apical_df.loc[apical_df['Record_ID'] == pid].iloc[:, :-1].values[0]))
        apex = int(any(apex_df.loc[apex_df['Record_ID'] == pid].iloc[:, :-1].values[0]))
        new_row = [pid, basal, mid, apical, apex]
        scar_loc_4_areas.append(new_row)

    return pd.DataFrame(data=scar_loc_4_areas, columns=['Record_ID', 'Basal', 'Mid', 'Apical', 'Apex'])


def get_ecg_feature_dataset(extracted_segments_dict: dict) -> pd.DataFrame:
    pids = list(extracted_segments_dict.keys())

    ecg_feature_ds = []
    ecg_features = ['Q', 'R', 'S', 'QRS_duration', 'QRS_energy',
                    'non_terminal_notches', 'terminal_notches', 'QRS_prominence', 'non_terminal_prominence',
                    'terminal_prominence',
                    'T', 't_prominence', 't_duration', 't_energy',
                    'st_slope', 'st_slope_max', 'st_slope_min', 'st_rvalue',
                    'non_terminal_has_crossed', 'terminal_has_crossed']

    count = 0
    for pid in pids:
        patient_feature_vector = []
        for lead_index in range(12):
            # if pid != 10347 or lead_index != 7:
            #     continue
            lead_feature_vector = []
            feature_dict = extract_features(extracted_segments_dict, pid=pid, lead_index=lead_index)
            for feature in ecg_features:
                lead_feature_vector.append(feature_dict[feature])
            patient_feature_vector.extend(lead_feature_vector)
        patient_feature_vector.insert(0, pid)
        ecg_feature_ds.append(patient_feature_vector)
        count += 1
        if count % 10 == 0:
            print(f'{count}/{len(pids)} subjects processed')
        v = 9

    ecg_columns = ['Record_ID']
    for lead_index in range(12):
        lead_name = Util.get_lead_name(lead_index)
        for feature in ecg_features:
            col_name = '(' + lead_name + ')' + feature
            ecg_columns.append(col_name)
    return pd.DataFrame(data=ecg_feature_ds, columns=ecg_columns)


def get_ecg_scar_dataset(region_name: str, select_top_features: bool = True) -> (pd.DataFrame, [str], [str]):
    try:
        dataset = pd.read_excel('cached_dataset.xlsx')
    except FileNotFoundError:
        scar_location_ds = get_scar_location_dataset()  # 'Record_ID', 'Basal', 'Mid', 'Apical', 'Apex'
        scar_location_ds.dropna(inplace=True)

        extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg,
                                       ann_dir_path=GlobalPaths.pla_annotation,
                                       metadata_path=GlobalPaths.cached_scar_ecg_meta,
                                       verbose=True)
        extracted_segments_dict = extractor.extract_segments()
        ecg_feature_ds = get_ecg_feature_dataset(extracted_segments_dict)

        dataset = pd.merge(left=scar_location_ds, right=ecg_feature_ds, how="inner", on=["Record_ID"])
        dataset.dropna(inplace=True)

        dataset.to_excel('cached_dataset.xlsx', index=False)

    discrete_features = ['non_terminal_notches', 'terminal_notches', 'non_terminal_has_crossed', 'terminal_has_crossed']
    continuous_features = ['Q', 'R', 'S', 'QRS_duration', 'QRS_energy', 'QRS_prominence', 'non_terminal_prominence',
                           'terminal_prominence', 'T', 't_prominence', 't_duration', 't_energy', 'st_slope',
                           'st_slope_max', 'st_slope_min', 'st_rvalue']
    if select_top_features:
        other_regions_set = {'Basal', 'Mid', 'Apical', 'Apex'}.difference({region_name})
        # selected_features = get_top_features(dataset=dataset,
        #                                      discrete_features=discrete_features,
        #                                      continuous_features=continuous_features,
        #                                      region_name=region_name, other_regions_set=other_regions_set)
        selected_features = get_top_features_all_ttest(dataset=dataset, region_name=region_name, other_regions_set=other_regions_set)
        dataset = dataset[[col for col in dataset.columns if col in selected_features] + ['Record_ID', region_name]]
    else:
        dataset = dataset[[col for col in dataset.columns if contain_leads(col, ['II', 'aVF', 'V2', 'V6'])] + ['Record_ID', region_name]]

    continuous_columns = [col for col in dataset.columns if col not in {'Record_ID', region_name} and col.split(')')[1] in continuous_features]
    discrete_columns = [col for col in dataset.columns if col not in {'Record_ID', region_name} and col.split(')')[1] in discrete_features]
    dataset = dataset[continuous_columns + discrete_columns + ['Record_ID', region_name]]
    return dataset, continuous_columns, discrete_columns


def generate_augmented_ecg_ds():
    scar_location_ds = get_scar_location_dataset()  # 'Record_ID', 'Basal', 'Mid', 'Apical', 'Apex'
    scar_pids = set(scar_location_ds['Record_ID'].values)

    extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg,
                                   ann_dir_path=GlobalPaths.pla_annotation,
                                   metadata_path=GlobalPaths.cached_scar_ecg_meta,
                                   verbose=True)
    extracted_segments_dict = extractor.extract_segments()
    ecg_pids = list(extracted_segments_dict.keys())

    ecg_meta_augmented = []
    scar_location_augmented_ds = []
    for ecg_pid in ecg_pids:
        if ecg_pid in scar_pids:
            # Step 1: Generate VCG-augmented ECGs.
            ecg = extracted_segments_dict[ecg_pid]['ecg_denoised']
            ecg_augmented_1 = vcg_augmentation(np.transpose(ecg.values))
            ecg_augmented_2 = vcg_augmentation(np.transpose(ecg.values))
            ecg_augmented_3 = vcg_augmentation(np.transpose(ecg.values))

            # Step 2: Store scar info for the new augmented ECGs.
            new_ecg_pid = ecg_pid * 1000
            basal = scar_location_ds.loc[scar_location_ds['Record_ID'] == ecg_pid]['Basal'].values[0]
            mid = scar_location_ds.loc[scar_location_ds['Record_ID'] == ecg_pid]['Mid'].values[0]
            apical = scar_location_ds.loc[scar_location_ds['Record_ID'] == ecg_pid]['Apical'].values[0]
            apex = scar_location_ds.loc[scar_location_ds['Record_ID'] == ecg_pid]['Apex'].values[0]
            scar_location_augmented_ds.append([new_ecg_pid + 1, basal, mid, apical, apex])
            scar_location_augmented_ds.append([new_ecg_pid + 2, basal, mid, apical, apex])
            scar_location_augmented_ds.append([new_ecg_pid + 3, basal, mid, apical, apex])

            # Step 3: Write augmented ECGs into Data/ECG/ScarECG/Augmented and store ECG metadata.
            new_ecg_id = extracted_segments_dict[ecg_pid]['ecg_id'] * 1000
            frequency = extracted_segments_dict[ecg_pid]['frequency']
            ecg_augmented_1.to_csv(f'Data/ECG/ScarECG/Augmented/{new_ecg_id + 1}.csv', index=False)
            ecg_meta_augmented.append([new_ecg_id + 1, new_ecg_pid + 1, frequency])
            ecg_augmented_2.to_csv(f'Data/ECG/ScarECG/Augmented/{new_ecg_id + 2}.csv', index=False)
            ecg_meta_augmented.append([new_ecg_id + 2, new_ecg_pid + 2, frequency])
            ecg_augmented_3.to_csv(f'Data/ECG/ScarECG/Augmented/{new_ecg_id + 3}.csv', index=False)
            ecg_meta_augmented.append([new_ecg_id + 3, new_ecg_pid + 3, frequency])

    pd.DataFrame(ecg_meta_augmented, columns=['ECG ID', 'Record_ID', 'Sample Base']).to_csv(
        'Data/ECG/ScarECG/Augmented/scar_ecg_augmented_meta.csv', index=False)
    pd.DataFrame(ecg_meta_augmented, columns=['ECG ID', 'Record_ID', 'Sample Base']).to_excel(
        'Data/ECG/ScarECG/Augmented/scar_ecg_augmented_meta.xlsx', index=False)
    pd.DataFrame(scar_location_augmented_ds, columns=['Record_ID', 'Basal', 'Mid', 'Apical', 'Apex']).to_excel(
        'Data/ECG/ScarECG/Augmented/scar_location_augmented.xlsx', index=False)
    print('ECG augmentation done!')


models = {'LogisticRegression': LogisticRegression(solver='lbfgs', multi_class='ovr'),
              'SVM': svm.LinearSVC(),
              'RF': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight=None, criterion='log_loss'),
              'XGB': xgb.XGBClassifier(objective="binary:logistic",
                                       colsample_bytree=1,
                                       gamma=0.25,
                                       learning_rate=0.1,
                                       max_depth=5,
                                       reg_lambda=10,
                                       scale_pos_weight=1,
                                       subsample=0.8),
              'MLP': MLPClassifier(alpha=1e-3, hidden_layer_sizes=(130,))}


def augment_smote(X: np.ndarray, y: np.ndarray, last_continuous_index: int):
    # Augments dataset following SMOTE-NC approach.
    # Assumption 1: All features are normalized.
    # Assumption 2: Continuous features are placed at [0 ... last_continuous_index].

    # Step 0: Check assumptions.
    # for i in range(last_continuous_index+1):
    #     if len(set(X[:, i])) == 2:
    #         raise AssertionError(f'Feature at column {i} is not continuous')

    # print(f'Augmenting dataset: {collections.Counter(y)}')
    categorical_features = list(range(last_continuous_index+1, X.shape[1]))
    sm = SMOTENC(categorical_features=categorical_features)
    # sm = ADASYN()
    X_res, y_res = sm.fit_resample(X, y)
    # print('Done!')
    return X_res, y_res


def predict_scar_smote(region_name: str, select_top_features: bool = False):
    try:
        dataset = pd.read_excel('cached_dataset.xlsx')
    except FileNotFoundError:
        scar_location_ds = get_scar_location_dataset()  # 'Record_ID', 'Basal', 'Mid', 'Apical', 'Apex'
        scar_location_ds.dropna(inplace=True)

        extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg,
                                       ann_dir_path=GlobalPaths.pla_annotation,
                                       metadata_path=GlobalPaths.cached_scar_ecg_meta,
                                       verbose=True)
        extracted_segments_dict = extractor.extract_segments()
        ecg_feature_ds = get_ecg_feature_dataset(extracted_segments_dict)

        dataset = pd.merge(left=scar_location_ds, right=ecg_feature_ds, how="inner", on=["Record_ID"])
        dataset.dropna(inplace=True)

        dataset.to_excel('cached_dataset.xlsx', index=False)

    discrete_features = ['non_terminal_notches', 'terminal_notches', 'non_terminal_has_crossed', 'terminal_has_crossed']
    continuous_features = ['Q', 'R', 'S', 'QRS_duration', 'QRS_energy', 'QRS_prominence', 'non_terminal_prominence',
                           'terminal_prominence', 'T', 't_prominence', 't_duration', 't_energy', 'st_slope',
                           'st_slope_max', 'st_slope_min', 'st_rvalue']
    if select_top_features:
        other_regions_set = {'Basal', 'Mid', 'Apical', 'Apex'}.difference({region_name})
        # selected_features = get_top_features(dataset=dataset,
        #                                      discrete_features=discrete_features,
        #                                      continuous_features=continuous_features,
        #                                      region_name=region_name, other_regions_set=other_regions_set)
        selected_features = get_top_features_all_ttest(dataset=dataset, region_name=region_name, other_regions_set=other_regions_set)
        dataset = dataset[[col for col in dataset.columns if col in selected_features] + ['Record_ID', region_name]]
    else:
        dataset = dataset[[col for col in dataset.columns if contain_leads(col, ['II', 'aVF', 'V2', 'V6'])] + ['Record_ID', region_name]]

    continuous_columns = [col for col in dataset.columns if col not in {'Record_ID', region_name} and col.split(')')[1] in continuous_features]
    discrete_columns = [col for col in dataset.columns if col not in {'Record_ID', region_name} and col.split(')')[1] in discrete_features]
    dataset = dataset[continuous_columns + discrete_columns + ['Record_ID', region_name]]

    # Check if `dataset` is well organized.
    if '_' in dataset.columns.values[-1]:
        raise AssertionError(f'"{dataset.columns.values[-1]}" is not a valid region name')
    if dataset.columns.values[-2] != 'Record_ID':
        raise AssertionError('Column -2 must be "Record_ID"')

    run = 5
    acc_runs, f1_runs, auc_runs = [], [], []
    for model_name in models:
        model = models[model_name]
        print(f'\n--- Model = {model_name} (testing for {run} times) ---')
        for _ in range(run):
            acc_folds, f1_folds, auc_folds = [], [], []
            kf = KFold(n_splits=10, shuffle=True, random_state=123)
            for split in kf.split(dataset):
                train = dataset.iloc[split[0]]
                test = dataset.iloc[split[1]]

                train_x, train_y = train.iloc[:, 0:-2], train.iloc[:, -1].values
                test_x, test_y = test.iloc[:, 0:-2], test.iloc[:, -1].values

                train_x_continuous = train_x[continuous_columns + [col for col in discrete_columns if '_has_' not in col]]
                # train_x_continuous = train_x[continuous_columns]

                train_x_categorical = train_x[[col for col in discrete_columns if '_has_' in col]]
                # train_x_categorical = train_x[discrete_columns]

                test_x_continuous = test_x[continuous_columns + [col for col in discrete_columns if '_has_' not in col]]
                # test_x_continuous = test_x[continuous_columns]

                test_x_categorical = test_x[[col for col in discrete_columns if '_has_' in col]]
                # test_x_categorical = test_x[discrete_columns]

                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                train_x = np.concatenate((scaler.fit_transform(train_x_continuous.values), train_x_categorical.values),
                                         axis=1)
                test_x = np.concatenate((scaler.transform(test_x_continuous.values), test_x_categorical.values),
                                        axis=1)

                # train_x_augmented, train_y_augmented = augment_smote(X=train_x, y=train_y, last_continuous_index=train_x_continuous.shape[1] - 1)
                # train_x = np.concatenate((train_x, train_x_augmented), axis=0)
                # train_y = np.concatenate((train_y, train_y_augmented), axis=0)
                # print(f'Final trainset: {collections.Counter(train_y)}')
                # print(f'Final testset: {collections.Counter(test_y)}')


                model.fit(train_x, train_y)
                preds = model.predict(test_x)

                acc = accuracy_score(test_y, preds)
                f1 = f1_score(test_y, preds)
                auc = roc_auc_score(test_y, preds)
                acc_folds.append(acc)
                f1_folds.append(f1)
                auc_folds.append(auc)
            acc_runs.append(statistics.mean(acc_folds))
            f1_runs.append(statistics.mean(f1_folds))
            auc_runs.append(statistics.mean(auc_folds))
        print(f'Accuracy = {round(statistics.mean(acc_runs) * 100, 2)}% ± {round(statistics.stdev(acc_runs) * 100, 2)}%')
        print(f'F1 = {round(statistics.mean(f1_runs) * 100, 2)}% ± {round(statistics.stdev(f1_runs) * 100, 2)}%')
        print(f'AUC = {round(statistics.mean(auc_runs) * 100, 2)}% ± {round(statistics.stdev(auc_runs) * 100, 2)}%')


def predict_scar_grid_search_rf(region_name: str, select_top_features: bool = False):
    dataset, continuous_columns, discrete_columns = get_ecg_scar_dataset(region_name=region_name, select_top_features=select_top_features)

    # Check if `dataset` is well organized.
    if '_' in dataset.columns.values[-1]:
        raise AssertionError(f'"{dataset.columns.values[-1]}" is not a valid region name')
    if dataset.columns.values[-2] != 'Record_ID':
        raise AssertionError('Column -2 must be "Record_ID"')

    # Phase 1: GridSearch for hyper-tuning RandomForestClassifier.
    param_grid = {
        "n_estimators": [50, 75, 100, 150, 200],
        "criterion": ['gini', 'entropy', 'log_loss'],
        "class_weight": [None, 'balanced', 'balanced_subsample'],
        "max_depth": [None, 3, 5, 7, 10],
        "n_jobs": [-1],
    }
    cl = RandomForestClassifier()
    grid_cv = GridSearchCV(cl, param_grid, n_jobs=-1, scoring='f1', cv=5, verbose=1)
    _ = grid_cv.fit(X=dataset.iloc[:, 0:-2].values, y=dataset.iloc[:, -1].values)
    print(f'GridSearch Best Score = {grid_cv.best_score_}')
    print(f'For Parameters:\n{grid_cv.best_params_}')

    model = grid_cv.best_estimator_

    # model.fit(dataset.iloc[:, 0:-2].values, dataset.iloc[:, -1].values)
    preds = model.predict(dataset.iloc[:, 0:-2].values)
    y_true = dataset.iloc[:, -1].values
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, preds)
    print(acc, f1, auc)

    # model = RandomForestClassifier(n_estimators=75, max_depth=None, class_weight='balanced_subsample', criterion='gini')
    # run = 5
    # acc_runs, f1_runs, auc_runs = [], [], []
    # for _ in range(run):
    #     acc_folds, f1_folds, auc_folds = [], [], []
    #     kf = KFold(n_splits=5, shuffle=True)
    #     for split in kf.split(dataset):
    #         train = dataset.iloc[split[0]]
    #         test = dataset.iloc[split[1]]
    #
    #         train_x, train_y = train.iloc[:, 0:-2].values, train.iloc[:, -1].values
    #         test_x, test_y = test.iloc[:, 0:-2].values, test.iloc[:, -1].values
    #
    #         model.fit(train_x, train_y)
    #         preds = model.predict(test_x)
    #
    #         acc = accuracy_score(test_y, preds)
    #         f1 = f1_score(test_y, preds)
    #         auc = roc_auc_score(test_y, preds)
    #         acc_folds.append(acc)
    #         f1_folds.append(f1)
    #         auc_folds.append(auc)
    #     acc_runs.append(statistics.mean(acc_folds))
    #     f1_runs.append(statistics.mean(f1_folds))
    #     auc_runs.append(statistics.mean(auc_folds))
    # print(f'Accuracy = {round(statistics.mean(acc_runs) * 100, 2)}% ± {round(statistics.stdev(acc_runs) * 100, 2)}%')
    # print(f'F1 = {round(statistics.mean(f1_runs) * 100, 2)}% ± {round(statistics.stdev(f1_runs) * 100, 2)}%')
    # print(f'AUC = {round(statistics.mean(auc_runs) * 100, 2)}% ± {round(statistics.stdev(auc_runs) * 100, 2)}%')


def predict_scar_grid_search_xgb(region_name: str, select_top_features: bool = False):
    dataset, continuous_columns, discrete_columns = get_ecg_scar_dataset(region_name=region_name, select_top_features=select_top_features)
    # Check if `dataset` is well organized.
    if '_' in dataset.columns.values[-1]:
        raise AssertionError(f'"{dataset.columns.values[-1]}" is not a valid region name')
    if dataset.columns.values[-2] != 'Record_ID':
        raise AssertionError('Column -2 must be "Record_ID"')

    param_grid = {
        "max_depth": [3, 4, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "gamma": [0, 0.25, 1, 3],
        "reg_lambda": [0, 1, 10, 30],
        "scale_pos_weight": [1, 3, 5],
        "subsample": [0.5, 0.8, 1],
        "colsample_bytree": [0.3, 0.5, 0.8, 1],
    }
    acc_runs, f1_runs, auc_runs = [], [], []
    for run in range(10):
        print(f'\n--- Run {run+1} ---')
        train_x, test_x, train_y, test_y = train_test_split(dataset.iloc[:, 0:-2].values,
                                                            dataset.iloc[:, -1].values,
                                                            test_size=0.33,
                                                            shuffle=True)

        xgb_cl = xgb.XGBClassifier(objective="binary:logistic")
        grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, scoring='f1', cv=5, verbose=1)
        _ = grid_cv.fit(X=train_x, y=train_y)
        print(f'GridSearch Best Score = {grid_cv.best_score_}')
        print(f'For Parameters:\n{grid_cv.best_params_}')

        model = grid_cv.best_estimator_
        preds = model.predict(test_x)
        acc_runs.append(accuracy_score(test_y, preds))
        f1_runs.append(f1_score(test_y, preds))
        auc_runs.append(roc_auc_score(test_y, preds))
        print(f'Accuracy = {round(statistics.mean(acc_runs) * 100, 2)}% ± {round(statistics.stdev(acc_runs) * 100, 2)}%')
        print(f'F1 = {round(statistics.mean(f1_runs) * 100, 2)}% ± {round(statistics.stdev(f1_runs) * 100, 2)}%')
        print(f'AUC = {round(statistics.mean(auc_runs) * 100, 2)}% ± {round(statistics.stdev(auc_runs) * 100, 2)}%')


    # Phase 1: GridSearch for hyper-tuning XGBoostClassifier.

    # xgb_cl = xgb.XGBClassifier(objective="binary:logistic")
    # grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, scoring='f1', cv=3, verbose=1)
    # _ = grid_cv.fit(X=dataset.iloc[:, 0:-2].values, y=dataset.iloc[:, -1].values)
    # print(f'GridSearch Best Score = {grid_cv.best_score_}')
    # print(f'For Parameters:\n{grid_cv.best_params_}')
    #
    # model = grid_cv.best_estimator_
    # preds = model.predict(dataset.iloc[:, 0:-2].values)
    # y_true = dataset.iloc[:, -1].values





    # run = 5
    # acc_runs, f1_runs, auc_runs = [], [], []
    # for _ in range(run):
    #     acc_folds, f1_folds, auc_folds = [], [], []
    #     kf = KFold(n_splits=5, shuffle=True)
    #     for split in kf.split(dataset):
    #         train = dataset.iloc[split[0]]
    #         test = dataset.iloc[split[1]]
    #
    #         train_x, train_y = train.iloc[:, 0:-2].values, train.iloc[:, -1].values
    #         test_x, test_y = test.iloc[:, 0:-2].values, test.iloc[:, -1].values
    #
    #         model.fit(train_x, train_y)
    #         preds = model.predict(test_x)
    #
    #         acc = accuracy_score(test_y, preds)
    #         f1 = f1_score(test_y, preds)
    #         auc = roc_auc_score(test_y, preds)
    #         acc_folds.append(acc)
    #         f1_folds.append(f1)
    #         auc_folds.append(auc)
    #     acc_runs.append(statistics.mean(acc_folds))
    #     f1_runs.append(statistics.mean(f1_folds))
    #     auc_runs.append(statistics.mean(auc_folds))
    # print(f'Accuracy = {round(statistics.mean(acc_runs) * 100, 2)}% ± {round(statistics.stdev(acc_runs) * 100, 2)}%')
    # print(f'F1 = {round(statistics.mean(f1_runs) * 100, 2)}% ± {round(statistics.stdev(f1_runs) * 100, 2)}%')
    # print(f'AUC = {round(statistics.mean(auc_runs) * 100, 2)}% ± {round(statistics.stdev(auc_runs) * 100, 2)}%')


def get_top_features(dataset: pd.DataFrame, discrete_features: [str], continuous_features: [str], region_name: str, other_regions_set: set):
    df = dataset[[col for col in dataset.columns if col not in other_regions_set]]
    ttest_result = []
    discrete_col_names = []
    for col in df.columns:
        if col not in ['Record_ID', region_name]:
            feature_name = col.split(')')[1]
            if feature_name in discrete_features:
                discrete_col_names.append(col)
            elif feature_name in continuous_features:
                _, p_value = ttest_ind(a=df.loc[df[region_name] == 0][col].values,
                                       b=df.loc[df[region_name] == 1][col].values,
                                       equal_var=False)
                ttest_result.append((col, p_value))
            else:
                raise AssertionError(f'Invalid feature {col}')
    ttest_result = sorted(ttest_result, key=lambda item: item[1])
    ttest_result = [x for x in ttest_result if x[1] < 0.05]

    df_discrete = df[discrete_col_names]
    mi = mutual_info_classif(X=df_discrete.values, y=df[region_name].values, discrete_features=True)
    mi_result = list(zip(discrete_col_names, mi))
    mi_result = sorted(mi_result, key=lambda item: item[1], reverse=True)
    mi_result = [x for x in mi_result if x[1] > 0.001]
    # EHRFeatureSelection.plot_feature_score(ttest_result, y_title='P-value', y_limit=0.05)
    selected_features = [x[0] for x in ttest_result] #+ [x[0] for x in mi_result]
    return selected_features


def feature_multiple_comparison():
    dataset = pd.read_excel('cached_dataset.xlsx')
    result = []
    for feature in dataset.columns:
        if feature not in ['Record_ID', 'Basal', 'Mid', 'Apical', 'Apex']:
            uncorrected_pvalues = []
            for region_name in ['Basal', 'Mid', 'Apical', 'Apex']:
                _, p_value = ttest_ind(a=dataset.loc[dataset[region_name] == 0][feature].values,
                                       b=dataset.loc[dataset[region_name] == 1][feature].values,
                                       equal_var=False)
                uncorrected_pvalues.append(p_value)
            reject, pvals_corrected, _, _ = multipletests(pvals=uncorrected_pvalues, method='bonferroni')
            result.append((feature, pvals_corrected))
    result_sorted = sorted(result, key=lambda x: sum(x[1]))

    result_sorted = result_sorted[:10]

    labels = [x[0] for x in result_sorted]
    basal_values = [x[1][0] for x in result_sorted]
    mid_values = [x[1][1] for x in result_sorted]
    apical_values = [x[1][2] for x in result_sorted]
    apex_values = [x[1][3] for x in result_sorted]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 5))
    rects1 = ax.bar(x - width / 2, basal_values, width, label='Basal')
    rects2 = ax.bar(x + width / 2, mid_values, width, label='Mid')
    rects3 = ax.bar(x + width / 2, apical_values, width, label='Apical')
    rects4 = ax.bar(x + width / 2, apex_values, width, label='Apex')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Corrected P-value')
    ax.set_title('Feature Significance in each Region')
    ax.set_xticks(x, labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    # ax.bar_label(rects3, padding=3)
    # ax.bar_label(rects4, padding=3)

    fig.tight_layout()

    plt.show()
    v = 9




    for region_name in ['a', 'b', 'c', 'd']:
        other_regions_set = set()
        df = dataset[[col for col in dataset.columns if col not in other_regions_set]]


def get_top_features_all_ttest(dataset: pd.DataFrame, region_name: str, other_regions_set: set):
    df = dataset[[col for col in dataset.columns if col not in other_regions_set]]
    ttest_result = []
    for col in df.columns:
        if col not in ['Record_ID', region_name]:
            _, p_value = ttest_ind(a=df.loc[df[region_name] == 0][col].values,
                                   b=df.loc[df[region_name] == 1][col].values,
                                   equal_var=False)
            ttest_result.append((col, p_value))
    ttest_result = sorted(ttest_result, key=lambda item: item[1])
    ttest_result = [x for x in ttest_result if x[1] < 0.05]

    # EHRFeatureSelection.plot_feature_score(ttest_result, y_title='P-value', y_limit=0.05)
    selected_features = [x[0] for x in ttest_result]
    return selected_features


def prepare_scar_ecg_dataset(augment: bool, augment_percent: float, region_name: str, select_top_features: bool = False):
    other_regions_set = {'Basal', 'Mid', 'Apical', 'Apex'}.difference({region_name})
    scar_location_ds = get_scar_location_dataset()  # 'Record_ID', 'Basal', 'Mid', 'Apical', 'Apex'
    scar_location_ds.dropna(inplace=True)

    extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg,
                                   ann_dir_path=GlobalPaths.pla_annotation,
                                   metadata_path=GlobalPaths.cached_scar_ecg_meta,
                                   verbose=True)
    extracted_segments_dict = extractor.extract_segments()
    ecg_feature_ds = get_ecg_feature_dataset(extracted_segments_dict)

    dataset = pd.merge(left=scar_location_ds, right=ecg_feature_ds, how="inner", on=["Record_ID"])

    if select_top_features:
        selected_features = get_top_features(dataset=dataset, region_name=region_name, other_regions_set=other_regions_set)



    if augment:
        scar_location_augmented_ds = pd.read_excel(GlobalPaths.scar_location_augmented)
        extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg_augmented,
                                       ann_dir_path=GlobalPaths.pla_annotation_augmented,
                                       metadata_path=GlobalPaths.scar_ecg_augmented_meta,
                                       verbose=True)
        extracted_segments_dict = extractor.extract_segments()
        ecg_augmented_feature_ds = get_ecg_feature_dataset(extracted_segments_dict)
        dataset_augmented = pd.merge(left=scar_location_augmented_ds, right=ecg_augmented_feature_ds, how="inner", on=["Record_ID"])
        augmented_pids = set(dataset_augmented['Record_ID'].values)

    models = {'LogisticRegression': LogisticRegression(solver='lbfgs', multi_class='ovr'),
              'SVM': svm.LinearSVC(),
              'RF': RandomForestClassifier(n_estimators=100, max_depth=5),
              'XGB': xgb.XGBClassifier(objective="binary:logistic",
                                       colsample_bytree=1,
                                       gamma=0.25,
                                       learning_rate=0.1,
                                       max_depth=5,
                                       reg_lambda=10,
                                       scale_pos_weight=1,
                                       subsample=0.8),
              'MLP': MLPClassifier(alpha=1e-3, hidden_layer_sizes=(130,))}

    for model_name in models:
        model = models[model_name]
        print(f'\n--- Model = {model_name} ---\n')

        acc_list = []
        f1_list = []
        auc_list = []
        kf = KFold(n_splits=10, shuffle=True, random_state=123)
        for split in kf.split(dataset):
            train = dataset.iloc[split[0]]
            test = dataset.iloc[split[1]]
            if augment:
                real_pids = list(set(train['Record_ID'].values))
                selected_pids = random.sample(real_pids, round(len(real_pids) * augment_percent))
                selected_augmented_ds = []
                for pid in selected_pids:
                    pid_augmented = pid * 1000 + random.randint(1, 3)
                    if pid_augmented in augmented_pids:
                        row = dataset_augmented.loc[dataset_augmented['Record_ID'] == pid_augmented].values[0]
                        selected_augmented_ds.append(row)
                selected_augmented_ds = pd.DataFrame(selected_augmented_ds, columns=dataset_augmented.columns)
                train = pd.concat([train, selected_augmented_ds], ignore_index=True)
                train = train.sample(frac=1).reset_index(drop=True)

            if select_top_features:
                train = train[[col for col in train.columns if col in selected_features] + ['Record_ID', region_name]]
                test = test[[col for col in test.columns if col in selected_features] + ['Record_ID', region_name]]
            else:
                train = train[
                    [col for col in train.columns if contain_leads(col, ['II', 'aVF', 'V2', 'V6'])] + ['Record_ID',
                                                                                                       region_name]]
                test = test[
                    [col for col in test.columns if contain_leads(col, ['II', 'aVF', 'V2', 'V6'])] + ['Record_ID',
                                                                                                      region_name]]

            train_x, train_y = train.iloc[:, 0:-2], train.iloc[:, -1].values
            test_x, test_y = test.iloc[:, 0:-2], test.iloc[:, -1].values

            train_x_continuous = train_x[[col for col in train_x.columns if 'has_' not in col]]
            train_x_categorical = train_x[[col for col in train_x.columns if 'has_' in col]]
            test_x_continuous = test_x[[col for col in test_x.columns if 'has_' not in col]]
            test_x_categorical = test_x[[col for col in test_x.columns if 'has_' in col]]

            scaler = preprocessing.StandardScaler()
            train_x = np.concatenate((scaler.fit_transform(train_x_continuous.values), train_x_categorical.values),
                                     axis=1)
            test_x = np.concatenate((scaler.transform(test_x_continuous.values), test_x_categorical.values),
                                    axis=1)

            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            acc = accuracy_score(test_y, preds)
            f1 = f1_score(test_y, preds)
            auc = roc_auc_score(test_y, preds)
            acc_list.append(acc)
            f1_list.append(f1)
            auc_list.append(auc)
        print(f'Accuracy = {round(statistics.mean(acc_list) * 100, 2)}%')
        print(f'F1 = {round(statistics.mean(f1_list) * 100, 2)}%')
        print(f'AUC = {round(statistics.mean(auc_list) * 100, 2)}%')







def process_scar_ecg_for_ml():
    try:
        result = pd.read_csv('Data/ecg_feature_scar.csv')
        # result = pd.read_csv('Data/ecg_feature_hypertrophy.csv')
        # predict_hypertrophy(dataset=result)
        predict_scar(dataset=result)
        # run_nn(result)
    except FileNotFoundError:
        scar_loc_4_areas = get_scar_location_dataset()
        # Prepare ECG feature dataset.
        extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg,
                                       ann_dir_path=GlobalPaths.pla_annotation,
                                       metadata_path=GlobalPaths.cached_scar_ecg_meta,
                                       verbose=True)
        extracted_segments_dict = extractor.extract_segments()
        pids = list(extracted_segments_dict.keys())

        ecg_feature_ds = []
        ecg_features = ['Q', 'R', 'S', 'QRS_duration', 'QRS_energy',
                        'non_terminal_notches', 'terminal_notches', 'QRS_prominence', 'non_terminal_prominence',
                        'terminal_prominence',
                        'T', 't_prominence', 't_duration', 't_energy',
                        'st_slope', 'st_slope_max', 'st_slope_min', 'st_rvalue',
                        'non_terminal_has_crossed', 'terminal_has_crossed']

        count = 0
        for pid in pids:
            patient_feature_vector = []
            for lead_index in range(12):
                # if pid != 10347 or lead_index != 7:
                #     continue
                lead_feature_vector = []
                feature_dict = extract_features(extracted_segments_dict, pid=pid, lead_index=lead_index)
                for feature in ecg_features:
                    lead_feature_vector.append(feature_dict[feature])
                patient_feature_vector.extend(lead_feature_vector)
            patient_feature_vector.insert(0, pid)
            ecg_feature_ds.append(patient_feature_vector)
            count += 1
            if count % 10 == 0:
                print(f'{count}/{len(pids)} subjects processed')
            # if count == 30:
            #     break
            v = 9

        ecg_columns = ['Record_ID']
        for lead_index in range(12):
            lead_name = Util.get_lead_name(lead_index)
            for feature in ecg_features:
                col_name = '_' + lead_name + '_' + feature
                ecg_columns.append(col_name)
        ecg_feature_ds = pd.DataFrame(data=ecg_feature_ds, columns=ecg_columns)

        result_scar = pd.merge(left=scar_loc_4_areas, right=ecg_feature_ds, how="inner", on=["Record_ID"])
        result_scar.to_csv('Data/ecg_feature_scar.csv', index=False)

        # result_hypertrophy = pd.merge(left=hypertrophy_df, right=ecg_feature_ds, how="inner", on=["Record_ID"])
        # result_hypertrophy.to_csv('Data/ecg_feature_hypertrophy.csv', index=False)


def contain_leads(col_name: str, lead_names: [str]):
    for lead_name in lead_names:
        if f'_{lead_name}_' in col_name:
            return True
    return False


def predict_hypertrophy(dataset: pd.DataFrame):
    # Consider only features from leads II, aVF, V2, V6.
    # TODO: Maybe try other leads too. Or, increase/decrease the number of leads.
    dataset = dataset[[col for col in dataset.columns if contain_leads(col, ['II', 'aVF', 'V2', 'V6'])] + ['Record_ID', 'HCM type']]
    dataset.dropna(inplace=True)

    models = {
        # 'LogisticRegression': LogisticRegression(solver='lbfgs', multi_class='ovr'),
        # 'SVM': svm.LinearSVC(),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=7),
        # 'XGB': xgb.XGBClassifier(objective="binary:logistic",
        #                                colsample_bytree=1,
        #                                gamma=0.25,
        #                                learning_rate=0.1,
        #                                max_depth=5,
        #                                reg_lambda=10,
        #                                scale_pos_weight=1,
        #                                subsample=0.8),
        # 'MLP': MLPClassifier(alpha=1e-3, hidden_layer_sizes=(130,))
    }

    for model_name in models:
        model = models[model_name]
        print(f'\n--- Model = {model_name} ---\n')

        acc_list, f1_list, auc_list = [], [], []
        kf = KFold(n_splits=10, shuffle=True, random_state=123)
        for split in kf.split(dataset):
            train = dataset.iloc[split[0]]
            test = dataset.iloc[split[1]]
            train_x, train_y = train.iloc[:, 0:-2], train.iloc[:, -1].values
            test_x, test_y = test.iloc[:, 0:-2], test.iloc[:, -1].values

            train_x_continuous = train_x[[col for col in train_x.columns if 'has_' not in col]]
            train_x_categorical = train_x[[col for col in train_x.columns if 'has_' in col]]
            test_x_continuous = test_x[[col for col in test_x.columns if 'has_' not in col]]
            test_x_categorical = test_x[[col for col in test_x.columns if 'has_' in col]]

            scaler = preprocessing.StandardScaler()
            train_x = np.concatenate((scaler.fit_transform(train_x_continuous.values), train_x_categorical.values), axis=1)
            test_x = np.concatenate((scaler.transform(test_x_continuous.values), test_x_categorical.values), axis=1)

            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            acc = accuracy_score(test_y, preds)
            f1 = f1_score(test_y, preds, average='weighted')
            # auc = roc_auc_score(test_y, preds, multi_class='ovr')
            acc_list.append(acc)
            f1_list.append(f1)
            # auc_list.append(auc)
            # print(confusion_matrix(test_y, preds))
            # print(f'Accuracy = {round(acc * 100, 2)}%')
            # print(f'F1 = {round(f1 * 100, 2)}%')
            # print(f'AUC = {round(auc * 100, 2)}%')
        print(f'Accuracy = {round(statistics.mean(acc_list) * 100, 2)}%')
        print(f'F1 = {round(statistics.mean(f1_list) * 100, 2)}%')
        # print(f'AUC = {round(statistics.mean(auc_list) * 100, 2)}%')


def predict_scar(dataset: pd.DataFrame):
    # Consider only features from leads II, aVF, V2, V6.
    # TODO: Maybe try other leads too. Or, increase/decrease the number of leads.
    dataset = dataset[[col for col in dataset.columns if contain_leads(col, ['II', 'aVF', 'V2', 'V6'])] + ['Record_ID', 'Basal', 'Mid', 'Apical', 'Apex']]
    dataset.dropna(inplace=True)

    # Perform binary classification for each of the basal, mid, apical, and apex areas.
    basal_ds = dataset[[col for col in dataset.columns if col not in ['Mid', 'Apical', 'Apex']]]
    mid_ds = dataset[[col for col in dataset.columns if col not in ['Basal', 'Apical', 'Apex']]]
    apical_ds = dataset[[col for col in dataset.columns if col not in ['Mid', 'Basal', 'Apex']]]
    apex_ds = dataset[[col for col in dataset.columns if col not in ['Mid', 'Apical', 'Basal']]]

    df = mid_ds
    target_col = 'Mid'
    ttest_result = []
    for col in df.columns:
        if col not in ['Record_ID', target_col]:
            _, p_value = ttest_ind(a=df.loc[df[target_col] == 0][col].values,
                                   b=df.loc[df[target_col] == 1][col].values,
                                   equal_var=False)
            ttest_result.append((col, p_value))
    ttest_result = sorted(ttest_result, key=lambda item: item[1])
    ttest_result = [x for x in ttest_result if x[1] < 0.05]
    EHRFeatureSelection.plot_feature_score(ttest_result, y_title='P-value', y_limit=0.05)

    selected_features = [x[0] for x in ttest_result]
    # df = df[[col for col in df.columns if col in selected_features] + ['Record_ID', target_col]]

    # Phase 1: GridSearch for hyper-tuning XGBoostClassifier.
    # param_grid = {
    #     "max_depth": [3, 4, 5, 7],
    #     "learning_rate": [0.01, 0.05, 0.1, 0.3],
    #     "gamma": [0, 0.25, 1, 3],
    #     "reg_lambda": [0, 1, 10, 30],
    #     "scale_pos_weight": [1, 3, 5],
    #     "subsample": [0.5, 0.8],
    #     "colsample_bytree": [0.3, 0.5, 0.8, 1],
    # }
    # xgb_cl = xgb.XGBClassifier(objective="binary:logistic")
    # grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, scoring="roc_auc", cv=3, verbose=1)
    # _ = grid_cv.fit(X=df.iloc[:, 0:-2].values, y=df.iloc[:, -1].values)
    # print(f'GridSearch Best Score = {grid_cv.best_score_}')
    # print(f'For Parameters:\n{grid_cv.best_params_}')

    # Phase 2: Perform classification via XGBoost using the best params obtained in the GridSearch phase.

    models = {'LogisticRegression': LogisticRegression(solver='lbfgs', multi_class='ovr'),
              'SVM': svm.LinearSVC(),
              'RF': RandomForestClassifier(n_estimators=100, max_depth=5),
              'XGB': xgb.XGBClassifier(objective="binary:logistic",
                                colsample_bytree=1,
                                gamma=0.25,
                                learning_rate=0.1,
                                max_depth=5,
                                reg_lambda=10,
                                scale_pos_weight=1,
                                subsample=0.8),
              'MLP': MLPClassifier(alpha=1e-3, hidden_layer_sizes=(130,))}

    for model_name in models:
        model = models[model_name]
        print(f'\n--- Model = {model_name} ---\n')

        acc_list = []
        f1_list = []
        auc_list = []
        kf = KFold(n_splits=10, shuffle=True, random_state=123)
        for split in kf.split(df):
            train = df.iloc[split[0]]
            test = df.iloc[split[1]]
            train_x, train_y = train.iloc[:, 0:-2], train.iloc[:, -1].values
            test_x, test_y = test.iloc[:, 0:-2], test.iloc[:, -1].values

            train_x_continuous = train_x[[col for col in train_x.columns if 'has_' not in col]]
            train_x_categorical = train_x[[col for col in train_x.columns if 'has_' in col]]
            test_x_continuous = test_x[[col for col in test_x.columns if 'has_' not in col]]
            test_x_categorical = test_x[[col for col in test_x.columns if 'has_' in col]]

            scaler = preprocessing.StandardScaler()
            train_x = np.concatenate((scaler.fit_transform(train_x_continuous.values), train_x_categorical.values),
                                     axis=1)
            test_x = np.concatenate((scaler.transform(test_x_continuous.values), test_x_categorical.values),
                                    axis=1)

            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            acc = accuracy_score(test_y, preds)
            f1 = f1_score(test_y, preds)
            auc = roc_auc_score(test_y, preds)
            acc_list.append(acc)
            f1_list.append(f1)
            auc_list.append(auc)
            # print(confusion_matrix(test_y, preds))
            # print(f'Accuracy = {round(acc * 100, 2)}%')
            # print(f'F1 = {round(f1 * 100, 2)}%')
            # print(f'AUC = {round(auc * 100, 2)}%')
        print(f'Accuracy = {round(statistics.mean(acc_list) * 100, 2)}%')
        print(f'F1 = {round(statistics.mean(f1_list) * 100, 2)}%')
        print(f'AUC = {round(statistics.mean(auc_list) * 100, 2)}%')


def run_nn(result: pd.DataFrame):
    # Consider only features from leads II, aVF, V2, V6.
    # TODO: Maybe try other leads too. Or, increase/decrease the number of leads.
    result = result[[col for col in result.columns
                     if '_II_' in col or
                     '_aVF_' in col or
                     '_V2_' in col or
                     '_V6_' in col] +
                    ['Record_ID', 'Basal', 'Mid', 'Apical', 'Apex']]

    # Perform binary classification for each of the basal, mid, apical, and apex areas.
    basal_ds = result[[col for col in result.columns if col not in ['Mid', 'Apical', 'Apex']]]
    mid_ds = result[[col for col in result.columns if col not in ['Basal', 'Apical', 'Apex']]]
    apical_ds = result[[col for col in result.columns if col not in ['Mid', 'Basal', 'Apex']]]
    apex_ds = result[[col for col in result.columns if col not in ['Mid', 'Apical', 'Basal']]]

    df = mid_ds
    acc_list = []
    f1_list = []
    auc_list = []
    kf = KFold(n_splits=10, shuffle=True, random_state=123)
    for split in kf.split(df):
        print(f'\n------')
        train = df.iloc[split[0]]
        test = df.iloc[split[1]]
        train_x, train_y = train.iloc[:, 0:-2].values, train.iloc[:, -1].values
        test_x, test_y = test.iloc[:, 0:-2].values, test.iloc[:, -1].values

        scaler = preprocessing.StandardScaler()
        train_x = np.concatenate((scaler.fit_transform(train_x[:, :-2]), train_x[:, -2:]), axis=1)
        test_x = np.concatenate((scaler.transform(test_x[:, :-2]), test_x[:, -2:]), axis=1)

        model = keras.Sequential()
        model.add(keras.layers.Dense(40, input_shape=(train_x.shape[1],), activation='relu'))
        model.add(keras.layers.Dense(20, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=[keras.metrics.Accuracy(), keras.metrics.AUC()])

        model.fit(train_x, train_y, epochs=50, batch_size=1)
        preds1 = model.predict(test_x)
        preds = (model.predict(test_x) > 0.5).astype(int)
        acc = accuracy_score(test_y, preds)
        f1 = f1_score(test_y, preds)
        auc = roc_auc_score(test_y, preds)
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
        # print(f'Accuracy = {round(acc * 100, 2)}%')
        # print(f'F1 = {round(f1 * 100, 2)}%')
        # print(f'AUC = {round(auc * 100, 2)}%')
    print(f'Accuracy = {round(statistics.mean(acc_list) * 100, 2)}%')
    print(f'F1 = {round(statistics.mean(f1_list) * 100, 2)}%')
    print(f'AUC = {round(statistics.mean(auc_list) * 100, 2)}%')


def vcg_augmentation(ecg: np.ndarray):
    dowers_matrix_inverse = [[0.156, -0.227, 0.022],
                     [-0.010, 0.887, 0.102],
                     [-0.172, 0.057, -0.229],
                     [-0.074, -0.019, -0.310],
                     [0.122, -0.106, -0.246],
                     [0.231, -0.022, -0.063],
                     [0.0239, 0.041, 0.055],
                     [0.194, 0.048, 0.108]]

    dowers_matrix = np.linalg.pinv(dowers_matrix_inverse)

    r_x = radians(random.randint(-45, 45))
    rotate_x = [[1, 0, 0],
                [0, cos(r_x), -sin(r_x)],
                [0, sin(r_x), cos(r_x)]]
    r_y = radians(random.randint(-45, 45))
    rotate_y = [[cos(r_y), 0, sin(r_y)],
                [0, 1, 0],
                [-sin(r_y), 0, cos(r_y)]]
    r_z = radians(random.randint(-45, 45))
    rotate_z = [[cos(r_z), -sin(r_z), 0],
                [sin(r_z), cos(r_z), 0],
                [0, 0, 1]]

    if random.uniform(0, 1) > 0.5:
        s_x = random.uniform(1, 1.5)
    else:
        s_x = random.uniform(1/1.5, 1)
    if random.uniform(0, 1) > 0.5:
        s_y = random.uniform(1, 1.5)
    else:
        s_y = random.uniform(1/1.5, 1)
    if random.uniform(0, 1) > 0.5:
        s_z = random.uniform(1, 1.5)
    else:
        s_z = random.uniform(1/1.5, 1)

    scale = [[s_x, 0, 0],
             [0, s_y, 0],
             [0, 0, s_z]]

    independent_leads = [0, 1, 6, 7, 8, 9, 10, 11]
    ind_ecg = ecg[independent_leads, :]

    a = np.matmul(np.transpose(dowers_matrix_inverse), ind_ecg)
    b = np.matmul(rotate_x, a)
    c = np.matmul(rotate_y, b)
    d = np.matmul(rotate_z, c)
    e = np.matmul(scale, d)
    f = np.matmul(np.transpose(dowers_matrix), e)

    lead_iii = f[0, :] - f[1, :]
    lead_avl = (f[0, :] - lead_iii) / 2
    lead_avr = -1 * (f[0, :] + f[1, :]) / 2
    lead_avf = (f[1, :] + lead_iii) / 2

    f1 = np.insert(f, 2, [lead_iii, lead_avr, lead_avl, lead_avf], axis=0)

    return pd.DataFrame(np.transpose(f1), columns=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])


if __name__ == '__main__':
    # process_website_ecgs()
    # process_t_waves()
    # process_website_ecg_for_ml()
    # process_scar_ecg_for_ml()
    # generate_augmented_ecg_ds()
    # prepare_scar_ecg_dataset(augment=False, augment_percent=1, region_name='Mid', select_top_features=False)
    # predict_scar_smote(region_name='Mid', select_top_features=True)
    # predict_scar_grid_search_rf(region_name='Mid', select_top_features=True)
    # predict_scar_grid_search_xgb(region_name='Mid', select_top_features=True)
    feature_multiple_comparison()
    # TODO -> Run train_test_split with shuffle to leave-out a test set, then run gridSearchCV with kfold = 5 on the
    #  train set to get the best_estimator, and then evaluate/predict the best_estimator using test set. Run the whole
    #  thing 10 times.
