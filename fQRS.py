import statistics

import numpy as np
from scipy import signal
from random import shuffle
from random import choice
from tslearn.preprocessing import TimeSeriesScalerMinMax
import matplotlib.pyplot as plt
from QTSegmentExtractor import QTSegmentExtractor

notch_prom_threshold = 0.05
wave_prom_threshold = 0.1
class Wave:
    def __init__(self, peak_index: int, prominence: float, width: float, amp: float = None):
        self.peak_index = peak_index
        self.prominence = prominence
        self.width = width
        self.amp = amp


class QRSComplex:
    def __init__(self, segment_raw: [float], segment_norm: [float], interpret: dict):
        self.segment_raw = segment_raw
        self.segment_norm = segment_norm
        self.interpret = interpret

    def get_wave(self, name: str):
        if name == 'notches':
            notches = []
            for notch in self.interpret[name]:
                notches.append(Wave(peak_index=notch.peak_index, prominence=notch.prominence, width=notch.width, amp=self.segment_raw[notch.peak_index]))
            return notches
        else:
            peak_index = self.interpret[name]
            if peak_index == -1:
                return None
            segment = self.segment_norm
            if name == 'Q' or name == 'S':
                segment = [-1 * x for x in self.segment_norm]
            prominence = signal.peak_prominences(x=segment, peaks=[peak_index])[0][0]
            if prominence == 0:
                assert 'Interpreted peak for is not a local minimum/maximum'
            width = signal.peak_widths(x=segment, peaks=[peak_index])[0][0]
            amp = self.segment_raw[peak_index]
            return Wave(peak_index=peak_index, prominence=prominence, width=width, amp=amp)


def cross_baseline(segment: [float], base_amp: float, peak_index: int) -> bool:
    left_leg_index = 0
    right_leg_index = len(segment) - 1
    prev_left_amp = segment[peak_index]
    for i in range(peak_index-1, -1, -1):
        current_amp = segment[i]
        if current_amp > prev_left_amp:
            left_leg_index = i + 1
            break
        prev_left_amp = current_amp

    prev_right_amp = segment[peak_index]
    for i in range(peak_index+1, len(segment), +1):
        current_amp = segment[i]
        if current_amp > prev_right_amp:
            right_leg_index = i - 1
            break
        prev_right_amp = current_amp

    peak_amp = segment[peak_index]
    left_amp = segment[left_leg_index]
    right_amp = segment[right_leg_index]

    if peak_amp > base_amp and left_amp < base_amp and right_amp < base_amp:
        return True
    return False


def get_global_peak(segment: [float]):
    max_point = (list(segment).index(max(segment)), max(segment))
    min_point = (list(segment).index(min(segment)), min(segment))
    global_extremum = max_point[0]
    if abs(min_point[1]) > abs(max_point[1]):
        global_extremum = min_point[0]
    return global_extremum


def interpret_qrs_peaks(segment_norm: [float], segment_orig: [float], base_amp: float, gmax_index: int):
    result = {'Q': -1, 'R': -1, 'S': -1, 'J': -1, 'notches': []}
    all_peak_indexes, _ = signal.find_peaks(x=segment_norm)
    all_peak_prominences = signal.peak_prominences(x=segment_norm, peaks=all_peak_indexes)[0]
    all_peak_widths = signal.peak_widths(x=segment_norm, peaks=all_peak_indexes)[0]

    all_positive_waves = []
    for i in range(len(all_peak_indexes)):
        wave = Wave(peak_index=all_peak_indexes[i], prominence=all_peak_prominences[i], width=all_peak_widths[i])
        all_positive_waves.append(wave)

    significant_positive_waves = []
    for wave in all_positive_waves:
        if wave.prominence > notch_prom_threshold and wave.peak_index != gmax_index:
            significant_positive_waves.append(wave)
    significant_positive_waves = sorted(significant_positive_waves, key=lambda x: x.peak_index)
    left_waves = [x for x in significant_positive_waves if x.peak_index < gmax_index]
    right_waves = [x for x in significant_positive_waves if x.peak_index > gmax_index]

    if segment_orig[gmax_index] > 0:
        result['R'] = gmax_index

        if len(left_waves) > 0:
            result['notches'].extend(left_waves)

        if len(right_waves) > 0:
            cross_waves = [x for x in right_waves if cross_baseline(segment_norm, base_amp, x.peak_index)]
            result['notches'].extend(cross_waves)
            no_cross_waves = [x for x in right_waves if not cross_baseline(segment_norm, base_amp, x.peak_index)]
            if len(no_cross_waves) > 0:
                candidate_j_wave = max(no_cross_waves, key=lambda x: x.prominence)
                if segment_orig[candidate_j_wave.peak_index] < 0.5 * segment_orig[result['R']] and candidate_j_wave.peak_index < result['R'] + round(0.25 * len(segment_norm)):
                    result['J'] = candidate_j_wave.peak_index
                    result['notches'].extend([x for x in no_cross_waves if x.peak_index != candidate_j_wave.peak_index])
                else:
                    result['notches'].extend(no_cross_waves)
            # plt.plot(segment_norm)
            # plt.show()
            v = 9
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
    valley_widths = signal.peak_widths(x=inverted_segment, peaks=valley_indexes)[0]

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

    if candidate_q_wave is not None and segment_norm[candidate_q_wave.peak_index] < base_amp:
        result['Q'] = candidate_q_wave.peak_index

    if candidate_s_wave is not None and segment_norm[candidate_s_wave.peak_index] < base_amp:
        result['S'] = candidate_s_wave.peak_index

    # plt.plot(segment_norm)
    # plt.show()

    return result


def debug(pid: int, seg_index: int, lead: int):
    lead_segment = extracted_segments_dict[pid]['segments'][seg_index][lead]
    qrs_segment = lead_segment[:round(len(lead_segment) / 3)]
    gmax_index = get_global_peak(qrs_segment)

    qrs_segment_norm = np.array(TimeSeriesScalerMinMax(value_range=(-1, 1)).fit_transform([qrs_segment]))
    qrs_segment_norm = np.reshape(qrs_segment_norm, (qrs_segment_norm.shape[0], qrs_segment_norm.shape[1])).ravel()

    qrs_base_amp = statistics.mean([qrs_segment_norm[0], qrs_segment_norm[-1]])

    result = interpret_qrs_peaks(segment_norm=qrs_segment_norm, segment_orig=qrs_segment,
                                 base_amp=qrs_base_amp, gmax_index=gmax_index)

    if result['R'] != -1:
        qs_dict = identify_qs(segment_norm=qrs_segment_norm, r_index=result['R'], base_amp=qrs_base_amp)
        if result['Q'] == -1:
            result['Q'] = qs_dict['Q']

        if result['S'] == -1:
            result['S'] = qs_dict['S']

    qrs = QRSComplex(segment_raw=qrs_segment, segment_norm=qrs_segment_norm, interpret=result)
    plot_qrs_waves(qrs)
    v = 9


def plot_qrs_waves(qrs: QRSComplex):
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(qrs.segment_norm)
    # plt.axhline(y=qrs_base_amp, color='r', linestyle='-')
    q = qrs.get_wave(name='Q')
    r = qrs.get_wave(name='R')
    s = qrs.get_wave(name='S')
    j = qrs.get_wave(name='J')
    notches = qrs.get_wave(name='notches')

    if q is not None:
        ax.scatter(x=q.peak_index, y=qrs.segment_norm[q.peak_index], color='y')
        ax.annotate('Q', (q.peak_index, qrs.segment_norm[q.peak_index]))

    if r is not None:
        ax.scatter(x=r.peak_index, y=qrs.segment_norm[r.peak_index], color='r')
        ax.annotate('R', (r.peak_index, qrs.segment_norm[r.peak_index]))

    if s is not None:
        ax.scatter(x=s.peak_index, y=qrs.segment_norm[s.peak_index], color='g')
        ax.annotate('S', (s.peak_index, qrs.segment_norm[s.peak_index]))

    if j is not None:
        ax.scatter(x=j.peak_index, y=qrs.segment_norm[j.peak_index], color='m')
        ax.annotate('J', (j.peak_index, qrs.segment_norm[j.peak_index]))

    for notch in notches:
        ax.scatter(x=notch.peak_index, y=qrs.segment_norm[notch.peak_index], color='b')
        ax.annotate('notch', (notch.peak_index, qrs.segment_norm[notch.peak_index]))

    plt.show()


if __name__ == '__main__':
    extractor = QTSegmentExtractor(verbose=True)
    extracted_segments_dict = extractor.extract_segments()
    pids = list(extracted_segments_dict.keys())
    shuffle(pids)

    is_debug = False
    if is_debug:
        debug(pid=10702, seg_index=9, lead=3)
    else:
        for pid in pids:
            qt_segments = extracted_segments_dict[pid]['segments']
            seg_index = choice(range(0, len(qt_segments)))
            segment = qt_segments[seg_index]
            lead = choice([0, 1, 2, 3])
            lead_segment = segment[lead]

            qrs_segment = lead_segment[:round(len(lead_segment) / 3)]
            gmax_index = get_global_peak(qrs_segment)

            qrs_segment_norm = np.array(TimeSeriesScalerMinMax(value_range=(-1, 1)).fit_transform([qrs_segment]))
            qrs_segment_norm = np.reshape(qrs_segment_norm, (qrs_segment_norm.shape[0], qrs_segment_norm.shape[1])).ravel()

            qrs_base_amp = statistics.mean([qrs_segment_norm[0], qrs_segment_norm[-1]])

            result = interpret_qrs_peaks(segment_norm=qrs_segment_norm, segment_orig=qrs_segment,
                                         base_amp=qrs_base_amp, gmax_index=gmax_index)

            if result['R'] != -1:
                qs_dict = identify_qs(segment_norm=qrs_segment_norm, r_index=result['R'], base_amp=qrs_base_amp)
                if result['Q'] == -1:
                    result['Q'] = qs_dict['Q']

                if result['S'] == -1:
                    result['S'] = qs_dict['S']

            qrs = QRSComplex(segment_raw=qrs_segment, segment_norm=qrs_segment_norm, interpret=result)
            plot_qrs_waves(qrs)
            v = 9

#CHECK: [10668, 4, 1], [10311, 9, 0]





