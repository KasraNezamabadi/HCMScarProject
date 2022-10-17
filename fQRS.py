import statistics

import numpy as np
from scipy import signal
from random import shuffle
from random import choice
from tslearn.preprocessing import TimeSeriesScalerMinMax
import matplotlib.pyplot as plt
from QTSegmentExtractor import QTSegmentExtractor


extractor = QTSegmentExtractor(verbose=True)
extracted_segments_dict = extractor.extract_segments()

pids = sorted(list(extracted_segments_dict.keys()))
shuffle(pids)


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


def get_global_peak(segment: [float], amp_baseline: float):
    max_point = (list(segment).index(max(segment)), max(segment))
    min_point = (list(segment).index(min(segment)), min(segment))
    max_baseline_divergence = max_point[1] - amp_baseline
    global_extremum = max_point[0]
    if abs(min_point[1]) > abs(max_point[1]):
        segment = [x * -1 for x in segment]
        max_baseline_divergence = amp_baseline - min_point[1]
        global_extremum = min_point[0]
    return global_extremum


def interpret_qrs_peaks(segment_norm: [float], segment_orig: [float], base_amp: float, gmax_index: int):
    result = {'Q': -1, 'R': -1, 'S': -1, 'J':-1, 'notches': []}
    all_peak_indexes, _ = signal.find_peaks(x=segment_norm)
    all_peak_prominences = signal.peak_prominences(x=segment_norm, peaks=all_peak_indexes)[0]
    all_peak_widths = signal.peak_widths(x=segment_norm, peaks=all_peak_indexes)[0]

    all_peaks = zip(all_peak_indexes, all_peak_prominences, all_peak_widths)
    significant_peaks = []
    for peak in all_peaks:
        if peak[2] > 0.1 and peak[0] != gmax_index:
            significant_peaks.append(peak)
    significant_peaks = sorted(significant_peaks, key=lambda x: x[0])
    left_peaks = [x for x in significant_peaks if x[0] < gmax_index]
    right_peaks = [x for x in significant_peaks if x[0] > gmax_index]

    if segment_orig[gmax_index] > 0:
        result['R'] = gmax_index

        if len(left_peaks) > 0:
            result['notches'].extend(left_peaks)

        if len(right_peaks) > 0:
            cross_peaks = [x for x in right_peaks if cross_baseline(segment_norm, base_amp, x[0])]
            result['notches'].extend(cross_peaks)
            no_cross_peaks = [x for x in right_peaks if not cross_baseline(segment_norm, base_amp, x[0])]
            if len(no_cross_peaks) > 0:
                J_index = max(no_cross_peaks, key=lambda item: item[1])[0]
                result['J'] = J_index
                result['notches'].extend([x for x in no_cross_peaks if x[0] != J_index])
    else:
        if len(left_peaks) == 0:
            result['Q'] = gmax_index
        else:
            cross_peaks = [x for x in left_peaks if cross_baseline(segment_norm, base_amp, x[0])]
            no_cross_peaks = [x for x in left_peaks if not cross_baseline(segment_norm, base_amp, x[0])]
            if len(cross_peaks) == 0:
                result['Q'] = gmax_index
                result['notches'].extend(no_cross_peaks)
            else:
                r_index = max(cross_peaks, key=lambda item: item[1])[0]
                result['R'] = r_index
                result['S'] = gmax_index
                result['notches'].extend([x for x in cross_peaks if x[0] != r_index])
                result['notches'].extend(no_cross_peaks)

        if len(right_peaks) > 0:
            cross_peaks = [x for x in right_peaks if cross_baseline(segment_norm, base_amp, x[0])]
            no_cross_peaks = [x for x in right_peaks if not cross_baseline(segment_norm, base_amp, x[0])]
            if len(cross_peaks) == 0:
                result['notches'].extend(no_cross_peaks)
            else:
                r_wave = max(cross_peaks, key=lambda item: item[1])
                if result['R'] == -1:
                    result['R'] = r_wave[0]
                else:
                    result['notches'].append(r_wave)
                result['notches'].extend([x for x in cross_peaks if x[0] != r_wave[0]])
                result['notches'].extend(no_cross_peaks)








def get_peaks(lead_segment: [float]):
    qrs_search_segment = lead_segment[:round(len(lead_segment) / 3)]
    gmax = get_global_peak(qrs_search_segment, statistics.mean([qrs_search_segment[0], qrs_search_segment[-1]]))

    qrs_search_segment = np.array(TimeSeriesScalerMinMax(value_range=(-1, 1)).fit_transform([qrs_search_segment]))
    qrs_search_segment = np.reshape(qrs_search_segment, (qrs_search_segment.shape[0], qrs_search_segment.shape[1]))
    qrs_search_segment = qrs_search_segment.ravel()

    qrs_base_amp = statistics.mean([qrs_search_segment[0], qrs_search_segment[-1]])

    peak_ind, _ = signal.find_peaks(x=qrs_search_segment)
    peak_prom = signal.peak_prominences(x=qrs_search_segment, peaks=peak_ind)[0]
    peak_widths = signal.peak_widths(x=qrs_search_segment, peaks=peak_ind)[0]

    peaks = zip(peak_ind, qrs_search_segment[peak_ind], peak_prom, peak_widths)
    qrs_peaks = []
    for peak in peaks:
        if peak[2] > 0.1 and peak[0] != gmax:
            qrs_peaks.append(peak)
    disp_peak_ind = [x[0] for x in qrs_peaks]
    notch_cross = []
    notch_not_cross = []
    for peak in disp_peak_ind:
        if cross_baseline(qrs_search_segment, qrs_base_amp, peak):
            notch_cross.append(peak)
        else:
            notch_not_cross.append(peak)

    plt.plot(qrs_search_segment)
    plt.axhline(y=qrs_base_amp, color='r', linestyle='-')
    plt.scatter(x=gmax, y=qrs_search_segment[gmax], color='r')

    plt.scatter(x=notch_cross, y=qrs_search_segment[notch_cross], color='b')
    plt.scatter(x=notch_not_cross, y=qrs_search_segment[notch_not_cross], color='g')
    plt.show()
    v = 9


for pid in pids:
    qt_segments = extracted_segments_dict[pid]['segments']
    segment = choice(qt_segments)
    lead = choice([0, 1, 2, 3])
    lead_segment = segment[lead]
    get_peaks(lead_segment)
    v = 9


