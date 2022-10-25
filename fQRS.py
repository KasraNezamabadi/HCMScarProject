import statistics

import numpy as np
import pandas as pd
from scipy import signal
from random import shuffle
from random import choice
from tslearn.preprocessing import TimeSeriesScalerMinMax
import matplotlib.pyplot as plt
from QTSegmentExtractor import QTSegmentExtractor
import GlobalPaths
from Utility import Util
from sklearn.metrics import classification_report

notch_prom_threshold = 0.1
wave_prom_threshold = 0.1


class Wave:
    def __init__(self, peak_index: int, prominence: float, width: float, amp: float = None):
        self.peak_index = peak_index
        self.prominence = prominence
        self.width = width
        self.amp = amp


class QRSComplex:
    def __init__(self, segment_raw: [float], segment_norm: [float], interpret: dict, base_amp: float):
        self.segment_raw = segment_raw
        self.segment_norm = segment_norm
        self.interpret = interpret
        self.base_amp = base_amp

        for wave_name in self.interpret:
            self.get_wave(name=wave_name)

    def get_wave(self, name: str):  # Returns None if the wave does not exist.
        if name == 'notches':
            notches = []
            for notch in self.interpret[name]:
                notches.append(Wave(peak_index=notch.peak_index,
                                    prominence=notch.prominence,
                                    width=notch.width,
                                    amp=self.segment_raw[notch.peak_index]))
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
                raise AssertionError(f'Interpreted peak {name} = {self.interpret[name]} is not a local minimum/maximum'
                                     f' (prominence = {prominence})')

            width = signal.peak_widths(x=segment, peaks=[peak_index], rel_height=1)[0][0]
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


def identify_j_wave(qrs_segment_norm: [float], qrs_segment_orig: [float], r_index: int, s_index: int, notches: [Wave], base_amp: float):
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
        if qrs_segment_norm[notch.peak_index] > base_amp: # First check
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

    if candidate_q_wave is not None and segment_norm[candidate_q_wave.peak_index] < base_amp:
        result['Q'] = candidate_q_wave.peak_index

    if candidate_s_wave is not None and segment_norm[candidate_s_wave.peak_index] < base_amp:
        result['S'] = candidate_s_wave.peak_index

    # plt.plot(segment_norm)
    # plt.show()

    return result


def get_wave(signal_norm: [float], peak_index: int):
    prominence = signal.peak_prominences(x=signal_norm, peaks=[peak_index])[0][0]
    if prominence == 0:
        plt.plot(signal_norm)
        plt.scatter(x=peak_index, y=signal_norm[peak_index], color='r')
        plt.show()
        raise AssertionError(f'Requested peak at index {peak_index} is not a local minimum/maximum'
                             f' (prominence = {prominence})')
    width = signal.peak_widths(x=signal_norm, peaks=[peak_index], rel_height=1)[0][0]
    return Wave(peak_index=peak_index, prominence=prominence, width=width)


def find_t_peak(t_segment_norm: [float], t_segment_orig: [float]):
    t_waves = []  # Single T-wave or biphasic T-wave.
    extrema = []
    peak_indexes, _ = signal.find_peaks(x=t_segment_orig)
    valley_indexes, _ = signal.find_peaks(x=[-1 * x for x in t_segment_orig])
    extrema.extend(peak_indexes)
    extrema.extend(valley_indexes)
    extrema_abs_amps = [abs(t_segment_orig[x]) for x in extrema]
    gmax_index = max(zip(extrema, extrema_abs_amps), key=lambda x: x[1])[0]
    # gmax_index = get_global_peak(t_segment_orig)

    if t_segment_orig[gmax_index] < 0:  # T-wave is negative.
        t_segment_norm = [-1 * x for x in t_segment_norm]

    t_wave = get_wave(signal_norm=t_segment_norm, peak_index=gmax_index)
    t_wave.amp = t_segment_orig[t_wave.peak_index]
    t_waves.append(t_wave)

    inverted_t_segment = [-1 * x for x in t_segment_norm]
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
    return t_waves


def plot_qrs_waves(qrs: QRSComplex, ax, i: int):
    ax[i].plot(qrs.segment_norm)

    q = qrs.get_wave(name='Q')
    r = qrs.get_wave(name='R')
    s = qrs.get_wave(name='S')
    j = qrs.get_wave(name='J')
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

    if j is not None:
        ax[i].scatter(x=j.peak_index, y=qrs.segment_norm[j.peak_index], color='m')
        ax[i].annotate(f'J-{round(j.prominence, 2)}', (j.peak_index, qrs.segment_norm[j.peak_index]))

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
                wave = Wave(peak_index=all_peak_indexes[i], prominence=all_peak_prominences[i], width=all_peak_widths[i])
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

    # Step 4: Get rid of notches in the terminal portion of the QRS complex.
    all_notches = interpret_result['notches']
    terminal_notches = []
    non_terminal_notches = []
    for notch in all_notches:
        if notch.peak_index < round(0.5 * len(qrs_segment_orig)):
            non_terminal_notches.append(notch)
        else:
            terminal_notches.append(notch)
    qrs = QRSComplex(segment_raw=qrs_segment_orig, segment_norm=qrs_segment_norm, interpret=interpret_result, base_amp=qrs_base_amp)
    return qrs, non_terminal_notches, terminal_notches


def process_website_ecgs():
    gt_ann = pd.read_excel(GlobalPaths.fqrs_annotation)
    extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.website_ecg,
                                   ann_dir_path=GlobalPaths.website_pla_annotation,
                                   metadata_path=GlobalPaths.website_ecg_meta,
                                   verbose=True)
    extracted_segments_dict = extractor.extract_segments()
    pids = list(extracted_segments_dict.keys())

    report_fp = []
    report_fn = []
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
                    qs_dict = identify_qs(segment_norm=qrs_segment_norm, r_index=interpret_result['R'], base_amp=qrs_base_amp)
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
                    qrs = QRSComplex(segment_raw=qrs_segment, segment_norm=qrs_segment_norm, interpret=interpret_result, base_amp=qrs_base_amp)
                    qrs_object_list.append(qrs)
                except AssertionError as err:
                    print(err)
                    continue

                if len(non_terminal_notches) == 0:
                    fqrs_lead.append(0)
                else:
                    fqrs_lead.append(1)

            if ecg_id == 712 and lead_name == 'I':
                v = 9

            if sum(fqrs_lead) > 0:
                result_auto.append(1)
            else:
                result_auto.append(0)

            has_fqrs = get_gt_ann(gt_ann, pid=pid, lead_name=lead_name)
            result_gt.append(has_fqrs)

            # Visualize FP/FN detection.
            if (sum(fqrs_lead) > 0 and has_fqrs == 0) or (sum(fqrs_lead) == 0 and has_fqrs == 1):
                fig, axes = plt.subplots(nrows=1, ncols=len(qrs_object_list), figsize=(10, 4))
                for j in range(len(qrs_object_list)):
                    plot_qrs_waves(qrs_object_list[j], ax=axes, i=j)
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                error_type = 'FP'
                if sum(fqrs_lead) == 0 and has_fqrs == 1:
                    error_type = 'FN'
                fig.suptitle(f'{error_type} Detection: PID={pid} ECG={ecg_id}-{lead_name}', fontsize=16)
                fig_name = f'Data/fQRSImages/{error_type}_{ecg_id}_{lead_name}.png'
                plt.savefig(fig_name)
                plt.show()
                row = [pid, ecg_id, lead_name]
                if sum(fqrs_lead) > 0 and has_fqrs == 0:
                    report_fp.append(row)
                else:
                    report_fn.append(row)

        print(f'\n--- Lead {lead_name} ---:')
        print(classification_report(result_gt, result_auto, target_names=['Non-fQRS', 'fQRS']))
        v = 0
    cols = ['Record_ID', 'ECG_ID', 'Lead']
    df_fp = pd.DataFrame(report_fp, columns=cols)
    df_fn = pd.DataFrame(report_fn, columns=cols)
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

    for lead_index in range(12):  # Identify fQRS and evaluate per lead.
        lead_name = Util.get_lead_name(index=lead_index)
        for pid in pids:
            ecg_id = extracted_segments_dict[pid]['ecg_id']
            qt_segments = extracted_segments_dict[pid]['segments']
            # Each PID in each lead can have several heartbeats; thus, several QT segments.
            for qt_segment in qt_segments:
                lead_qt_segment = qt_segment[lead_index]
                # Assumption: The QRS complex occurs within the first one-third of the QT interval.
                t_segment = lead_qt_segment[round(len(lead_qt_segment) / 3):]

                # Step 2: Normalize the QT segment (not just the QRS segment) into [-1, 1]. It is needed to maintain a
                # fixed prominence threshold when identifying notches.
                t_segment_norm = normalize(segment=lead_qt_segment)[round(len(lead_qt_segment) / 3):]
                t_waves = find_t_peak(t_segment_norm=t_segment_norm, t_segment_orig=t_segment)
                plt.plot(normalize(segment=lead_qt_segment))
                offset = round(len(lead_qt_segment) / 3)
                x = t_waves[0].peak_index + offset
                plt.scatter(x=x, y=normalize(segment=lead_qt_segment)[x], color='b')
                if len(t_waves) > 1:
                    x = t_waves[1].peak_index + offset
                    plt.scatter(x=x, y=normalize(segment=lead_qt_segment)[x], color='y')
                plt.title(f'{ecg_id}-{lead_name}: Prominence = {round(t_waves[0].prominence, 2)}, Width = {round(t_waves[0].width, 2)}')
                plt.show()
                v = 9


def process_for_ml():
    extractor = QTSegmentExtractor(ecg_dir_path=GlobalPaths.website_ecg,
                                   ann_dir_path=GlobalPaths.website_pla_annotation,
                                   metadata_path=GlobalPaths.website_ecg_meta,
                                   verbose=True)
    extracted_segments_dict = extractor.extract_segments()
    pids = list(extracted_segments_dict.keys())
    for pid in pids:
        ecg_id = extracted_segments_dict[pid]['ecg_id']
        for lead_index in range(12):
            lead_name = Util.get_lead_name(index=lead_index)
            lead_qt_segments = [x[lead_index, :] for x in extracted_segments_dict[pid]['segments']]
            # Buffer vars for feature extraction from lead.
            lead_qrs_list = []
            lead_t_list = []
            lead_qrs_segment_norm = []
            lead_t_segment_norm = []
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
                    qrs, _, _ = identify_qrs_waves(qrs_segment_orig=qrs_segment,
                                                   qrs_segment_norm=qrs_segment_norm)
                except AssertionError:
                    continue

                t_waves = find_t_peak(t_segment_norm=t_segment_norm, t_segment_orig=t_segment)
                lead_qrs_list.append(qrs)
                lead_t_list.append(t_waves)
                lead_t_segment_norm.append(t_segment_norm)
                lead_qrs_segment_norm.append(qrs_segment_norm)

            lead_feature_vector = {'Q': 0, 'R': 0, 'S': 0,
                                   'max_#_notches': 0, 'cross_baseline': False, 'max_prominence': 0,
                                   'T': 0, 'T2': False, 't_prominence': 0, 't_width': 0}

            # Step 1: Extract Q, R, and S amps.
            q_list = []
            r_list = []
            s_list = []
            for qrs in lead_qrs_list:
                q = qrs.get_wave(name='Q')
                r = qrs.get_wave(name='R')
                s = qrs.get_wave(name='S')
                if q is not None:
                    q_list.append(q)
                if r is not None:
                    r_list.append(r)
                if s is not None:
                    s_list.append(s)
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

            # Step 2: Handle notches.
            notch_list = [qrs.get_wave(name='notches') for qrs in lead_qrs_list]
            lead_feature_vector['max_#_notches'] = max([len(notch_set) for notch_set in notch_list])
            if lead_feature_vector['max_#_notches'] > 0:
                lead_feature_vector['max_prominence'] = max([max([notch.prominence for notch in notch_set if len(notch_set) > 0]) for notch_set in notch_list if len(notch_set) > 0])

            notch_cross_list = []
            for i in range(len(lead_qrs_list)):
                qrs = lead_qrs_list[i]
                qrs_segment = lead_qrs_segment_norm[i]
                qrs_base_amp = qrs.base_amp
                notches = qrs.get_wave(name='notches')
                has_crossed = False
                for notch in notches:
                    if cross_baseline(segment=qrs_segment, base_amp=qrs_base_amp, peak_index=notch.peak_index):
                        has_crossed = True
                        break
                notch_cross_list.append(has_crossed)
            lead_feature_vector['cross_baseline'] = any(notch_cross_list)

            # Step 3: Handle T-wave.
            # TODO: A wave's width must be proportional to its sampling frequency.
            lead_feature_vector['T1'] = statistics.mean([t_waves[0].amp for t_waves in lead_t_list])
            lead_feature_vector['t_prominence'] = statistics.mean([t_waves[0].prominence for t_waves in lead_t_list])
            lead_feature_vector['t_width'] = statistics.mean([t_waves[0].width for t_waves in lead_t_list])
            lead_feature_vector['T2'] = any([True if len(t_waves) > 1 else False for t_waves in lead_t_list])

            for key in lead_feature_vector:
                if isinstance(lead_feature_vector[key], float):
                    lead_feature_vector[key] = round(lead_feature_vector[key], 2)

            lead_signal = extracted_segments_dict[pid]['ecg_denoised'][lead_name].values
            plt.figure(figsize=(15, 5))
            plt.plot(lead_signal)
            plt.title(f'{pid}-{ecg_id}-{lead_name}\n{lead_feature_vector}', fontsize=10)
            plt.show()
            v = 9


if __name__ == '__main__':
    # process_website_ecgs()
    # process_t_waves()
    process_for_ml()









