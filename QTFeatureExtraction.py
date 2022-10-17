from dataclasses import dataclass
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt

import DataManagement as DM
from Utility import SignalProcessing


@dataclass
class PeakColor:
    qrs_extremum = np.array([255, 0, 0])/255.0
    t_peak = np.array([0, 0, 255])/255.0


# color_table = [[97, 12, 99], [84, 186, 185], [236, 129, 179],
#                [135, 128, 94], [176, 155, 113], [216, 204, 163],
#                [129, 9, 85], [247, 236, 9]])/255.0


def is_baseline(point_x: int, segment: [float]):
    amp_baseline = statistics.mean([segment[0], segment[1], segment[-2], segment[-1]])
    amp_point = segment[point_x]
    if abs(amp_point) > abs(amp_baseline) + 0.2 * abs(amp_baseline):
        return False
    return True


def get_all_peaks(segment: [float]):
    result_positive = SignalProcessing.get_peaks(segment)
    inverted_segment = [x * -1 for x in segment]
    result_negative = SignalProcessing.get_peaks(inverted_segment)
    result = []
    for i in range(len(result_positive)):
        result.append(result_positive[i] + result_negative[i])

    return result


def find_qrs_peak(segment: [float]):
    peaks_x = get_all_peaks(segment)
    candidates_x = peaks_x[0] + peaks_x[1]
    candidates_x = [point_x for point_x in candidates_x
                    if point_x < (1 / 3) * len(segment)
                    and not is_baseline(point_x, segment)]
    qrs_peak_x = 0
    qrs_peak_y = 0
    for candidate_x in candidates_x:
        if abs(segment[candidate_x]) > qrs_peak_y:
            qrs_peak_y = abs(segment[candidate_x])
            qrs_peak_x = candidate_x

    return qrs_peak_x


def find_t_peak(qrs_peak_x: int, segment: [float]):
    peaks_x = get_all_peaks(segment)
    t_peak_x = 0
    t_peak_y = 0
    for candidates_x in peaks_x:
        candidates_x = [point_x for point_x in candidates_x
                        if point_x > qrs_peak_x + 25
                        and not is_baseline(point_x, segment)]
        for candidate_x in candidates_x:
            if abs(segment[candidate_x]) > t_peak_y:
                t_peak_y = abs(segment[candidate_x])
                t_peak_x = candidate_x

    if t_peak_x == 0:
        fig = plt.figure()
        plt.plot(segment)
        plt.scatter(x=qrs_peak_x, y=segment[qrs_peak_x], color=PeakColor.qrs_extremum)
        plt.show()
        v = 0
    return t_peak_x


if __name__ == '__main__':
    parser = DM.EHRECGParser()
    qt_dataset = parser.qt_dataset
    pid = 10679

    if pid is None:
        for exp in range(10):
            qt_object = qt_dataset[random.randint(0, len(qt_dataset)-1)]
            qt_segments = qt_object['preprocessed']
            qt_segment = qt_segments[random.randint(0, len(qt_segments)-1)]
            fig, ax = plt.subplots(4, figsize=(3, 10))
            de = qt_object['de']
            pid = qt_object['pid']
            fig.suptitle(f'DE = {de}, PID={pid}')
            for lead in range(4):
                lead_qt_segment = qt_segment[lead, :]

                ax[lead].plot(lead_qt_segment)
                qrs_peak_x = find_qrs_peak(lead_qt_segment)
                ax[lead].scatter(x=qrs_peak_x, y=lead_qt_segment[qrs_peak_x], color=PeakColor.qrs_extremum)
                t_peak_x = find_t_peak(qrs_peak_x=qrs_peak_x, segment=lead_qt_segment)
                ax[lead].scatter(x=t_peak_x, y=lead_qt_segment[t_peak_x], color=PeakColor.t_peak)
            plt.show()
    else:
        qt_object = None
        for temp in qt_dataset:
            if temp['pid'] == pid:
                qt_object = temp
                qt_segments = qt_object['preprocessed']
                qt_segment = qt_segments[random.randint(0, len(qt_segments) - 1)]
                fig, ax = plt.subplots(4, figsize=(3, 10))
                de = qt_object['de']
                pid = qt_object['pid']
                fig.suptitle(f'DE = {de}, PID={pid}')
                for lead in range(4):
                    lead_qt_segment = qt_segment[lead, :]

                    ax[lead].plot(lead_qt_segment)
                    qrs_peak_x = find_qrs_peak(lead_qt_segment)
                    ax[lead].scatter(x=qrs_peak_x, y=lead_qt_segment[qrs_peak_x], color=PeakColor.qrs_extremum)
                    t_peak_x = find_t_peak(qrs_peak_x=qrs_peak_x, segment=lead_qt_segment)
                    ax[lead].scatter(x=t_peak_x, y=lead_qt_segment[t_peak_x], color=PeakColor.t_peak)
                plt.show()
            else:
                continue

