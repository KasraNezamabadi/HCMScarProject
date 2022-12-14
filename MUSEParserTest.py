import os
import struct
import base64
import xmltodict
import matplotlib.pyplot as plt
import pandas as pd
import GlobalPaths

if __name__ == '__main__':
    xml_names = [f for f in os.listdir(GlobalPaths.muse) if not f.startswith('.')]
    for xml_name in xml_names:
        muse_dict = xmltodict.parse(open(os.path.join(GlobalPaths.muse, xml_name), 'rb').read().decode('utf8'))[
            'RestingECG']
        ecg_pid = int(muse_dict['PatientDemographics']['PatientID'])
        if ecg_pid != 10003:
            continue
        muse_date_str = muse_dict['TestDemographics']['AcquisitionDate'] + ' ' + muse_dict['TestDemographics']['AcquisitionTime']
        ecg_date = pd.to_datetime(muse_date_str)
        frequency = int(muse_dict['Waveform'][1]['SampleBase'])

        text = muse_dict['Waveform'][1]['LeadData'][2]['WaveFormData']
        gain = float(muse_dict['Waveform'][1]['LeadData'][2]['LeadAmplitudeUnitsPerBit'])
        text = text.strip()
        signal_byte = base64.b64decode(text)
        lead_signal = []
        for t in range(0, len(signal_byte), 2):
            sample = round(struct.unpack('h', bytes([signal_byte[t], signal_byte[t + 1]]))[0] * gain)
            lead_signal.append(sample)
        plt.figure(f'PID=10003 Date={ecg_date}', figsize=(15, 5))
        plt.plot(lead_signal)
        plt.title(f'PID=10003 Date={ecg_date}')
        plt.show()
        v = 9

