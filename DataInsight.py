import pandas as pd

import DataManagement as DM
from Classification import DatasetManager
import matplotlib.pyplot as plt
import numpy as np
import os
import GlobalPaths

if __name__ == '__main__':
    ecg_names = [f for f in os.listdir(GlobalPaths.ecg) if not f.startswith('.')
                 and 'meta' not in f
                 and f.endswith('.csv')]
    for ecg_name in ecg_names:
        path = os.path.join(GlobalPaths.ecg, ecg_name)
        ecg = pd.read_csv(path, header=None, skiprows=1, names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                                                                'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
        pid = int(ecg_name.split('.')[0])
        fig, ax = plt.subplots(4, figsize=(12, 4))
        fig.suptitle(f'PID = {pid}', fontsize=15)
        ax[0].plot(ecg['II'].values)
        ax[1].plot(ecg['III'].values)
        ax[2].plot(ecg['aVF'].values)
        ax[3].plot(ecg['V2'].values)
        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.savefig(f'Data/ECG/ScarECG/Plot/{pid}.png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.clf()
        plt.close()



    # parser = DM.EHRECGParser()
    #
    # n_folds = 5
    # ds_manager = DatasetManager(ehr_df=parser.ehr_df_imputed,
    #                             ecg_ds=parser.qt_dataset,
    #                             n_folds=n_folds,
    #                             feature_select_ehr=True,
    #                             development_mode=True)
    #
    # x_train_ecg, x_train_ehr, y_train, \
    # x_test_ecg, x_test_ehr, y_test, \
    # ehr_train, ehr_test, ecg_train, ecg_test = ds_manager.get_fold_development(fold_index=0,
    #                                                                            augment_mode='ehr')
    #
    # for i in range(len(ecg_train)):
    #     ecg_object = ecg_train[i]
    #     ehr_object = ehr_train.iloc[i]
    #     class_label = y_train[i]
    #     # First check -> Class label must be equal the one indicated in MRI data frame
    #     record_id = ecg_object['pid']
    #     mri_row = parser.mri_df.loc[parser.mri_df['Record_ID'] == record_id]
    #     if not mri_row.empty and float(mri_row['DE']) != class_label:
    #         assert f'Class Label Mismatch for PID {record_id}'
    #     qt_segment_list = ecg_object['preprocessed']
    #     fig, ax = plt.subplots(nrows=4, ncols=len(qt_segment_list), figsize=(12, 4))
    #     fig.suptitle(f'QT segments for {record_id} DE={class_label}', fontsize=15)
    #     for qt_index in range(len(qt_segment_list)):
    #         qt_segment = qt_segment_list[qt_index]
    #         qt_segment = np.array(qt_segment)
    #         if qt_segment.shape[0] != 4:
    #             qt_segment = np.transpose(qt_segment)
    #         for j in range(4):
    #             lead_qt_segment = qt_segment[j, :]
    #             ax[j][qt_index].plot(lead_qt_segment)
    #             ax[j][qt_index].set_xticks([])
    #             ax[j][qt_index].set_yticks([])
    #     plt.tight_layout()
    #     print(f'Saving pic for PID={record_id}')
    #
    #     plt.savefig(f'DataInsight/{record_id}.png', bbox_inches='tight', dpi=300)
    #     plt.show()
    #     plt.clf()
    #     plt.close()



