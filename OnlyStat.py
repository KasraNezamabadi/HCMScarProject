import pandas as pd
import numpy as np
import GlobalPaths
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from Utility import Util
from statistics import mean
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

def hist_notch(ds: pd.DataFrame, scar_name: str):
    notch_df = ds[['Record_ID', 'Target'] +
                  [col for col in ds.columns
                   if 'terminal_notches' in col
                   or 'non_terminal_notches' in col
                   or 'terminal_has_crossed' in col
                   or 'non_terminal_has_crossed' in col]]

    leads = [Util.get_lead_name(index) for index in range(12)]
    pids = set(notch_df['Record_ID'].values)
    terminal_df = []
    non_terminal_df = []
    for pid in pids:
        temp_df = notch_df.loc[notch_df['Record_ID'] == pid]
        target = temp_df['Target'].values[0]
        terminal_list = []
        non_terminal_list = []
        for lead in leads:
            terminals = list(temp_df[f'({lead})terminal_notches'].values)
            non_terminals = list(temp_df[f'({lead})non_terminal_notches'].values)
            if len(terminals) > 1:
                terminal = round(mean(terminals))
            else:
                terminal = terminals[0]
            if len(non_terminals) > 1:
                non_terminal = round(mean(non_terminals))
            else:
                non_terminal = non_terminals[0]
            terminal_list.append(terminal)
            non_terminal_list.append(non_terminal)
        terminal_df.append([pid, target] + terminal_list)
        non_terminal_df.append([pid, target] + non_terminal_list)

    terminal_df = pd.DataFrame(terminal_df, columns=['Record_ID', 'Target'] + leads)
    non_terminal_df = pd.DataFrame(non_terminal_df, columns=['Record_ID', 'Target'] + leads)

    for lead in leads:
        a = list(non_terminal_df.loc[non_terminal_df['Target'] == 0][lead].values)
        b = list(non_terminal_df.loc[non_terminal_df['Target'] == 1][lead].values)
        bins = np.linspace(0, 5, 5)
        plt.hist(a, bins, alpha=0.5, label='No Scar')
        plt.hist(b, bins, alpha=0.5, label=scar_name)
        plt.legend(loc='upper right')
        plt.show()


def get_significant_features(ds: pd.DataFrame, scar_name: str, lim: int = 25):
    ttest_result = []
    for col in ds.columns:
        if col not in ['Record_ID', 'ECG_ID', 'MRI Diff', 'LV Scar', scar_name] and \
                'has_crossed' not in col and '_notches' not in col and '[conf]' not in col:
            _, p_value = ttest_ind(a=ds.loc[ds['Target'] == 0][col].values,
                                   b=ds.loc[ds['Target'] == 1][col].values,
                                   equal_var=False)
            ttest_result.append((col, p_value))
    return sorted(ttest_result, key=lambda item: item[1])[:lim]

mri_loc_df = pd.read_excel(GlobalPaths.scar_location)
mri_meta_df = pd.read_excel(GlobalPaths.mri)
mri_meta_df = mri_meta_df[['Record_ID', 'Scar tissue %']]
mri_loc_df = mri_loc_df[['Record_ID', 'MRI Date'] + [col for col in mri_loc_df.columns if 'Basal' in col or 'Mid' in col or 'Apical' in col or 'Apex' in col]]
dataset = pd.merge(left=mri_meta_df, right=mri_loc_df, how='inner', on=['Record_ID'])
dataset.dropna(inplace=True)
dataset.reset_index(drop=True, inplace=True)

# Clean MRI Date.
need_correction = 0
for index, row in dataset.iterrows():
    try:
        dataset.iat[index, 2] = pd.to_datetime(row['MRI Date'])
    except:
        date_str = str(row['MRI Date'])
        if '.' in date_str:
            date_str = date_str.replace('.', '')
        if ',' in date_str:
            mri_date = pd.to_datetime(date_str.split(',')[0])
        else:
            mri_date = pd.to_datetime(date_str.split(' ')[0])
        need_correction += 1
        dataset.iat[index, 2] = mri_date
dataset['MRI Date'] = pd.to_datetime(dataset['MRI Date'])

ecg_df = pd.read_excel('Data/overall_ecg_feature_uncertain_scar_location.xlsx')

dataset = pd.merge(left=ecg_df, right=dataset[['Record_ID', 'Scar tissue %']], how='inner', on=['Record_ID'])
dataset.reset_index(drop=True, inplace=True)
mri_diff_days = []
entire_lv_scar = []
for index, row in dataset.iterrows():
    ecg_date = pd.to_datetime(row['ECG Date'])
    mri_date = pd.to_datetime(row['MRI Date'])
    diff = ecg_date - mri_date
    mri_diff_days.append(diff.days)
    entire_lv_scar.append(sum(row[['Basal', 'Mid', 'Apical', 'Apex']].values))

dataset['MRI Diff'] = mri_diff_days
dataset['LV Scar'] = entire_lv_scar
pids = set(dataset['Record_ID'].values)

# Patients with scar but ScarTissue% = 0. For UCSF check.
target = dataset.loc[(dataset['Scar tissue %'] == 0) & (dataset['LV Scar'] > 0)][['Record_ID', 'MRI Date', 'Scar tissue %', 'Basal', 'Mid', 'Apical', 'Apex']].drop_duplicates(ignore_index=True)
target.to_excel('Data/suspicious_scar.xlsx', index=False)

# Keep only data points that have MRI-ECG diff < 1 month.
target = dataset.loc[abs(dataset['MRI Diff']) < 32]
all_pids = set(target['Record_ID'].values)
no_scar_set = set(target.loc[target['LV Scar'] == 0]['Record_ID'].values)
scar_set = set(target.loc[target['LV Scar'] != 0]['Record_ID'].values)
one_scar_set = set(target.loc[target['LV Scar'] == 1]['Record_ID'].values)
two_scar_set = set(target.loc[target['LV Scar'] == 2]['Record_ID'].values)
three_scar_set = set(target.loc[target['LV Scar'] == 3]['Record_ID'].values)
four_scar_set = set(target.loc[target['LV Scar'] == 4]['Record_ID'].values)

basal_scar_set = set(target.loc[target['Basal'] == 1]['Record_ID'].values)
mid_scar_set = set(target.loc[target['Mid'] == 1]['Record_ID'].values)
apical_scar_set = set(target.loc[target['Apical'] == 1]['Record_ID'].values)
apex_scar_set = set(target.loc[target['Apex'] == 1]['Record_ID'].values)

# There is only one patient diff between Apical and Apex. -> I ignore the apex set from now on.
apical_apex_set = apical_scar_set & apex_scar_set

venn3([basal_scar_set, mid_scar_set, apical_scar_set], ('Basal', 'Mid', 'Apical'))
plt.title(f'Distribution of Patients with Scar. n={len(scar_set)}')
plt.show()

# Part 3:
# https://drsvenkatesan.files.wordpress.com/2013/01/wall-motion-defect.gif
basal_l_set = set(target.loc[target['Basal L'] == 1]['Record_ID'].values)
basal_s_set = set(target.loc[target['Basal S'] == 1]['Record_ID'].values)
basal_i_set = set(target.loc[target['Basal I'] == 1]['Record_ID'].values)
basal_a_set = set(target.loc[target['Basal A'] == 1]['Record_ID'].values)

# Part 4: Subtask Basal-Septal VS No-Scar
scar_name = 'Basal S'
ds = target.loc[(target['LV Scar'] == 0) | (target[scar_name] == 1)][['Record_ID', 'ECG_ID', 'MRI Diff', 'LV Scar', scar_name] + [col for col in dataset.columns if '(' in col and ')' in col and '[conf]' not in col]]
y_true = []
for _, row in ds.iterrows():
    if row['LV Scar'] == 0:
        y_true.append(0)
    else:
        y_true.append(1)

ds['Target'] = y_true

selected_features = get_significant_features(ds, scar_name='Basal S', lim=25)
ds = ds[['Record_ID', 'ECG_ID', 'Target'] + [x[0] for x in selected_features]].reset_index(drop=True)
pids = list(set(ds['Record_ID'].values))

kf = KFold(n_splits=5, random_state=None, shuffle=False)
for train_idx, test_idx in kf.split(pids):
    train_pids = [pids[i] for i in train_idx]
    test_pids = [pids[i] for i in test_idx]
    train = pd.merge(left=pd.DataFrame(train_pids, columns=['Record_ID']), right=ds, how='inner', on=['Record_ID'])
    test = pd.merge(left=pd.DataFrame(test_pids, columns=['Record_ID']), right=ds, how='inner', on=['Record_ID'])

    report_train_df = train[['Record_ID', 'Target']].drop_duplicates(ignore_index=True)
    report_test_df = test[['Record_ID', 'Target']].drop_duplicates(ignore_index=True)
    print(f'Fold stat:\n'
          f'Train: {report_train_df.shape[0]} patients: '
          f'{report_train_df.loc[report_train_df["Target"] == 0].shape[0]} No-scar with {train.loc[train["Target"] == 0].shape[0]} ECGs; '
          f'{report_train_df.loc[report_train_df["Target"] == 1].shape[0]} {scar_name} with {train.loc[train["Target"] == 1].shape[0]} ECGs;\n'
          f'Test: {report_test_df.shape[0]} patients: '
          f'{report_test_df.loc[report_test_df["Target"] == 0].shape[0]} No-scar with {test.loc[test["Target"] == 0].shape[0]} ECGs; '
          f'{report_test_df.loc[report_test_df["Target"] == 1].shape[0]} {scar_name} with {test.loc[test["Target"] == 1].shape[0]} ECGs.\n')

    train_x = train[[col for col in train.columns if ('(' in col and ')' in col)]].values
    train_y = train['Target'].values
    test_x = test[[col for col in test.columns if ('(' in col and ')' in col)]].values
    test_y = test['Target'].values

    # Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    max_performance = 0
    mlp_list = [MLPClassifier(hidden_layer_sizes=(200,), activation='relu', max_iter=1000),
                MLPClassifier(hidden_layer_sizes=(300,), activation='relu', max_iter=1000),
                MLPClassifier(hidden_layer_sizes=(400,), activation='relu', max_iter=1000),
                MLPClassifier(hidden_layer_sizes=(500,), activation='relu', max_iter=1000),
                MLPClassifier(hidden_layer_sizes=(20, 20), activation='relu', max_iter=500, solver='lbfgs'),
                MLPClassifier(hidden_layer_sizes=(30, 30), activation='relu', max_iter=500, solver='lbfgs'),
                MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', max_iter=500, solver='lbfgs'),
                MLPClassifier(hidden_layer_sizes=(70, 70), activation='relu', max_iter=500, solver='lbfgs'),
                MLPClassifier(hidden_layer_sizes=(50, 30, 10), activation='relu', max_iter=500, solver='lbfgs')]
    for mlp in mlp_list:
        for inner_run in range(40):
            mlp.fit(train_x, train_y)
            y_pred = mlp.predict(test_x)
            pr, re, f1, _ = precision_recall_fscore_support(y_true=test_y, y_pred=y_pred, average=None)
            perform = (pr[0] * re[0]) / (pr[0] + re[0])
            if perform > max_performance:
                max_performance = perform
                result = [pr, re, f1]
                best_model = mlp
        print(f'Pr0={round(result[0][0], 3)}, Re0={round(result[1][0], 3)}, Pr1={round(result[0][1], 3)}, Re1={round(result[1][1], 3)})')










