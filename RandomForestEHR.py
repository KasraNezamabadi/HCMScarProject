import statistics
import DataManagement as DM
from Classification import DatasetManager, SKLearnClassificationManager
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    parser = DM.EHRECGParser()
    n_folds = 5
    ds_manager = DatasetManager(ehr_df=parser.ehr_df_imputed,
                                ecg_ds=parser.qt_dataset, n_folds=n_folds,
                                development_mode=False)

    pr_0_list = []
    pr_1_list = []
    re_0_list = []
    re_1_list = []
    for fold in range(n_folds):
        print(f'--- Fold {fold+1} ---')
        x_train_ecg, x_train_ehr, y_train, \
        x_test_ecg, x_test_ehr, y_test, \
        continuous_features_indexes = ds_manager.get_fold(fold_index=fold, augment_mode=None, standardize_ehr=True)

        x_train_ehr_continuous = x_train_ehr[:, continuous_features_indexes]
        x_test_ehr_continuous = x_test_ehr[:, continuous_features_indexes]

        print(f'First data point: {x_train_ehr_continuous[0]}')

        model = RandomForestClassifier()
        classifier = SKLearnClassificationManager(model, x_train=x_train_ehr_continuous, y_train=y_train,
                                                  x_test=x_test_ehr_continuous, y_test=y_test)

        pr_0, re_0, pr_1, re_1 = classifier.fit_and_evaluate()
        pr_0_list.append(pr_0)
        pr_1_list.append(pr_1)
        re_0_list.append(re_0)
        re_1_list.append(re_1)
        print(f'\n--- Overall Performance ---')
        print(
            f'DE=0 -> Precision: {round(statistics.mean(pr_0_list), 2)}, Recall: {round(statistics.mean(re_0_list), 2)}')
        print(
            f'DE=1 -> Precision: {round(statistics.mean(pr_1_list), 2)}, Recall: {round(statistics.mean(re_1_list), 2)}')
