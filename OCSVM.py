import statistics
import DataManagement as DM
from Classification import DatasetManager, SKLearnClassificationManager
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':
    parser = DM.EHRECGParser()
    n_folds = 5
    ds_manager = DatasetManager(ehr_df=parser.ehr_df_imputed,
                                ecg_ds=parser.qt_dataset,
                                n_folds=n_folds,
                                feature_select_ehr=False,
                                development_mode=False)

    for fold in range(n_folds):
        x_train_ecg, x_train_ehr, y_train, x_test_ecg, x_test_ehr, y_test = ds_manager.get_fold(fold_index=fold,
                                                                                                augment_mode=None,
                                                                                                standardize_ehr='pandas')

        x_train_ehr_continuous = x_train_ehr[:, ds_manager.ehr_cont_feature_index]
        x_test_ehr_continuous = x_test_ehr[:, ds_manager.ehr_cont_feature_index]

        x_train_ehr_continuous = x_train_ehr_continuous[y_train == 1, :]
        clf = OneClassSVM(gamma='auto').fit(x_train_ehr_continuous)
        y_pred = clf.predict(x_test_ehr_continuous)
        y_pred = [1 if y == 1 else 0 for y in y_pred]
        print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['DE=0', 'DE=1']))


