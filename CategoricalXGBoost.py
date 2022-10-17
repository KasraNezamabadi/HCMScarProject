import statistics
import pandas as pd
import DataManagement as DM
from Classification import DatasetManager, SKLearnClassificationManager
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':
    parser = DM.EHRECGParser()

    n_folds = 5
    ds_manager = DatasetManager(ehr_df=parser.ehr_df_imputed,
                                ecg_ds=parser.qt_dataset,
                                n_folds=n_folds,
                                feature_select_ehr=True,
                                development_mode=False)

    for fold in range(n_folds):
        x_train_ecg, train_ehr_df, y_train, x_test_ecg, test_ehr_df, y_test = ds_manager.get_fold_pandas(fold_index=fold,
                                                                                                         augment_mode=None,
                                                                                                         standardize_ehr='pandas')

        continuous_features = list(
            set(train_ehr_df.columns.values) & set(DM.EHRAttributeManager.get_continuous_attrs(include_record_id=False)))
        x_train_ehr_continuous = train_ehr_df[continuous_features]
        x_test_ehr_continuous = test_ehr_df[continuous_features]

        nominal_features = list(set(train_ehr_df.columns.values) & set(DM.EHRAttributeManager.get_nominal_attrs(include_record_id=False)))
        x_train_ehr_nominal = train_ehr_df[nominal_features]
        x_test_ehr_nominal = test_ehr_df[nominal_features]

        clf = XGBClassifier(tree_method='hist', enable_categorical=True)
        # X is the dataframe we created in previous snippet
        clf.fit(x_train_ehr_nominal, y_train)
        y_pred = clf.predict(x_test_ehr_nominal)

        v = 9

