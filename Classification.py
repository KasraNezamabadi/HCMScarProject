import pandas as pd
import numpy as np
import random
from datetime import datetime
import sklearn
import sklearn.naive_bayes as naive_bayes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import DataManagement as DM


class DatasetManager:
    def __init__(self, ehr_df: pd.DataFrame, ecg_ds: np.ndarray, n_folds: int, feature_select_ehr: bool = True,
                 development_mode: bool = False):

        if ehr_df.shape[0] != ecg_ds.shape[0]:
            assert 'EHR and ECG Dataset must have same number of data points'
        self.ehr_df = ehr_df
        self.ecg_ds = ecg_ds
        self.ehr_cont_feature_index = []

        continuous_features_set = set(DM.EHRAttributeManager.get_continuous_attrs(include_record_id=False))
        ehr_features = list(self.ehr_df.columns.values)
        for i in range(len(ehr_features)):
            feature = ehr_features[i]
            if feature in continuous_features_set:
                self.ehr_cont_feature_index.append(i)

        if feature_select_ehr:
            self._keep_top_ehr_features(verbose=False)

        if development_mode:
            print('Development Mode On -> Keep only 75 data points for speed in debugging')
            self.ehr_df = self.ehr_df.iloc[:75]
            self.ecg_ds = self.ecg_ds[:75]

        self.ehr_dataset = []
        self.ecg_dataset = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)
        for train_index, test_index in kf.split(self.ehr_df):
            ehr_train, ehr_test = self.ehr_df.iloc[train_index], self.ehr_df.iloc[test_index]
            ehr_train = ehr_train.reset_index(drop=True)
            ehr_test = ehr_test.reset_index(drop=True)
            ecg_train, ecg_test = self.ecg_ds[train_index], self.ecg_ds[test_index]
            self.ehr_dataset.append((ehr_train, ehr_test))
            self.ecg_dataset.append((ecg_train, ecg_test))

    def get_fold_pandas(self, fold_index: int,
                        augment_mode: str = None,
                        standardize_ehr: str = None):

        ehr_train, ehr_test = self.ehr_dataset[fold_index][0], self.ehr_dataset[fold_index][1]
        ecg_train, ecg_test = self.ecg_dataset[fold_index][0], self.ecg_dataset[fold_index][1]

        if augment_mode is not None:
            augmentor = DM.ScarAugmentor(ehr_df=ehr_train, ecg_ds=ecg_train)
            ehr_augmented, ecg_augmented = augmentor.smote_ehr_ecg(dist_mode=augment_mode)

            ehr_train = pd.concat([ehr_train, ehr_augmented], axis=0)
            ehr_train = ehr_train.sample(frac=1).reset_index(drop=True)
            ecg_train = np.concatenate((ecg_train, ecg_augmented))
            print('--- Augmentation Done!')

        # TODO -> Remove it before augmentation?!
        ehr_train = ehr_train.drop(columns='Reason for termination')
        ehr_test = ehr_test.drop(columns='Reason for termination')

        if standardize_ehr is not None:
            if standardize_ehr == 'onehot':
                ehr_train, ehr_test = self._standardize_one_hot_ehr(ehr_train=ehr_train, ehr_test=ehr_test)
            elif standardize_ehr == 'pandas':
                ehr_train, ehr_test = self._standardize_pandas_ehr(ehr_train=ehr_train, ehr_test=ehr_test)

        x_train_ecg, _, y_train = self._get_numpy_x_y(ecg_ds=ecg_train, ehr_df=ehr_train)
        x_test_ecg, _, y_test = self._get_numpy_x_y(ecg_ds=ecg_test, ehr_df=ehr_test)

        return x_train_ecg, ehr_train, y_train, x_test_ecg, ehr_test, y_test

    def get_fold(self, fold_index: int,
                 augment_mode: str = None,
                 standardize_ehr: str = None):

        ehr_train, ehr_test = self.ehr_dataset[fold_index][0], self.ehr_dataset[fold_index][1]
        ecg_train, ecg_test = self.ecg_dataset[fold_index][0], self.ecg_dataset[fold_index][1]

        # TODO -> Remove it before augmentation?!
        ehr_train = ehr_train.drop(columns='Reason for termination')
        ehr_test = ehr_test.drop(columns='Reason for termination')

        if augment_mode is not None:
            augmentor = DM.ScarAugmentor(ehr_df=ehr_train, ecg_ds=ecg_train)
            ehr_augmented = augmentor.smote_ehr()
            # ehr_augmented, ecg_augmented = augmentor.smote_ehr_ecg(dist_mode=augment_mode)

            ehr_train = pd.concat([ehr_train, ehr_augmented], axis=0)
            ehr_train = ehr_train.sample(frac=1).reset_index(drop=True)
            # ecg_train = np.concatenate((ecg_train, ecg_augmented))
            print('--- Augmentation Done!')

        if standardize_ehr is not None:
            if standardize_ehr == 'onehot':
                ehr_train, ehr_test = self._standardize_one_hot_ehr(ehr_train=ehr_train, ehr_test=ehr_test)
            elif standardize_ehr == 'pandas':
                ehr_train, ehr_test = self._standardize_pandas_ehr(ehr_train=ehr_train, ehr_test=ehr_test)

        x_train_ecg, x_train_ehr, y_train = self._get_numpy_x_y(ecg_ds=ecg_train, ehr_df=ehr_train)
        x_test_ecg, x_test_ehr, y_test = self._get_numpy_x_y(ecg_ds=ecg_test, ehr_df=ehr_test)

        return x_train_ecg, x_train_ehr, y_train, x_test_ecg, x_test_ehr, y_test



    def get_fold_ehr(self, fold_index: int,
                     augment_mode: str = None,
                     standardize_ehr: str = None):

        ehr_train, ehr_test = self.ehr_dataset[fold_index][0], self.ehr_dataset[fold_index][1]

        # TODO -> Remove it before augmentation?!
        ehr_train = ehr_train.drop(columns='Reason for termination')
        ehr_test = ehr_test.drop(columns='Reason for termination')

        if augment_mode is not None:
            augmentor = DM.ScarAugmentor(ehr_df=ehr_train, ecg_ds=[])
            ehr_augmented = augmentor.smote_ehr()
            ehr_train = pd.concat([ehr_train, ehr_augmented], axis=0)
            ehr_train = ehr_train.sample(frac=1).reset_index(drop=True)
            # ecg_train = np.concatenate((ecg_train, ecg_augmented))
            print('--- Augmentation Done!')

        if standardize_ehr is not None:
            if standardize_ehr == 'onehot':
                ehr_train, ehr_test = self._standardize_one_hot_ehr(ehr_train=ehr_train, ehr_test=ehr_test)
            elif standardize_ehr == 'pandas':
                ehr_train, ehr_test = self._standardize_pandas_ehr(ehr_train=ehr_train, ehr_test=ehr_test)

        x_train_ehr = ehr_train.values
        x_train_ehr = np.array(x_train_ehr[:, :-2]).astype('float32')
        y_train = np.array(ehr_train['DE'].values).astype('float32')
        x_test_ehr = ehr_test.values
        x_test_ehr = np.array(x_test_ehr[:, :-2]).astype('float32')
        y_test = np.array(ehr_test['DE'].values).astype('float32')

        return x_train_ehr, y_train, x_test_ehr, y_test


    def get_fold_development(self, fold_index: int,
                 augment_mode: str = None,
                 standardize_ehr: bool = True):

        ehr_train, ehr_test = self.ehr_dataset[fold_index][0], self.ehr_dataset[fold_index][1]
        ecg_train, ecg_test = self.ecg_dataset[fold_index][0], self.ecg_dataset[fold_index][1]

        if augment_mode is not None:
            augmentor = DM.ScarAugmentor(ehr_df=ehr_train, ecg_ds=ecg_train)
            ehr_augmented, ecg_augmented = augmentor.smote_ehr_ecg(dist_mode=augment_mode)

            ehr_train = pd.concat([ehr_train, ehr_augmented], axis=0)
            ehr_train = ehr_train.sample(frac=1).reset_index(drop=True)
            ecg_train = np.concatenate((ecg_train, ecg_augmented))
            print('--- Augmentation Done!')

        # TODO -> Remove it before augmentation?!
        ehr_train = ehr_train.drop(columns='Reason for termination')
        ehr_test = ehr_test.drop(columns='Reason for termination')

        if standardize_ehr:
            ehr_train, ehr_test = self._standardize_one_hot_ehr(ehr_train=ehr_train, ehr_test=ehr_test)

        x_train_ecg, x_train_ehr, y_train = self._get_numpy_x_y(ecg_ds=ecg_train, ehr_df=ehr_train)
        x_test_ecg, x_test_ehr, y_test = self._get_numpy_x_y(ecg_ds=ecg_test, ehr_df=ehr_test)

        return x_train_ecg, x_train_ehr, y_train, x_test_ecg, x_test_ehr, y_test, ehr_train, ehr_test, ecg_train, ecg_test

    def _keep_top_ehr_features(self, verbose: bool = False):
        f_selector = DM.EHRFeatureSelection(ehr_df=self.ehr_df)
        continuous_features = f_selector.get_top_continuous_features()
        nominal_features = f_selector.get_top_nominal_features()
        if verbose:
            print('\nPerforming Feature Selection')
            print(f'--- Top Nominal Features: {nominal_features}')
            print(f'--- Top Continuous Features: {continuous_features}')
        selected_cols = ['Record_ID', 'DE'] + continuous_features + nominal_features
        self.ehr_df = self.ehr_df.reindex(columns=selected_cols)
        self.ehr_cont_feature_index = list(range(len(continuous_features)))

    @staticmethod
    def _get_numpy_x_y(ecg_ds: np.ndarray, ehr_df: pd.DataFrame):
        """
        Convert EHR and ECG data structures to numpy matrix compatible with ML models in keras and sklearn
        :param ecg_ds: A list of dict each has `pid`: int, `de`: int, `preprocessed`: list of 4 x 96 QT segments
        :type ndarray of dict
        :param ehr_df: imputed EHR data frame (no missing value)
        :type DataFrame
        :return: Three ndarray: ecg matrix, ehr matrix, and labels (de)
        """
        x_ecg = []
        x_ehr = []
        y = []
        for ecg_object in ecg_ds:
            pid = ecg_object['pid']
            de = float(ecg_object['de'])
            ehr_row = list(ehr_df.loc[ehr_df['Record_ID'] == pid].values[0])
            ehr_row = ehr_row[:-1]  # -> Last element is Record_ID
            qt_segments = ecg_object['preprocessed']
            qt_segment = qt_segments[random.randint(0, len(qt_segments) - 1)]

            qt_segment = np.array(qt_segment)
            if qt_segment.shape[1] != 4:
                qt_segment = np.transpose(qt_segment)
            # ehr_row = np.array(ehr_row)

            x_ecg.append(qt_segment)
            # x_ehr.append(ehr_row)
            # y.append(de)
        x_ecg = np.array(x_ecg).astype('float32')
        x_ehr = np.array(x_ehr).astype('float32')
        y = np.array(y).astype('float32')
        return x_ecg, x_ehr, y

    @staticmethod
    def _standardize_one_hot_ehr(ehr_train: pd.DataFrame, ehr_test: pd.DataFrame):
        train_record_ids = ehr_train['Record_ID'].values
        test_record_ids = ehr_test['Record_ID'].values
        train_des = ehr_train['DE'].values
        test_des = ehr_test['DE'].values

        # -> Standardize Continuous Part of EHR DataFrame
        ehr_train_continuous = DM.EHRAttributeManager.get_continuous_df(df=ehr_train)
        ehr_test_continuous = DM.EHRAttributeManager.get_continuous_df(df=ehr_test)
        standard_scaler = StandardScaler()
        standard_scaler.fit(ehr_train_continuous)
        ehr_train_continuous = pd.DataFrame(data=standard_scaler.transform(ehr_train_continuous),
                                            columns=ehr_train_continuous.columns.values)
        ehr_test_continuous = pd.DataFrame(data=standard_scaler.transform(ehr_test_continuous),
                                           columns=ehr_test_continuous.columns.values)

        # -> One-Hot Encode Nominal Features
        train_nominal_df = DM.EHRAttributeManager.get_nominal_df(df=ehr_train)
        test_nominal_df = DM.EHRAttributeManager.get_nominal_df(df=ehr_test)

        one_hot_encoder = OneHotEncoder(drop='if_binary')
        one_hot_encoder.fit(pd.concat([train_nominal_df, test_nominal_df]))

        ehr_train_nominal = pd.DataFrame(one_hot_encoder.transform(train_nominal_df).toarray())
        ehr_test_nominal = pd.DataFrame(one_hot_encoder.transform(test_nominal_df).toarray())

        result_train = pd.concat([ehr_train_continuous, ehr_train_nominal], axis=1)
        result_test = pd.concat([ehr_test_continuous, ehr_test_nominal], axis=1)
        result_train['Record_ID'] = train_record_ids
        result_test['Record_ID'] = test_record_ids
        result_train['DE'] = train_des
        result_test['DE'] = test_des

        return result_train, result_test

    @staticmethod
    def _standardize_pandas_ehr(ehr_train: pd.DataFrame, ehr_test: pd.DataFrame):
        train_record_ids = ehr_train['Record_ID'].values
        test_record_ids = ehr_test['Record_ID'].values

        # -> Standardize Continuous Part of EHR DataFrame
        ehr_train_continuous = DM.EHRAttributeManager.get_continuous_df(df=ehr_train)
        ehr_test_continuous = DM.EHRAttributeManager.get_continuous_df(df=ehr_test)
        standard_scaler = StandardScaler()
        standard_scaler.fit(ehr_train_continuous)
        ehr_train_continuous = pd.DataFrame(data=standard_scaler.transform(ehr_train_continuous),
                                            columns=ehr_train_continuous.columns.values)
        ehr_test_continuous = pd.DataFrame(data=standard_scaler.transform(ehr_test_continuous),
                                           columns=ehr_test_continuous.columns.values)

        # -> Convert nominal columns to pandas categorical data type
        train_nominal_df = DM.EHRAttributeManager.get_nominal_df(df=ehr_train)
        test_nominal_df = DM.EHRAttributeManager.get_nominal_df(df=ehr_test)

        for column in train_nominal_df.columns:
            train_nominal_df[column] = train_nominal_df[column].astype('category')
            test_nominal_df[column] = test_nominal_df[column].astype('category')

        result_train = pd.concat([ehr_train_continuous, train_nominal_df], axis=1)
        result_test = pd.concat([ehr_test_continuous, test_nominal_df], axis=1)
        result_train['Record_ID'] = train_record_ids
        result_test['Record_ID'] = test_record_ids

        return result_train, result_test


class ClassificationManager:
    def __init__(self, model: tf.keras.Model,
                 loss: tf.keras.losses,
                 input_train: dict, target_train: np.ndarray,
                 input_test: dict, target_test: np.ndarray,
                 epochs: int = 10, batch_size: int = 8, validation_split: float = 0.1,
                 model_name: str = 'my_model', plot: bool = True, fold_index: int = None):

        self.model = model
        self.loss = loss
        self.input_train = input_train
        self.target_train = target_train
        self.input_test = input_test
        self.target_test = target_test
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='ClassificationResult/Checkpoints',
                                                                      monitor='val_accuracy',
                                                                      mode='max',
                                                                      save_best_only=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model_name = model_name
        self.should_plot = plot
        self.fold_index = fold_index
        #

    def _compile(self):
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def _plot_train_val_loss(self, history: tf.keras.callbacks.History):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss ({self.model_name})')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        pic_name = self.model_name
        if self.fold_index is not None:
            pic_name += f'_fold_{self.fold_index}_'

        date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        pic_name += f'_{date_str}.png'
        plt.show()
        plt.savefig(f'ClassificationResult/Plots/{pic_name}', bbox_inches='tight')
        # plt.clf()
        # plt.close()

    def fit_and_evaluate(self, verbose: bool = True):
        self._compile()
        history = self.model.fit(x=self.input_train,
                                 y=self.target_train,
                                 validation_split=self.validation_split,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 callbacks=self.checkpoint_callback,
                                 verbose=False)

        y_pred = self.model.predict(x=self.input_test)
        if len(y_pred[0]) > 1:  # -> Last layer is Softmax with more than 1 units
            y_pred = [np.argmax(prediction) for prediction in y_pred]
        else:  # Last layer is Sigmoid with only 1 unit
            y_pred = [1 if prediction[0] >= 0.5 else 0 for prediction in y_pred]

        if verbose:
            print('Confusion Matrix')
            conf_matrix = confusion_matrix(y_true=self.target_test, y_pred=y_pred, labels=[0, 1])
            print(conf_matrix)
            print('Classification Report')
            print(classification_report(self.target_test, y_pred, labels=[0, 1], target_names=['DE=0', 'DE=1']))
        if self.should_plot:
            self._plot_train_val_loss(history=history)

        report = classification_report(self.target_test, y_pred, labels=[0, 1], target_names=['DE=0', 'DE=1'],
                                       output_dict=True)
        return report['DE=0']['precision'], \
               report['DE=0']['recall'], \
               report['DE=1']['precision'], \
               report['DE=1']['recall']


class SKLearnClassificationManager:
    def __init__(self, model,
                 x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def fit_and_evaluate(self, verbose: bool = True):
        self.model.fit(X=self.x_train, y=self.y_train)
        y_pred = self.model.predict(X=self.x_test)
        if verbose:
            print(f'First 5 Actual Labels: {self.y_test[:5]}')
            print(f'First 5 Predicted Labels: {y_pred[:5]}')
            print('Confusion Matrix')
            conf_matrix = confusion_matrix(y_true=self.y_test, y_pred=y_pred, labels=[0, 1])
            print(conf_matrix)
            print('Classification Report')
            print(classification_report(self.y_test, y_pred, labels=[0, 1], target_names=['DE=0', 'DE=1']))
        report = classification_report(self.y_test, y_pred, labels=[0, 1], target_names=['DE=0', 'DE=1'], output_dict=True)
        return report['DE=0']['precision'], report['DE=0']['recall'], report['DE=1']['precision'], report['DE=1']['recall']

