import statistics
import DataManagement as DM
from Classification import DatasetManager, ClassificationManager
import os

import tensorflow as tf


def build_model(ehr_input_shape: int):
    ehr_input = tf.keras.Input(shape=ehr_input_shape, name="ehr")

    x_dense = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(ehr_input)
    # x_dense = tf.keras.layers.Dropout(rate=0.3)(x_dense)
    x_dense = tf.keras.layers.LayerNormalization()(x_dense)
    x_dense = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(x_dense)
    # x_dense = tf.keras.layers.Dropout(rate=0.3)(x_dense)
    # x_dense = tf.keras.layers.LayerNormalization()(x_dense)
    outputs = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)(x_dense)
    return tf.keras.Model(inputs=ehr_input, outputs=outputs)


def build_model_v2(ehr_input_shape: int):
    ehr_input = tf.keras.Input(shape=ehr_input_shape, name="ehr")

    x_dense = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(ehr_input)
    x_dense = tf.keras.layers.Dropout(rate=0.3)(x_dense)
    x_dense = tf.keras.layers.LayerNormalization()(x_dense)
    x_dense = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(x_dense)
    x_dense = tf.keras.layers.Dropout(rate=0.3)(x_dense)
    outputs = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)(x_dense)
    return tf.keras.Model(inputs=ehr_input, outputs=outputs)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    parser = DM.EHRECGParser()
    n_folds = 5
    pr_0_list_total = []
    pr_1_list_total = []
    re_0_list_total = []
    re_1_list_total = []
    has_shown_cardinality = False
    for exp in range(10):
        print(f'\n--- Run {exp+1} ---')
        ds_manager = DatasetManager(ehr_df=parser.ehr_df_imputed,
                                    ecg_ds=parser.qt_dataset,
                                    n_folds=n_folds,
                                    feature_select_ehr=True,
                                    development_mode=True)

        fold_pr_0_list = []
        fold_pr_1_list = []
        fold_re_0_list = []
        fold_re_1_list = []
        for fold in range(n_folds):
            # print(f'\n--- Fold {fold+1} ---')
            x_train_ecg, x_train_ehr, y_train, x_test_ecg, x_test_ehr, y_test = ds_manager.get_fold_ehr(fold_index=fold,
                                                                                                        augment_mode='ehr',
                                                                                                        standardize_ehr='onehot')
            if not has_shown_cardinality:
                print(f'\nTrain -> |DE=0| = {sum(y_train == 0)}, |DE=1| = {sum(y_train == 1)}')
                print(f'Test -> |DE=0| = {sum(y_test == 0)}, |DE=1| = {sum(y_test == 1)}')
                has_shown_cardinality = True

            input_train = {'ehr': x_train_ehr}
            input_test = {'ehr': x_test_ehr}
            model = build_model(ehr_input_shape=x_train_ehr.shape[1])
            classifier = ClassificationManager(model=model,
                                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                               input_train=input_train, input_test=input_test,
                                               target_train=y_train, target_test=y_test, model_name='FCNForEHR',
                                               epochs=40,
                                               plot=False, fold_index=fold)

            # print(model.summary())
            pr_0, re_0, pr_1, re_1 = classifier.fit_and_evaluate(verbose=False)
            fold_pr_0_list.append(pr_0)
            fold_pr_1_list.append(pr_1)
            fold_re_0_list.append(re_0)
            fold_re_1_list.append(re_1)
        print(f'\nRun {exp+1} Performance:')
        print(
            f'DE=0 -> Precision: {round(statistics.mean(fold_pr_0_list)*100, 1)}, '
            f'Recall: {round(statistics.mean(fold_re_0_list)*100, 1)}')
        print(
            f'DE=1 -> Precision: {round(statistics.mean(fold_pr_1_list)*100, 1)}, '
            f'Recall: {round(statistics.mean(fold_re_1_list)*100, 1)}')

        pr_0_list_total.append(statistics.mean(fold_pr_0_list))
        pr_1_list_total.append(statistics.mean(fold_pr_1_list))
        re_0_list_total.append(statistics.mean(fold_re_0_list))
        re_1_list_total.append(statistics.mean(fold_re_1_list))

    print(f'\n--- Overall Performance ---')
    print(
        f'DE=0 -> Precision: {round(statistics.mean(pr_0_list_total) * 100, 1)}, '
        f'Recall: {round(statistics.mean(re_0_list_total) * 100, 1)}')
    print(
        f'DE=1 -> Precision: {round(statistics.mean(pr_1_list_total) * 100, 1)}, '
        f'Recall: {round(statistics.mean(re_1_list_total) * 100, 1)}')