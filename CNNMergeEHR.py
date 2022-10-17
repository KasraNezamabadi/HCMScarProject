import DataManagement as DM
from Classification import DatasetManager, ClassificationManager
import tensorflow as tf


def build_model(ecg_input_shape: (int, int, int), ehr_input_shape: int):
    ecg_input = tf.keras.Input(shape=ecg_input_shape, name="ecg")  # -> each datapoint of shape 96 x 4
    ehr_input = tf.keras.Input(shape=ehr_input_shape, name="ehr")

    # Step 1 -> QT segment feature extraction
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, ecg_input_shape[1]), activation=tf.keras.activations.relu,
                               input_shape=(ecg_input_shape[0], ecg_input_shape[1], ecg_input_shape[2]))(ecg_input)
    x = tf.reshape(x, [-1, x.shape[1], x.shape[-1]])
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)
    x = tf.expand_dims(x, axis=-1)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x_dense = tf.keras.layers.Flatten()(x)

    # Step 2 -> Concatenate ECG features extracted via CNN with EHR features
    # NOTE -> Standardize ECG features before concat. The EHR features are already standardized.
    x_dense = tf.keras.layers.LayerNormalization()(x_dense)
    x_dense = tf.keras.layers.concatenate(inputs=[x_dense, ehr_input], axis=1)

    # Step 3 -> Feed everything to FCN
    x_dense = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu)(x_dense)
    x_dense = tf.keras.layers.Dropout(rate=0.3)(x_dense)
    x_dense = tf.keras.layers.LayerNormalization()(x_dense)
    x_dense = tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu)(x_dense)
    x_dense = tf.keras.layers.Dropout(rate=0.3)(x_dense)
    outputs = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)(x_dense)
    return tf.keras.Model(inputs=[ecg_input, ehr_input], outputs=outputs)


if __name__ == '__main__':
    parser = DM.EHRECGParser()
    n_folds = 5
    ds_manager = DatasetManager(ehr_df=parser.ehr_df_imputed, ecg_ds=parser.qt_dataset, n_folds=n_folds,
                                development_mode=True)

    for fold in range(n_folds):
        print(f'--- Fold {fold+1} ---')
        x_train_ecg, x_train_ehr, y_train, x_test_ecg, x_test_ehr, y_test = ds_manager.get_fold(fold_index=fold,
                                                                                                augment_mode='both',
                                                                                                standardize_ehr=True)

        # For Conv2D which only accepts 4D tensor
        x_train_ecg = tf.expand_dims(x_train_ecg, axis=-1)
        x_test_ecg = tf.expand_dims(x_test_ecg, axis=-1)

        input_train = {'ecg': x_train_ecg, 'ehr': x_train_ehr}
        input_test = {'ecg': x_test_ecg, 'ehr': x_test_ehr}

        model = build_model(ecg_input_shape=(x_train_ecg.shape[1], x_train_ecg.shape[2], 1),
                            ehr_input_shape=x_train_ehr.shape[1])

        classifier = ClassificationManager(model=model,
                                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                           input_train=input_train, input_test=input_test,
                                           target_train=y_train, target_test=y_test, model_name='CNNMergeEHR',
                                           plot=True, fold_index=fold)

        # print(model.summary())
        classifier.fit_and_evaluate()
