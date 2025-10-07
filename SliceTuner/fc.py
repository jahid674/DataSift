from __future__ import absolute_import, division, print_function

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")  # pick a GPU if you want

import numpy as np
import copy
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Configure GPU (TF2/Keras 3)
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        tf.config.set_visible_devices(_gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(_gpus[0], True)
    except RuntimeError as e:
        print("GPU config already set:", e)


class FC:
    """
    Minimal softmax classifier wrapper used by the Baseline runner.

    Returns per-slice:
      - loss
      - n_examples
      - n_positive_predictions (for chosen favorable label)
      - positive_rate
    """

    def __init__(self, train_data, train_label, val_data, val_label, val_data_dict,
                 batch_size, epochs, lr, num_class, num_label, slice_index,
                 favorable_label=1):
        self.train_data = (copy.deepcopy(train_data), copy.deepcopy(train_label))
        self.val_data   = (copy.deepcopy(val_data),   copy.deepcopy(val_label))
        self.val_data_dict = copy.deepcopy(val_data_dict)
        self.batch_size = int(batch_size)
        self.epochs     = int(epochs)
        self.lr         = float(lr)
        self.num_class  = int(num_class)
        self.num_label  = int(num_label)
        self.slice_index = copy.deepcopy(slice_index)
        self.favorable_label = int(favorable_label)

        self.model = self._build_model()

    def _build_model(self):
        input_dim = int(self.train_data[0].shape[1])
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(self.num_label, activation='softmax')
        ])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            metrics=['accuracy']
        )
        return model

    def fc_train(self, process_num):
        """
        Returns:
            loss_list:        [num_class] float
            pos_rate_list:    [num_class] float
            pos_count_list:   [num_class] int    (prediction positives)
            count_list:       [num_class] int    (per-slice validation size)
            process_num:      int
        """
        ckpt_path = f"Model_{process_num}_{os.getpid()}.weights.h5"
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=False)
        cp = ModelCheckpoint(filepath=ckpt_path, monitor='val_loss',
                             save_best_only=True, save_weights_only=True, verbose=0)

        self.model.fit(
            self.train_data[0], self.train_data[1],
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            validation_data=(self.val_data[0], self.val_data[1]),
            callbacks=[es, cp]
        )
        if os.path.exists(ckpt_path):
            self.model.load_weights(ckpt_path)

        loss_list, pos_rate_list, pos_count_list, count_list = [], [], [], []
        for i in range(self.num_class):
            Xi, yi = self.val_data_dict[i]
            n_i = int(len(Xi))
            count_list.append(n_i)

            # slice loss
            loss_i = self.model.evaluate(Xi, yi, verbose=0)[0]
            loss_list.append(float(loss_i))

            # predictions → positive counts and rates
            if n_i > 0:
                preds = self.model.predict(Xi, verbose=0)
                pred_labels = np.argmax(preds, axis=1)
                pos_i = int(np.sum(pred_labels == self.favorable_label))
                pos_count_list.append(pos_i)
                pos_rate_list.append(float(pos_i / n_i))
            else:
                pos_count_list.append(0)
                pos_rate_list.append(0.0)

        keras.backend.clear_session()
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        except Exception:
            pass

        return loss_list, pos_rate_list, pos_count_list, count_list, process_num
