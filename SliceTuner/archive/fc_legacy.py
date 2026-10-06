from __future__ import absolute_import, division, print_function

# (Optional) pick a specific GPU BEFORE importing tensorflow
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")  # machine GPU #2

import numpy as np
import copy
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configure GPU in TF2/Keras 3 (no sessions)
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        # after masking, there may be only one visible GPU → index 0
        tf.config.set_visible_devices(_gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(_gpus[0], True)
    except RuntimeError as e:
        print("GPU config already set:", e)

class FC:
    def __init__(self, train_data, train_label, val_data, val_label, val_data_dict,
                 batch_size, epochs, lr, num_class, num_label, slice_index):
        """
        Args:
            train_data: Training data
            train_label: Training label (one-hot for categorical_crossentropy)
            val_data: Validation data
            val_label: Validation label
            val_data_dict: dict/list with per-slice (X, y)
            batch_size: batch size
            epochs: training epochs
            lr: learning rate
            num_class: number of slices
            num_label: number of classes
            slice_index: e.g., [(0,3), ...] for feature-based slice filters
        """
        self.train_data = (copy.deepcopy(train_data), copy.deepcopy(train_label))
        self.val_data   = (copy.deepcopy(val_data),   copy.deepcopy(val_label))
        self.batch_size = batch_size
        self.epochs     = epochs
        self.lr         = lr
        self.val_data_dict = copy.deepcopy(val_data_dict)
        self.num_class  = num_class
        self.num_label  = num_label
        self.slice_index = slice_index

        self.model = self.build_FC_classifier()

    def build_FC_classifier(self):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(12,)))  # adjust if feature dim != 62
        model.add(layers.Dense(self.num_label, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            metrics=['accuracy']
        )
        return model

    def fc_train(self, process_num):
        slice_num = self.check_num()

        # unique checkpoint per process
        ckpt_path = f"Model_{process_num}_{os.getpid()}.weights.h5"
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=False)
        cp = ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )

        self.model.fit(
            self.train_data[0], self.train_data[1],
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            validation_data=(self.val_data[0], self.val_data[1]),
            callbacks=[es, cp]
        )

        self.model.load_weights(ckpt_path)

        loss_list = []
        for i in range(self.num_class):
            Xi, yi = self.val_data_dict[i]
            loss_i = self.model.evaluate(Xi, yi, verbose=0)[0]
            loss_list.append(loss_i)

        # clean up to avoid graph/memory bloat across processes
        keras.backend.clear_session()
        return loss_list, slice_num, process_num

    def check_num(self):
        counts = {}
        X = self.train_data[0]
        for i in range(self.num_class):
            a, b = self.slice_index[i]
            counts[i] = int(np.sum((X[:, a] == 1) & (X[:, b] == 1)))
        return counts
