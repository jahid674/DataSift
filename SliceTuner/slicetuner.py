# slicetuner_runner.py
from __future__ import absolute_import, division, print_function

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import tensorflow as tf

from baseline import Baseline         # <-- your saved Baseline
from system_t import System_T         # <-- your saved System_T


SliceSpec = Union[int, Tuple[int, int]]  # single-indicator column or (col_a, col_b) intersection


class SliceTunerRunner:
    """
    Wraps your existing Baseline/System_T to work with prebuilt inputs:
      - X_train, y_train (seed)
      - X_val,   y_val   (global validation)
      - val_data_dict: {slice_id: (X_val_slice, y_val_slice)}
      - add_data_dict: {slice_id: (X_pool_slice, y_pool_slice)}
      - slice_index: List[SliceSpec] where each spec is either an int (indicator column)
                     or a (col_a, col_b) tuple meaning intersection.
      - slice_desc:   List[str] names for reporting (optional but recommended)
      - privileged_slice_indices / protected_slice_indices: fairness groups by slice id
      - favorable_label: which label is counted as “positive” for metrics (0 or 1)

    This class:
      * normalizes labels to one-hot with a consistent num_label,
      * computes initial_data_array by counting X_train slice membership,
      * runs Baseline/System_T with your exact classes.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        val_data_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
        add_data_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
        slice_index:   List[SliceSpec],
        slice_desc:    Optional[List[str]] = None,
        privileged_slice_indices: Optional[List[int]] = None,
        protected_slice_indices:  Optional[List[int]] = None,
        favorable_label: int = 1,
    ):
        # --- store raw arrays ---
        self.X_train = np.asarray(X_train)
        self.y_train_raw = np.asarray(y_train)
        self.X_val   = np.asarray(X_val)
        self.y_val_raw   = np.asarray(y_val)

        # per-slice dicts
        self.val_data_dict  = {int(k): (np.asarray(v[0]), np.asarray(v[1])) for k, v in val_data_dict.items()}
        self.add_data_dict  = {int(k): (np.asarray(v[0]), np.asarray(v[1])) for k, v in add_data_dict.items()}

        # slices
        self.slice_index = list(slice_index)
        self.slice_desc  = list(slice_desc) if slice_desc is not None else [f"slice_{i}" for i in range(len(slice_index))]
        assert len(self.slice_index) == len(self.slice_desc), "slice_index and slice_desc must align"

        self.num_class = len(self.slice_index)

        # fairness config
        self.favorable_label = int(favorable_label)
        self.privileged_slice_indices = list(privileged_slice_indices or [])
        self.protected_slice_indices  = list(protected_slice_indices or [])

        # --- normalize labels to one-hot everywhere with consistent num_label ---
        self.num_label, self.Y_train, self.Y_val = self._normalize_labels(self.y_train_raw, self.y_val_raw)

        # fix per-slice dict label widths
        self._normalize_dict_labels(self.val_data_dict, self.num_label)
        self._normalize_dict_labels(self.add_data_dict, self.num_label)

        # --- compute initial_data_array from X_train + slice_index ---
        self.initial_data_array = self._count_per_slice(self.X_train, self.slice_index).astype(int)

    # ---------------------- public API ----------------------

    def run_baseline(
        self,
        method: str,
        budget: int,
        cost_func: Union[List[float], np.ndarray],
        lr: float = 1e-3,
        epochs: int = 2000,
        num_iter: int = 10,
        batch_size: int = 32,
    ) -> Baseline:
        """
        Calls your Baseline with the prepared inputs.
        """
        bl = Baseline(
            (self.X_train, self.Y_train),
            (self.X_val,   self.Y_val),
            self.val_data_dict,
            self.initial_data_array,
            self.num_class,
            self.num_label,
            self.slice_index,
            self.add_data_dict,
            method=method,
            favorable_label=self.favorable_label,
            privileged_slice_indices=self.privileged_slice_indices,
            protected_slice_indices=self.protected_slice_indices
        )
        bl.performance(
            budget=int(budget),
            cost_func=np.array(cost_func, dtype=float),
            num_iter=int(num_iter),
            batch_size=int(batch_size),
            lr=float(lr),
            epochs=int(epochs)
        )
        return bl

    def run_system_t(
        self,
        budget: int,
        cost_func: Union[List[float], np.ndarray],
        lr: float = 1e-3,
        epochs: int = 2000,
        k: int = 10,
        Lambda: float = 0.1,
        num_iter: int = 5,
        strategy: str = "Moderate",
        show_figure: bool = False,
        batch_size: int = 32,
    ) -> System_T:
        """
        Calls your System_T with the prepared inputs.
        """
        st = System_T(
            (self.X_train, self.Y_train),
            (self.X_val,   self.Y_val),
            self.val_data_dict,
            self.initial_data_array,
            self.num_class,
            self.num_label,
            self.slice_index,
            self.add_data_dict,
            favorable_label=self.favorable_label,
            privileged_slice_indices=self.privileged_slice_indices,
            protected_slice_indices=self.protected_slice_indices
        )
        st.selective_collect(
            budget=int(budget),
            k=int(k),
            batch_size=int(batch_size),
            lr=float(lr),
            epochs=int(epochs),
            cost_func=np.array(cost_func, dtype=float),
            Lambda=float(Lambda),
            num_iter=int(num_iter),
            slice_desc=self.slice_desc,
            strategy=str(strategy),
            show_figure=bool(show_figure)
        )
        return st

    # ---------------------- helpers ----------------------

    def _normalize_labels(self, y_train_raw: np.ndarray, y_val_raw: np.ndarray):
        """
        Ensure both train/val labels are one-hot with the same num_label.
        Accepts 1D integer labels or already-one-hot arrays.
        """
        # detect num_label
        if y_train_raw.ndim == 1:
            num_label = int(len(np.unique(y_train_raw)))
        else:
            num_label = int(y_train_raw.shape[1])

        # to one-hot
        if y_train_raw.ndim == 1:
            Y_train = tf.keras.utils.to_categorical(y_train_raw.astype(int), num_classes=num_label)
        else:
            Y_train = y_train_raw

        if y_val_raw.ndim == 1:
            Y_val = tf.keras.utils.to_categorical(y_val_raw.astype(int), num_classes=num_label)
        else:
            Y_val = y_val_raw

        # shape checks
        assert Y_train.shape[1] == num_label, "train one-hot width mismatch"
        assert Y_val.shape[1]   == num_label, "val one-hot width mismatch"
        return num_label, Y_train, Y_val

    def _normalize_dict_labels(self, data_dict: Dict[int, Tuple[np.ndarray, np.ndarray]], num_label: int):
        """
        Ensure per-slice dict labels are one-hot with consistent width.
        """
        for k in list(data_dict.keys()):
            Xk, yk = data_dict[k]
            if yk.ndim == 1:
                yk = tf.keras.utils.to_categorical(yk.astype(int), num_classes=num_label)
            else:
                # If slice happens to miss a class, pad to num_label
                if yk.shape[1] != num_label:
                    # pad or re-map if necessary (simple safe pad to right)
                    y_pad = np.zeros((yk.shape[0], num_label), dtype=yk.dtype)
                    y_pad[:, :yk.shape[1]] = yk
                    yk = y_pad
            data_dict[k] = (Xk, yk)

        # sanity: number of slices should match
        assert len(data_dict) == self.num_class, "data_dict length must match num_class"

    def _count_per_slice(self, X: np.ndarray, slice_index: List[SliceSpec]) -> np.ndarray:
        """
        Count training examples per slice from X using slice_index specs.
        Supports:
          - int: column index where X[:, idx] == 1
          - (a, b): both columns equal 1 (intersection)
        """
        counts = []
        for spec in slice_index:
            if isinstance(spec, tuple):
                a, b = int(spec[0]), int(spec[1])
                counts.append(int(np.sum((X[:, a] == 1) & (X[:, b] == 1))))
            else:
                idx = int(spec)
                counts.append(int(np.sum(X[:, idx] == 1)))
        return np.array(counts, dtype=int)
