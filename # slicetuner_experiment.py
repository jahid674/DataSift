# slicetuner_experiment.py
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Tuple, Union, Dict, Optional

# Your existing classes (import from your project)
from baseline import Baseline
from system_t import System_T


class SliceTunerExperiment:
    """
    A small orchestration class to:
      1) Build slices from indicator columns (single or intersections),
      2) Create per-slice train/val/pool splits,
      3) Run Baseline or System_T with your current fairness settings.

    Assumptions:
      - `df` contains all features + the target column.
      - Slice specs refer to **binary indicator** columns (0/1).
        * You can pass single indicators: "sex_Female"
        * or intersections: ("sex_Female", "race_White")
      - You can designate which slice specs belong to privileged/protected groups
        by listing their indices/names in `privileged_specs` / `protected_specs`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        slice_specs: List[Union[str, Tuple[str, str]] ] = None,
        privileged_specs: Optional[List[Union[str, Tuple[str, str], int]]] = None,
        protected_specs: Optional[List[Union[str, Tuple[str, str], int]]]  = None,
        favorable_label: int = 1,
        random_state: int = 42,
    ):
        """
        Args:
            df: Full dataframe with features, indicators, and target.
            target_col: Name of the target column (0/1 integer labels).
            feature_cols: Optional explicit feature column order (else df.drop(target_col).columns).
            slice_specs: List of slice definitions.
                         Each item is either:
                           - a string (indicator col name), or
                           - a (str, str) tuple meaning intersection of two indicator columns == 1.
            privileged_specs / protected_specs:
                         Which slice specs map to privileged / protected groups
                         (you can pass by spec string/tuple or by slice index).
            favorable_label: Which class is "positive" for fairness metrics (0 or 1).
            random_state: For reproducible shuffles per slice.
        """
        assert target_col in df.columns, f"{target_col} not found in df."
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = int(random_state)

        # Features
        if feature_cols is None:
            self.feature_cols = [c for c in df.columns if c != target_col]
        else:
            self.feature_cols = list(feature_cols)

        # Slice specs
        assert slice_specs and len(slice_specs) >= 1, "Provide at least one slice spec."
        self.slice_specs = list(slice_specs)

        # Fairness config
        self.favorable_label = int(favorable_label)
        self.privileged_specs = [] if privileged_specs is None else list(privileged_specs)
        self.protected_specs  = [] if protected_specs  is None else list(protected_specs)

        # Will be populated after build_splits()
        self.slice_index: List[Union[int, Tuple[int, int]]] = []
        self.slice_desc: List[str] = []
        self.val_data_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.add_data_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.initial_data_array: np.ndarray = np.array([], dtype=int)
        self.num_class: int = 0
        self.num_label: int = 0
        self.X_train: np.ndarray = None
        self.Y_train: np.ndarray = None
        self.X_val:   np.ndarray = None
        self.Y_val:   np.ndarray = None
        self._priv_idx: List[int] = []
        self._prot_idx: List[int] = []

    # ---------- public API ----------

    def build_splits(
        self,
        train_per_slice: Union[int, Dict[int, int]] = 200,
        val_per_slice: int = 500,
        shuffle_within_slice: bool = True,
    ):
        """
        Create per-slice train/val/pool and the arrays/dicts Baseline/System_T expect.

        Args:
            train_per_slice: Either a single int (same for all slices) or a dict {slice_id: n_train}.
            val_per_slice: Validation count per slice.
            shuffle_within_slice: Shuffle each slice before taking train/val/pool.
        """
        # Prepare X/y arrays, global num_label
        X_all = self.df[self.feature_cols].to_numpy()
        y_raw = self.df[self.target_col].to_numpy().astype(int)
        self.num_label = int(len(np.unique(y_raw)))
        Y_all = tf.keras.utils.to_categorical(y_raw, num_classes=self.num_label)

        # Resolve each slice mask + description + slice_index entry
        slice_masks = []
        self.slice_index = []
        self.slice_desc = []
        for spec in self.slice_specs:
            if isinstance(spec, str):
                # single indicator == 1
                assert spec in self.feature_cols, f"Indicator '{spec}' not in features."
                col_idx = self.feature_cols.index(spec)
                mask = (X_all[:, col_idx] == 1)
                slice_masks.append(mask)
                self.slice_index.append(col_idx)               # store column index
                self.slice_desc.append(spec)
            else:
                # tuple intersection (both == 1)
                a, b = spec
                assert a in self.feature_cols and b in self.feature_cols, f"Indicators '{spec}' not in features."
                a_idx = self.feature_cols.index(a)
                b_idx = self.feature_cols.index(b)
                mask = (X_all[:, a_idx] == 1) & (X_all[:, b_idx] == 1)
                slice_masks.append(mask)
                self.slice_index.append((a_idx, b_idx))        # store pair of indices
                self.slice_desc.append(f"{a} & {b}")

        self.num_class = len(self.slice_specs)

        # Helper to grab and (optionally) shuffle a slice
        rng = np.random.default_rng(self.random_state)
        def take_slice(X, Y, mask):
            Xs = X[mask]
            Ys = Y[mask]
            if shuffle_within_slice and len(Xs) > 1:
                idx = np.arange(len(Xs))
                rng.shuffle(idx)
                Xs, Ys = Xs[idx], Ys[idx]
            return Xs, Ys

        # Determine per-slice train sizes
        if isinstance(train_per_slice, int):
            train_sizes = {i: int(train_per_slice) for i in range(self.num_class)}
        else:
            train_sizes = {int(i): int(n) for i, n in train_per_slice.items()}

        # Build splits & dicts
        train_blocks_X, train_blocks_Y = [], []
        val_blocks_X,   val_blocks_Y   = [], []
        self.val_data_dict.clear()
        self.add_data_dict.clear()
        initial_sizes = []

        for i, mask in enumerate(slice_masks):
            Xs, Ys = take_slice(X_all, Y_all, mask)

            n_train = min(train_sizes.get(i, 0), len(Xs))
            n_val   = min(val_per_slice, max(0, len(Xs) - n_train))

            X_train_i, Y_train_i = Xs[:n_train], Ys[:n_train]
            X_val_i,   Y_val_i   = Xs[n_train:n_train+n_val], Ys[n_train:n_train+n_val]
            X_pool_i,  Y_pool_i  = Xs[n_train+n_val:], Ys[n_train+n_val:]

            initial_sizes.append(len(X_train_i))
            self.val_data_dict[i] = (X_val_i, Y_val_i)
            self.add_data_dict[i] = (X_pool_i, Y_pool_i)

            train_blocks_X.append(X_train_i)
            train_blocks_Y.append(Y_train_i)
            val_blocks_X.append(X_val_i)
            val_blocks_Y.append(Y_val_i)

        # Concatenate global train/val
        self.X_train = np.concatenate(train_blocks_X, axis=0) if train_blocks_X else np.empty((0, X_all.shape[1]))
        self.Y_train = np.concatenate(train_blocks_Y, axis=0) if train_blocks_Y else np.empty((0, self.num_label))
        self.X_val   = np.concatenate(val_blocks_X,   axis=0) if val_blocks_X   else np.empty((0, X_all.shape[1]))
        self.Y_val   = np.concatenate(val_blocks_Y,   axis=0) if val_blocks_Y   else np.empty((0, self.num_label))
        self.initial_data_array = np.array(initial_sizes, dtype=int)

        # Derive group indices for fairness from privileged_specs / protected_specs
        self._priv_idx = self._resolve_group_specs(self.privileged_specs)
        self._prot_idx = self._resolve_group_specs(self.protected_specs)

        # Final shuffle of training set (global)
        if len(self.X_train) > 1:
            idx = np.arange(len(self.X_train))
            rng.shuffle(idx)
            self.X_train, self.Y_train = self.X_train[idx], self.Y_train[idx]

        # Quick sanity
        assert self.Y_train.shape[1] == self.num_label
        assert self.Y_val.shape[1] == self.num_label

    def run_baseline(
        self,
        method: str,
        budget: int,
        cost_func: Union[List[float], np.ndarray],
        lr: float = 1e-3,
        epochs: int = 2000,
        num_iter: int = 10,
    ) -> Baseline:
        """
        Construct and run your Baseline with current splits and fairness groups.
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
            privileged_slice_indices=self._priv_idx,
            protected_slice_indices=self._prot_idx
        )
        bl.performance(
            budget=budget,
            cost_func=cost_func,
            num_iter=num_iter,
            batch_size=32,
            lr=lr,
            epochs=epochs
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
    ) -> System_T:
        """
        Construct and run your System_T with current splits and fairness groups.
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
            privileged_slice_indices=self._priv_idx,
            protected_slice_indices=self._prot_idx
        )
        st.selective_collect(
            budget=budget,
            k=k,
            batch_size=32,
            lr=lr,
            epochs=epochs,
            cost_func=cost_func,
            Lambda=Lambda,
            num_iter=num_iter,
            slice_desc=self.slice_desc,
            strategy=strategy,
            show_figure=show_figure
        )
        return st

    # ---------- helpers ----------

    def _resolve_group_specs(self, specs_list: List[Union[str, Tuple[str, str], int]]) -> List[int]:
        """
        Convert a list of group specs (names or indices) into concrete slice indices.
        """
        out = []
        for s in specs_list:
            if isinstance(s, int):
                assert 0 <= s < len(self.slice_specs)
                out.append(int(s))
            else:
                # spec by name/tuple -> find its index in slice_specs
                idx = None
                for i, spec in enumerate(self.slice_specs):
                    if spec == s:
                        idx = i
                        break
                assert idx is not None, f"Group spec {s} not found in slice_specs."
                out.append(idx)
        return out
