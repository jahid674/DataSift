from __future__ import absolute_import, division, print_function

import copy
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import cvxpy as cp

# NEW: sklearn for external evaluation models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fc import FC  # your FC that returns: loss_list, pos_rate_list, pos_count_list, count_list, process_num


class System_T:
    def __init__(self, train, val, val_data_dict, data_num_array, num_class, num_label,
                 slice_index, add_data_dict,
                 favorable_label=1,
                 privileged_slice_indices=None,
                 protected_slice_indices=None):
        """
        Args:
            train: (X_train, y_train_onehot)
            val: (X_val, y_val_onehot)
            val_data_dict: dict/list slice_id -> (X_slice_val, y_slice_val)
            data_num_array: initial per-slice training counts (len = num_class)
            num_class: number of slices
            num_label: number of classes
            slice_index: list of (a,b) feature indices used to define each slice
            add_data_dict: per-slice pools for simulated acquisition
            favorable_label: which class index counts as "positive" (for disparity)
            privileged_slice_indices: list of slice indices belonging to the privileged group
            protected_slice_indices: list of slice indices belonging to the protected group
        """
        self.train = (copy.deepcopy(train[0]), copy.deepcopy(train[1]))
        self.val   = (copy.deepcopy(val[0]),   copy.deepcopy(val[1]))
        self.val_data_dict  = copy.deepcopy(val_data_dict)
        self.data_num_array = np.array(copy.deepcopy(data_num_array)).astype(int)
        self.add_data_dict  = copy.deepcopy(add_data_dict)
        self.num_class  = int(num_class)
        self.num_label  = int(num_label)
        self.slice_index = copy.deepcopy(slice_index)

        # fairness settings (used for final evaluation)
        self.favorable_label = int(favorable_label)
        self.privileged_slice_indices = [] if privileged_slice_indices is None else [int(i) for i in privileged_slice_indices]
        self.protected_slice_indices  = [] if protected_slice_indices  is None else [int(i) for i in protected_slice_indices]

    def selective_collect(self, budget, k, batch_size, lr, epochs, cost_func,
                          Lambda, num_iter, slice_desc, strategy="one-shot", show_figure=False,
                          # NEW: external model options
                          ext_model_type="logreg", ext_C=1.0, ext_kernel="rbf"):
        """
        Selective data collection (Slice Tuner). After collection, trains an external
        Logistic Regression / SVM and reports accuracy + statistical parity (prot - priv).

        Args:
            ...
            ext_model_type: "logreg" or "svm"
            ext_C:          C parameter for the external model
            ext_kernel:     kernel for SVM ("linear", "rbf", ...)
        """
        self.budget = float(budget)
        self.batch_size = int(batch_size)
        self.Lambda = float(Lambda)
        self.cost_func = np.array(cost_func, dtype=float)
        self.lr = float(lr)
        self.epochs = int(epochs)

        # subset sizes for learning-curve fitting (same for all slices)
        initial_k = 100
        num_k_ = initial_k + np.arange(0, k) * (len(self.train[0]) - initial_k) / max(1, (k - 1))
        num_k = [int(i) for i in num_k_]

        iteration = 0
        self.T = 1.0  # imbalance change limit
        self.train_on_subsets(num_k, num_iter)

        IR = self.get_imbalance_ratio(self.data_num_array)
        total_num_examples = np.zeros(self.num_class, dtype=int)

        while True:
            # one-shot allocation from loss curves
            num_examples = self.one_shot(slice_desc, show_figure)

            after_IR = self.get_imbalance_ratio(self.data_num_array + num_examples)

            # imbalance-limited step (if not one-shot)
            if strategy != "one-shot" and abs(after_IR - IR) > self.T:
                target_ratio = IR + self.T * np.sign(after_IR - IR)
                change_ratio = self.get_change_ratio(self.data_num_array, num_examples, target_ratio)
                num_examples = np.array([int(num_examples[i] * change_ratio) for i in range(self.num_class)], dtype=int)
                after_IR = self.get_imbalance_ratio(self.data_num_array + num_examples)

            # spend budget, update sizes
            spent = int(np.sum(np.add(np.multiply(num_examples, self.cost_func), 0.5)))
            self.budget -= spent
            self.data_num_array = (self.data_num_array + num_examples).astype(int)
            self.increase_limit(strategy)
            IR = after_IR
            total_num_examples += num_examples
            iteration += 1

            print("======= Collect Data =======")
            print(num_examples.astype(int))
            print(f"Total Cost: {spent}, Remaining Budget: {self.budget}")

            # move samples from pools into train
            self.collect_data(num_examples)

            # stop when (almost) out of budget: run final evaluation on full current train
            if self.budget < 5:
                self.train_after_collect_data(num_iter)
                print("\n======= Performance (FC summary) =======")
                print(total_num_examples.astype(int))
                print(f"Number of iteration: {iteration}")
                print(f"Strategy: {strategy}, Lambda: {self.Lambda}, Initial Budget: {budget}")

                (avg_loss, loss_std,
                 avg_gap, std_gap,
                 avg_priv, std_priv,
                 avg_prot, std_prot) = self.show_performance()

                print("FC Loss: %.5f (%.5f)\n"
                      "FC Group Disparity (prot - priv): %.5f (%.5f)\n"
                      "FC Priv positive rate: %.5f (%.5f)\n"
                      "FC Prot positive rate: %.5f (%.5f)\n" %
                      (avg_loss, loss_std, avg_gap, std_gap, avg_priv, std_priv, avg_prot, std_prot))

                # === NEW: external model (LogReg / SVM) on collected train ===
                acc, sp_gap = self.evaluate_sklearn(
                    model_type=ext_model_type, C=ext_C, kernel=ext_kernel
                )
                print("\n======= External Model Evaluation =======")
                print(f"Model: {ext_model_type} (C={ext_C}{', kernel='+ext_kernel if ext_model_type=='svm' else ''})")
                print(f"Accuracy (val): {acc:.4f}")
                print(f"Statistical Parity (prot - priv): {sp_gap:.5f}")
                break
            else:
                # re-fit curves on updated train
                num_k_ = initial_k + np.arange(0, k) * (len(self.train[0]) - initial_k) / max(1, (k - 1))
                num_k = [int(i) for i in num_k_]
                self.train_on_subsets(num_k, num_iter)

        return avg_gap  # keep prior return for backward compatibility

    # --------------------------
    # Curve fitting (loss-based)
    # --------------------------
    def train_on_subsets(self, num_k, num_iter):
        """
        Train models on multiple subset sizes to generate points for learning curves.
        We average loss per slice across num_iter runs.
        """
        max_workers = len(num_k)
        self.loss_output, self.slice_num = [], []

        for _ in range(self.num_class):
            self.loss_output.append([0.0] * len(num_k))
            self.slice_num.append([0] * len(num_k))

        for rep in range(num_iter):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                jobs = [executor.submit(self.fc_training, num_k, kk) for kk in range(len(num_k))]
                for job in concurrent.futures.as_completed(jobs):
                    if job.cancelled():
                        continue
                    loss_list, _, _, _, process_num = job.result()
                    for j in range(self.num_class):
                        self.loss_output[j][process_num] += (loss_list[j] / num_iter)
                        if rep == 0:
                            self.slice_num[j][process_num] = int(num_k[process_num])

    def train_after_collect_data(self, num_iter):
        """
        After acquisition completes, evaluate on the full current train num_iter times.
        Collect final loss and group disparity statistics (prot - priv).
        """
        self.total_loss = []
        self.group_disparity_runs = []
        self.priv_rate_runs = []
        self.prot_rate_runs = []

        max_workers = max(1, int(num_iter))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            jobs = [executor.submit(self.fc_training_full, i) for i in range(num_iter)]
            for job in concurrent.futures.as_completed(jobs):
                if job.cancelled():
                    continue
                loss_list, pos_rate_list, pos_count_list, count_list, _ = job.result()
                self.total_loss.append(float(np.average(loss_list)))
                gap, priv_rate, prot_rate = self._disparity_from_counts(pos_count_list, count_list)
                if gap is not None:
                    self.group_disparity_runs.append(gap)
                    self.priv_rate_runs.append(priv_rate)
                    self.prot_rate_runs.append(prot_rate)

        if len(self.total_loss) == 0:
            self.total_loss = [0.0]

    def fc_training(self, num_k, k):
        net = FC(self.train[0][:num_k[k]], self.train[1][:num_k[k]],
                 self.val[0], self.val[1], self.val_data_dict,
                 self.batch_size, epochs=self.epochs, lr=self.lr,
                 num_class=self.num_class, num_label=self.num_label,
                 slice_index=self.slice_index,
                 favorable_label=self.favorable_label)
        return net.fc_train(k)

    def fc_training_full(self, process_num):
        net = FC(self.train[0], self.train[1],
                 self.val[0], self.val[1], self.val_data_dict,
                 self.batch_size, epochs=self.epochs, lr=self.lr,
                 num_class=self.num_class, num_label=self.num_label,
                 slice_index=self.slice_index,
                 favorable_label=self.favorable_label)
        return net.fc_train(process_num)

    # --------------------------
    # Acquisition helpers
    # --------------------------
    def collect_data(self, num_examples):
        def _shuffle(d, l):
            idx = np.arange(len(d))
            np.random.shuffle(idx)
            return d[idx], l[idx]

        X_tr, y_tr = self.train
        for i in range(self.num_class):
            k = int(num_examples[i])
            if k <= 0:
                continue
            X_pool_i, y_pool_i = self.add_data_dict[i]
            k = min(k, len(X_pool_i))
            if k <= 0:
                continue
            X_tr = np.concatenate([X_tr, X_pool_i[:k]], axis=0)
            y_tr = np.concatenate([y_tr, y_pool_i[:k]], axis=0)
            self.add_data_dict[i] = (X_pool_i[k:], y_pool_i[k:])
        self.train = _shuffle(X_tr, y_tr)

    def one_shot(self, slice_desc, show_figure):
        A, B, estimate_loss = self.fit_learning_curve(slice_desc, show_figure)
        return self.op_func(A, B, self.data_num_array, estimate_loss)

    def fit_learning_curve(self, slice_desc, show_figure):
        def weight_list(weight):
            return [1.0 / (float(w) ** 0.5 if w > 0 else 1.0) for w in weight]

        def power_law(x, a, b):
            return b * np.power(x, -a)

        A, B, estimate_loss = [], [], []
        for i in range(self.num_class):
            xdata_dense = np.linspace(self.slice_num[i][0], self.slice_num[i][-1], 1000)
            sigma = weight_list(self.slice_num[i])
            popt, _ = curve_fit(power_law,
                                xdata=np.array(self.slice_num[i], dtype=float),
                                ydata=np.array(self.loss_output[i], dtype=float),
                                sigma=np.array(sigma, dtype=float),
                                absolute_sigma=True,
                                bounds=(0, [np.inf, np.inf]),
                                maxfev=5000)
            a_hat, b_hat = popt[0], popt[1]
            A.append(-a_hat)  # sign convention as in your original
            B.append(b_hat)
            estimate_loss.append(b_hat * (float(self.data_num_array[i]) ** (-a_hat)))

            if show_figure:
                plt.figure(1, figsize=(12, 8))
                plt.plot(self.slice_num[i], self.loss_output[i], 'o-', linewidth=1.0, markersize=4, label=slice_desc[i])
                plt.plot(xdata_dense, power_law(xdata_dense, *popt), linewidth=2.0,
                         label=r"$y={%.3f}x^{-%.3f}$" % (b_hat, a_hat))
                plt.tick_params(labelsize=20)
                plt.xlabel('Number of training examples', fontsize=25)
                plt.ylabel('Validation Loss', fontsize=25)
                plt.legend(prop={'size': 20})
                plt.tight_layout()
                plt.show()

        return A, B, estimate_loss

    def op_func(self, A, B, N, estimate_loss):
        try:
            x = cp.Variable(self.num_class)
            ob_func = 0
            for i in range(self.num_class):
                loss = cp.multiply(B[i], cp.power((x[i] + N[i]), A[i]))
                counter_loss = cp.sum(estimate_loss) / float(self.num_class)
                ob_func += loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)

            constraints = [cp.sum(cp.multiply(x, self.cost_func)) == self.budget, x >= 0]
            prob = cp.Problem(cp.Minimize(ob_func), constraints)
            prob.solve(solver="ECOS_BB")
        except Exception:
            x = cp.Variable(self.num_class, integer=True)
            ob_func = 0
            for i in range(self.num_class):
                loss = cp.multiply(B[i], cp.power((x[i] + N[i]), A[i]))
                counter_loss = cp.sum(estimate_loss) / float(self.num_class)
                ob_func += loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)

            constraints = [cp.sum(cp.multiply(x, self.cost_func)) == self.budget, x >= 0]
            prob = cp.Problem(cp.Minimize(ob_func), constraints)
            prob.solve(solver="ECOS_BB")

        return np.add(x.value, 0.5).astype(int)

    # --------------------------
    # Imbalance controls
    # --------------------------
    def increase_limit(self, strategy):
        if strategy == "Aggressive":
            self.T *= 2.0
        elif strategy == "Moderate":
            self.T += 1.0
        else:
            self.T = self.T

    def get_imbalance_ratio(self, data_array):
        data_array = np.array(data_array, dtype=float)
        return float(np.max(data_array) / max(1.0, np.min(data_array)))

    def get_change_ratio(self, data_array, num_examples, target_ratio):
        def F(x, num, add, target):
            new_sizes = [int(add[i] * x) + int(num[i]) for i in range(self.num_class)]
            return max(new_sizes) - target * min(new_sizes)

        ratio = scipy.optimize.fsolve(F, x0=0.5, args=(self.data_num_array, num_examples, target_ratio))
        if ratio > 1:
            ratio = scipy.optimize.fsolve(F, x0=0.25, args=(self.data_num_array, num_examples, target_ratio))
        elif ratio < 0:
            ratio = scipy.optimize.fsolve(F, x0=0.75, args=(self.data_num_array, num_examples, target_ratio))
        return float(ratio)

    # --------------------------
    # Disparity computation
    # --------------------------
    def _disparity_from_counts(self, pos_count_list, count_list):
        if len(self.privileged_slice_indices) == 0 or len(self.protected_slice_indices) == 0:
            return None, None, None

        pos = np.array(pos_count_list, dtype=float)
        tot = np.array(count_list, dtype=float)

        priv_pos = float(np.sum(pos[self.privileged_slice_indices]))
        priv_tot = float(np.sum(tot[self.privileged_slice_indices]))
        prot_pos = float(np.sum(pos[self.protected_slice_indices]))
        prot_tot = float(np.sum(tot[self.protected_slice_indices]))

        priv_rate = priv_pos / (priv_tot + 1.0)
        prot_rate = prot_pos / (prot_tot + 1.0)
        gap = prot_rate - priv_rate
        return gap, priv_rate, prot_rate

    def show_performance(self):
        loss_arr = np.array(self.total_loss, dtype=float) if len(self.total_loss) else np.array([0.0])
        gap_arr  = np.array(self.group_disparity_runs, dtype=float) if len(self.group_disparity_runs) else np.array([0.0])
        priv_arr = np.array(self.priv_rate_runs, dtype=float) if len(self.priv_rate_runs) else np.array([0.0])
        prot_arr = np.array(self.prot_rate_runs, dtype=float) if len(self.prot_rate_runs) else np.array([0.0])

        return (float(np.mean(loss_arr)), float(np.std(loss_arr)),
                float(np.mean(gap_arr)),  float(np.std(gap_arr)),
                float(np.mean(priv_arr)), float(np.std(priv_arr)),
                float(np.mean(prot_arr)), float(np.std(prot_arr)))

    # ==========================
    # NEW: External model eval
    # ==========================
    def evaluate_sklearn(self, model_type="logreg", C=1.0, kernel="rbf"):
        """
        Train a Logistic Regression or SVM on the CURRENT collected training set
        and evaluate on the global validation set + per-slice validation dict.

        Returns:
            accuracy (float), statistical_parity (prot - priv) (float)
        """
        Xtr = self.train[0]
        ytr = np.argmax(self.train[1], axis=1)  # back to labels

        Xv  = self.val[0]
        yv  = np.argmax(self.val[1], axis=1)

        # Model
        if model_type.lower() == "logreg":
            clf = make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(C=float(C), max_iter=1000, n_jobs=None)
            )
        elif model_type.lower() == "svm":
            clf = make_pipeline(
                StandardScaler(with_mean=False),
                SVC(C=float(C), kernel=kernel, probability=False)
            )
        else:
            raise ValueError("model_type must be 'logreg' or 'svm'")

        # Fit
        clf.fit(Xtr, ytr)

        # Accuracy on global validation
        yhat = clf.predict(Xv)
        acc = float(np.mean(yhat == yv))

        # Statistical parity on validation (prot - priv) using slice dicts
        # Count predicted positives per group with +1 smoothing
        priv_pos = 0
        priv_tot = 0
        prot_pos = 0
        prot_tot = 0

        for i in range(self.num_class):
            Xi, yi_onehot = self.val_data_dict[i]
            if len(Xi) == 0:
                continue
            ypred_i = clf.predict(Xi)
            pos_i = int(np.sum(ypred_i == self.favorable_label))
            n_i   = int(len(Xi))

            if i in self.privileged_slice_indices:
                priv_pos += pos_i
                priv_tot += n_i
            if i in self.protected_slice_indices:
                prot_pos += pos_i
                prot_tot += n_i

        priv_rate = priv_pos / (priv_tot + 1.0)
        prot_rate = prot_pos / (prot_tot + 1.0)
        sp_gap = float(prot_rate - priv_rate)

        return acc, sp_gap
