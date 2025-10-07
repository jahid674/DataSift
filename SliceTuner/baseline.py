from __future__ import absolute_import, division, print_function

import copy
import concurrent.futures
import numpy as np

from fc import FC


class Baseline:
    """
    Allocation + training runner.

    Fairness metric (matches your function):
      group_disparity = P_priv(positive) - P_prot(positive)
      where P_group(positive) = (# positive predictions in group) / (# group samples + 1)

    Groups are defined by the *union of slices* you pass via:
      - privileged_slice_indices (e.g., [Male-White, Male-Black])
      - protected_slice_indices  (e.g., [Female-White, Female-Black])
    """

    def __init__(self, train, val, val_data_dict, data_num_array, num_class, num_label,
                 slice_index, add_data_dict, method,
                 favorable_label=1,
                 privileged_slice_indices=None,
                 protected_slice_indices=None):
        self.train = (copy.deepcopy(train[0]), copy.deepcopy(train[1]))
        self.val   = (copy.deepcopy(val[0]),   copy.deepcopy(val[1]))
        self.val_data_dict   = copy.deepcopy(val_data_dict)
        self.data_num_array  = np.array(copy.deepcopy(data_num_array)).astype(int)
        self.add_data_dict   = copy.deepcopy(add_data_dict)
        self.num_class       = int(num_class)
        self.num_label       = int(num_label)
        self.slice_index     = copy.deepcopy(slice_index)
        self.method          = str(method)

        self.favorable_label = int(favorable_label)
        self.privileged_slice_indices = [] if privileged_slice_indices is None else [int(i) for i in privileged_slice_indices]
        self.protected_slice_indices  = [] if protected_slice_indices  is None else [int(i) for i in protected_slice_indices]

        self.budget = None
        self.cost_func = None
        self.batch_size = None
        self.lr = None
        self.epochs = None

    def performance(self, budget, cost_func, num_iter, batch_size, lr, epochs):
        self.budget = float(budget)
        self.cost_func = np.array(cost_func, dtype=float)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.epochs = int(epochs)

        self.loss_output = [0.0] * self.num_class
        self.total_loss = []

        # disparity trackers (per run)
        self.group_disparity_runs = []   # scalar per run
        self.priv_rate_runs = []         # for inspection
        self.prot_rate_runs = []         # for inspection

        # Decide allocation
        if self.method == 'Uniform':
            num_examples = self._allocate_uniform()
        elif self.method == 'Waterfilling':
            num_examples = self._allocate_waterfilling()

        self._train_after_collect_data(num_examples, int(num_iter))

        print(f"Method: {self.method}, Budget: {budget}")
        print("======= Collect Data =======")
        print(num_examples.astype(int))
        print("======= Performance (Group Disparity: prot - priv) =======")
        avg_loss, loss_std, avg_gap, std_gap, avg_priv, std_priv, avg_prot, std_prot = self._show_performance()
        print("Loss: %.5f (%.5f)\n"
              "Group Disparity (prot - priv): %.5f (%.5f)\n"
              "Priv positive rate: %.5f (%.5f)\n"
              "Prot positive rate: %.5f (%.5f)\n" %
              (avg_loss, loss_std, avg_gap, std_gap, avg_priv, std_priv, avg_prot, std_prot))

    def _allocate_uniform(self):
        if self.budget <= 0:
            return np.zeros(self.num_class, dtype=int)
        per_slice_budget = self.budget / float(self.num_class)
        c = np.maximum(self.cost_func, 1e-12)
        return np.floor(per_slice_budget / c).astype(int)

    def _allocate_waterfilling(self):
        if self.budget <= 0:
            return np.zeros(self.num_class, dtype=int)
        current = self.data_num_array.astype(float).copy()
        remaining = float(self.budget)
        added = np.zeros(self.num_class, dtype=int)
        while remaining > 0:
            idx = int(np.argmin(current))
            cost = float(self.cost_func[idx])
            if cost <= 0 or remaining - cost < 0:
                break
            current[idx] += 1.0
            added[idx] += 1
            remaining -= cost
        return added

    def _fc_training(self, process_num):
        net = FC(self.train[0], self.train[1],
                 self.val[0], self.val[1],
                 self.val_data_dict,
                 self.batch_size, epochs=self.epochs, lr=self.lr,
                 num_class=self.num_class, num_label=self.num_label,
                 slice_index=self.slice_index,
                 favorable_label=self.favorable_label)
        return net.fc_train(process_num)

    def _train_after_collect_data(self, num_examples, num_iter):
        self._collect_data(num_examples)

        max_workers = max(1, int(num_iter))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self._fc_training, i) for i in range(num_iter)]
            for job in concurrent.futures.as_completed(futures):
                if job.cancelled():
                    continue
                loss_list, pos_rate_list, pos_count_list, count_list, _ = job.result()

                # aggregate loss
                self.total_loss.append(float(np.average(loss_list)))
                for j in range(self.num_class):
                    self.loss_output[j] += (loss_list[j] / num_iter)

                # compute disparity per run using counts (priv - prot) with "+1" smoothing
                self._measure_disparity_from_counts(pos_count_list, count_list)

    def _collect_data(self, num_examples):
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

        X_tr, y_tr = _shuffle(X_tr, y_tr)
        self.train = (X_tr, y_tr)

    def _measure_disparity_from_counts(self, pos_count_list, count_list):
        pos_count = np.array(pos_count_list, dtype=float)
        counts    = np.array(count_list, dtype=float)

        # indices
        priv_idx = np.array(self.privileged_slice_indices, dtype=int)
        prot_idx = np.array(self.protected_slice_indices, dtype=int)

        if priv_idx.size == 0 or prot_idx.size == 0:
            # If not provided, fall back to "first half = priv, second half = prot" (or do nothing)
            # Here we choose to do nothing but keep interface robust.
            return

        priv_pos = float(np.sum(pos_count[priv_idx]))
        priv_tot = float(np.sum(counts[priv_idx]))
        prot_pos = float(np.sum(pos_count[prot_idx]))
        prot_tot = float(np.sum(counts[prot_idx]))

        # "+1" smoothing in denominators, matching your function
        priv_rate = priv_pos / (priv_tot + 1.0)
        prot_rate = prot_pos / (prot_tot + 1.0)
        gap = prot_rate-priv_rate

        self.group_disparity_runs.append(gap)
        self.priv_rate_runs.append(priv_rate)
        self.prot_rate_runs.append(prot_rate)

    def _show_performance(self):
        final_loss = [float(self.loss_output[i]) for i in range(self.num_class)]
        avg_loss = float(np.average(final_loss)) if len(final_loss) else 0.0
        loss_std = float(np.std(self.total_loss)) if len(self.total_loss) else 0.0

        gaps = np.array(self.group_disparity_runs, dtype=float) if self.group_disparity_runs else np.array([0.0])
        privs = np.array(self.priv_rate_runs, dtype=float) if self.priv_rate_runs else np.array([0.0])
        prots = np.array(self.prot_rate_runs, dtype=float) if self.prot_rate_runs else np.array([0.0])

        return (avg_loss, loss_std,
                float(np.mean(gaps)), float(np.std(gaps)),
                float(np.mean(privs)), float(np.std(privs)),
                float(np.mean(prots)), float(np.std(prots)))
