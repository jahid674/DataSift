# Algorithm contracts and preserved behavior

The public functions in `datasift.algorithms` retain the arguments, return tuple
order, and numerical definitions from commit `87966fb`. The refactor focuses on
repeated work and code organization, without changing acquisition policies.

## Input contract

- Training, validation, test, and pool frames share the same columns and order.
  The target is binary; all model features are numeric.
- Fairness functions interpret evaluation index labels as probability-array
  positions. Reset evaluation indices with `reset_index(drop=True)`.
- Partition methods receive a list of DataFrames. Flat-pool baselines receive
  one DataFrame. Row order is significant for influence-ranked methods.
- The initial model is already trained using `prepare_train_data` on the initial
  training frame. `predict_proba` returns positive-class probabilities of shape
  `(n_samples,)`, unlike sklearn's usual `(n_samples, 2)` output.
- Use positive integer batch sizes and iteration limits, a nonnegative budget,
  and a nonnegative fairness threshold. Metrics require both demographic groups,
  positive ground-truth outcomes in each group, and nonzero probability sums.

`LogisticRegression`, `SVM`, and `NeuralNetwork` support the default PyTorch
training path. Random forests and decision trees can run retraining-based
acquisition, but do not provide parameter gradients for influence computation.
Historical `use_sklearn=True` adapters are retained; the logistic/SVM adapters
have legacy parameter assumptions and are not supported by the pinned runtime.

## Returns

Bandit variants return nine values:

| Position | Value |
| --- | --- |
| 0 | Accepted, expanded training DataFrame |
| 1 | Attempted iteration history, initially `[0]` |
| 2 | Accepted iteration-count history, initially `[0]` |
| 3 | Test fairness for the initial model and accepted batches |
| 4 | Test fairness for the initial model and evaluated batches |
| 5 | Test accuracy for the initial model and accepted batches |
| 6 | Timing history; some variants retain the original `[0]` placeholder |
| 7 | Partition history, with original one-based numbering and repeated entries |
| 8 | Total recorded iteration duration |

Flat-pool variants return seven values:

```text
expanded_train, accepted_iteration_indices, attempted_iteration_indices,
attempted_test_fairness, accepted_test_fairness, attempted_test_accuracy,
total_iteration_duration
```

## Decisions and stopping

Validation data governs acquisition decisions; test data supplies reported
metrics. Each algorithm's original exceptions and policies remain in place:

- `mab_algorithm`, `mab_inf_algorithm`, and `mab_algorithm_base` require a strict
  improvement in validation fairness relative to the best accepted model.
- `mab_inf_algorithm` additionally breaks when accepted **test** fairness meets
  `tau`, before recording the attempt or elapsed duration. This legacy exception
  is preserved, so this variant does use test data for that stopping check.
- Distance and accuracy bandit variants compare a rolling candidate state and
  remove pool points only when a batch is accepted. Other bandit variants remove
  evaluated points even when rejected.
- Flat-pool fairness baselines accept any nonzero fairness change, including
  deterioration. `random_algorithm_acc` accepts any nonzero accuracy change.
- Budgets decrease for accepted batches. Existing rounded batch-count limits,
  partial final batches, iteration indexing, and tie-breaking rules are unchanged.
- Empty partitions are disabled using `-inf` utilities during selection. A
  partition smaller than the requested batch is skipped; the batch is not shrunk
  merely to fit a small partition.

Evaluation feature matrices are scaled once using the initial training data.
Candidate training data is scaled with a newly fitted scaler. This historical
choice is preserved; changing it would change experiment outputs.

The reward update loop changes the current count row sequentially. Each utility
uses the count row as it exists at that point in the loop. Vectorizing this loop
or using a final-row count for every partition would change subsequent choices.
Reward-history reductions also retain their original floating-point order.

## Metrics and dataset limitations

Fairness uses probabilities, not thresholded class labels. Group differences are
protected minus privileged; accuracy uses `p > 0.5`. Partition label disparity
has the opposite sign and adds `1` to each group-count denominator.

Degenerate metric denominators and identical/empty centroid sets retain their
original exceptions or NaN behavior. No new epsilon or zero fallback is inserted.

The existing travel-task identifiers differ between `load_data` (`travel_time`)
and extraction (`travel`). Fairness functions do not implement this task, and
German Credit has no loader. Those behaviors are documented rather than changed.

## Refactor and validation

Centroids are computed once per partition per distance call, replacing repeated
means inside every pairwise comparison. Pandas column alignment and norm/sum
order remain unchanged. Group-disparity calculations use boolean counts instead
of constructing intermediate DataFrames. Distance-based bandit variants build
their neighbor map once per iteration, instead of once for every partition.
Exhausted-partition selection is shared while retaining its original tie and
utility-update behavior.

Differential tests compare all nine acquisition variants across three metrics,
multiple seeds, accepted/rejected batches, budget boundaries, stopping thresholds,
and exhausted partitions. Additional tests compare actual model training,
gradient/Hessian values, dataset preprocessing, and compatibility imports.
The frozen source in `tests/reference/` is the oracle and is not installed.

Recorded wall-clock durations cannot remain identical after a performance
optimization. Tests use an identical synthetic clock to verify the timing
history structure and placement of timing calls. Numerical histories and row
selection are compared exactly within the same runtime.
