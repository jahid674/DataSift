# Regression reference

These files are the unmodified implementation from repository commit `87966fb`.
They are retained only as a differential test oracle, under the repository's
MIT license, and are excluded from the installed package. Do not refactor them.

Tests compare the current implementation with these files using identical
inputs, models, random seeds, and a deterministic clock. This catches changes
in acquired rows, tuple ordering, fairness and accuracy histories, partition
choices, stopping conditions, random-number consumption, gradients, and Hessians.
