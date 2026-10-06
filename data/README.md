# Dataset inputs

The loaders prefer an input in the working directory, then `data/raw/` in a
repository checkout. The Python wheel contains code and dtype configuration;
it does not include datasets.

| Dataset | Files or download | Loader |
| --- | --- | --- |
| Adult | Bundled `raw/adult.data` and `raw/adult.test` | `load_data("adult")` |
| Give Me Some Credit | Download `cs-training.csv` into `raw/` | `load_data("credit")` |
| ACS Income | Folktables downloads Census data | `load_data("income")` |
| ACS Employment | Folktables downloads Census data | `load_data("employment")` |
| ACS Public Coverage | Folktables downloads Census data | `load_data("public")` |
| ACS Mobility | Folktables downloads Census data | `load_data("mobility")` |
| German Credit | Bundled `raw/german.data` | Retained asset; no supported loader |

## Adult

Source: Barry Becker and Ron Kohavi (1996),
[Adult, UCI Machine Learning Repository](https://doi.org/10.24432/C5XW20),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The bundled files are preserved byte-for-byte from commit `87966fb`. They already
omit whitespace around categorical fields and trailing periods on test labels.
The loader retains its original handling of missing values, category mappings,
binary age/hour transformations, column removal, and index resetting. Downloaded
original UCI files need equivalent formatting normalization before this loader
can use them; replacing the bundled files can change experiment inputs.

## Give Me Some Credit

Download from the [Kaggle competition](https://www.kaggle.com/competitions/GiveMeSomeCredit)
and place `cs-training.csv` in `data/raw/`. The loader uses the labeled training
CSV to make its existing 80/20 train/test split with `random_state=42`.
`cs-test.csv` is not used by `load_credit`.

Existing local CSVs are retained on disk but excluded from new commits.
Their earlier inclusion in repository history is not removed by this cleanup.

## American Community Survey

[Folktables](https://github.com/socialfoundations/folktables) downloads the
requested ACS survey through `ACSDataSource`. The default wrapper uses California,
2018, one-year person data. Downloads are cached relative to the working directory
and excluded from new commits. The existing `SEX`/`RAC1P` encoding and split
behavior are retained. For different states or years, call `load_acs_data`
directly.

## German Credit

Source: Hans Hofmann (1994),
[Statlog (German Credit Data), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The bundled file is retained unchanged as a historical research input.
