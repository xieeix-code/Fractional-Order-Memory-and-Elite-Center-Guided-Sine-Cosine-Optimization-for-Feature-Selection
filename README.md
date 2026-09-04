# FOSCA

Official MATLAB implementation of **Fractional-Order Memory and Elite-Center-Guided Sine–Cosine Optimization for Feature Selection**.

FOSCA is a wrapper-based feature-selection method designed for a fixed function-evaluation budget. It keeps the KNN evaluator and binary threshold fixed while improving the internal search dynamics of the sine cosine algorithm through:

- short-memory fractional-order position reconstruction;
- dynamic fitness-weighted elite-center guidance;
- nonlinear search-factor decay; and
- greedy candidate acceptance with exact function-evaluation counting.

Paper: [Fractal and Fractional, 10(8), 526 (2026)](https://doi.org/10.3390/fractalfract10080526)

## Requirements

- MATLAB R2025b (the version used for this implementation)
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox only for the batch experiment in `A_main.m`

## Repository structure

```text
.
├── A_main.m                         # Batch experiment driver
├── FOSCA_FE.m                       # Core FOSCA optimizer
├── FS_FOSCA_wrapper.m               # KNN wrapper and binary conversion
├── 14DataSets/                      # Fourteen classification datasets
└── helper function of main/
    ├── runFSAlgorithm.m             # Algorithm dispatcher
    ├── evaluateSubsetKNN.m          # Held-out test evaluation
    ├── Breakpoint_protection.m      # Checkpoint saving
    └── initialization.m             # Population initialization
```

## Before running

MATLAB requires a file name to match its primary function name. Rename:

```text
FOSCA_FE.m          -> FOSCA2_FE.m
FS_FOSCA_wrapper.m  -> FS_FOSCA2_wrapper.m
```

The names `FOSCA`, `FOSCA-2`, and `FOSCA2` in this repository refer to the same proposed algorithm.

## Quick start

Run the following from the repository root after applying the two file renames above:

```matlab
addpath(genpath(pwd));
rng(1);

T = readtable(fullfile('14DataSets', 'A_Vehicle.xlsx'));
X = T{:, 1:end-1};
Y = T{:, end};
if iscell(Y) || isstring(Y) || ischar(Y)
    Y = categorical(Y);
end

outerCV = cvpartition(Y, 'HoldOut', 0.30);
XTrain = X(training(outerCV), :);
YTrain = Y(training(outerCV));

[bestFitness, bestSubset, curve] = FS_FOSCA2_wrapper( ...
    XTrain, YTrain, 30, 10000, 5, 5);

fprintf('Best fitness: %.6f\n', bestFitness);
fprintf('Selected features: %d / %d\n', sum(bestSubset), numel(bestSubset));
```

Arguments passed to `FS_FOSCA2_wrapper` are:

```text
XTrain, YTrain, population size, maximum FEs, CV folds, KNN neighbors
```

The outputs are the best wrapper fitness, a binary feature mask, and the best-so-far fitness after every function evaluation.

## Fitness function

Each candidate position is thresholded at `0.5` to obtain a feature mask. Empty masks are repaired by selecting the feature with the largest continuous coordinate. Candidate quality is then minimized as

```text
fitness = 0.99 * five-fold KNN misclassification error
        + 0.01 * selected-feature ratio
```

KNN inputs are standardized before classification. Each objective call counts as one function evaluation.

## Reproducing the paper-scale configuration

In `A_main.m`:

1. Change the dataset path to:

   ```matlab
   datasetPath = fullfile(pwd, '14DataSets');
   ```

2. Use the paper settings:

   ```matlab
   numRun  = 30;
   PopSize = 30;
   MaxFE   = 10000;
   kFold   = 5;
   kKNN    = 5;
   algNames = {'FOSCA-2'};
   targetDatasets = {};
   ```

3. Run `A_main.m`.

Each spreadsheet stores samples by row, features in all columns except the last, and the class label in the last column.

The included dispatcher also lists comparison and ablation methods used during development. Their wrapper files are not part of this minimal release; keep `algNames = {'FOSCA-2'}` unless those implementations are added separately.

## Reproducibility notes

- The paper used a fixed 70/30 train/test split, 30 independent optimization runs, population size 30, and 10,000 function evaluations.
- `A_main.m` currently sets the random seed inside each optimization run. Set a seed before the outer `cvpartition` as well if the train/test split must be reproduced exactly.
- The wrapper creates a new five-fold partition at each fitness call. This matches the evaluation regime used for the original comparisons but makes the fitness function stochastic.
- The code uses five neighbors for KNN. This value was not explicitly reported in the paper text.
- Runtime depends strongly on dataset dimensionality because every function evaluation performs five KNN fits.

## Citation

```bibtex
@article{xie2026fosca,
  author  = {Xie, Yuhang and Li, Wei and Qin, Bin and Xu, Kai and Gao, Shang},
  title   = {Fractional-Order Memory and Elite-Center-Guided Sine--Cosine Optimization for Feature Selection},
  journal = {Fractal and Fractional},
  year    = {2026},
  volume  = {10},
  number  = {8},
  pages   = {526},
  doi     = {10.3390/fractalfract10080526}
}
```

If you use this code, please cite the paper above.
