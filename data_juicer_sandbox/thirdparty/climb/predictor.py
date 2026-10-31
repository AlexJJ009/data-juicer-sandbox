# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is adapted from the RegMix project:
# https://github.com/sail-sg/regmix/blob/main/regression_fitting/regression.ipynb

"""
For predictor training, we use a LightGBM regression model, which fits mixture-performance pairs well with limited data.
To prevent overfitting, we set L1 and L2 regularization, early stopping, a maximum depth of four, and require at least five samples per leaf.
Additionally, we employed a separate validation set and an early stopping mechanism, halting training after 20 rounds of no improvement.
"""

import lightgbm as lgb
import pandas as pd

SEED = 42


def fit_predictor(df: pd.DataFrame, domain_names: list[str], target_column: str) -> lgb.LGBMRegressor:
    # Shuffle the DataFrame
    shuffled_df = df.sample(frac=1, random_state=SEED)

    # Split the DataFrame into train and test sets
    split_idx = int(len(shuffled_df) * 0.9)
    if split_idx == 0 or split_idx == len(shuffled_df):
        msg = (
            f"fit_predictor needs at least 2 rows to produce non-empty 90/10 train/test splits, "
            f"got {len(shuffled_df)}."
        )
        raise ValueError(msg)
    train_df = shuffled_df.iloc[:split_idx]
    test_df = shuffled_df.iloc[split_idx:]

    # Cap stopping_rounds at len(test_df) so early stopping stays proportional to the eval set
    # Nemotron-CLIMB algorithm starts with 64 proxy model evaluations -> test=7, far below the original stopping_rounds=20
    stopping_rounds = min(20, len(test_df))

    x_train = train_df[domain_names].to_numpy()
    y_train = train_df[target_column].to_numpy()
    x_test = test_df[domain_names].to_numpy()
    y_test = test_df[target_column].to_numpy()

    # Train the predictor
    hyper_params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": ["l1", "l2"],
        "num_iterations": 1000,
        "seed": 42,
        "learning_rate": 1e-2,
        "verbosity": -1,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "max_depth": 4,
        "min_child_samples": 5,
    }

    gbm = lgb.LGBMRegressor(**hyper_params)

    return gbm.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
        ],
    )

