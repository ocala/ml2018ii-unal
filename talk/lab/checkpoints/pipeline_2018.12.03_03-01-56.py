import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Average CV score on the training set was:-9.47696266889e+16
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.9, learning_rate=0.01, loss="quantile", max_depth=1, max_features=0.75, min_samples_leaf=15, min_samples_split=17, n_estimators=100, subsample=0.7)),
        FunctionTransformer(copy)
    ),
    ExtraTreesRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=2, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
