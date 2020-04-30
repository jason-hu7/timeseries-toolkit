from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd


# Calculate feature importance.
def forest_feat_importance(
    ts_features: pd.DataFrame,
    ts_target: pd.DataFrame,
    task: str = 'regression',
    plot_show: bool = True
) -> pd.DataFrame:

    assert task == 'regression' or task == 'classification'

    if task == 'regression':
        forest = RandomForestRegressor(n_jobs=-1, verbose=False, random_state=123)
    else:
        forest = RandomForestClassifier(n_jobs=-1, verbose=False, random_state=123)

    gscv = GridSearchCV(cv=5, estimator=forest, verbose=False, n_jobs=-1,
                        param_grid={
                            "n_estimators": [50, 70, 100],
                            "max_depth": [3, 5, 10, 20],
                            "min_samples_split": [30, 50, 100, 200]
                        })
    gscv.fit(ts_features, ts_target)

    if task == 'regression':
        forest = RandomForestRegressor(gscv.best_params_)
    else:
        forest = RandomForestClassifier(gscv.best_params_)
    forest.fit(ts_features, ts_target)

    fi = pd.DataFrame({
        'cols': ts_features.columns,
        'imp': forest.feature_importances_
        }).sort_values('imp', ascending=False)

    if plot_show:
        fi.plot('cols', 'imp', 'barh', figsize=(20, 7), legend=False)

    return fi
