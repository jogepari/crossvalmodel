## CrossValModel
### About
This module aims to combine cross-validation of **sklearn**-like estimator with voting prediction of trained models into a single class with concise interface.

### Basic usage
The example below demonstrates fitting 5 instances of **LGBMRegressor** on `X_train` (split by **Kfold**), and predicting `X_test` with an average of all trained models' predictions.
```python
from crossvalmodel import CrossValRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

base_model = LGBMRegressor
base_params = dict(n_estimators=3000)
base_fit_params = dict(early_stopping_rounds=50, verbose=100,)

base_estimator = base_model(**base_params)
splitter = sklearn.model_selection.KFold(n_splits=5, shuffle=True)

cvm = CrossValRegressor(base_estimator, splitter)
cvm.fit(X_train, y_train, **base_fit_params)
print(mean_squared_error(*cvm.get_oof_predictions(), squared=False))

cvm_pred = cvm.predict(X_test)
```
Same works for **CrossValClassifier**, predicting with either soft or hard vote.

### Why bother?
While sklearn provides somewhat similar functionality in [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate) with `return_estimator=True` option to preserve models, it's not always convenient. This module allows one line `.predict(X_test)` to naturally vote right away on new data and easily pickle for future inference, but there's also something else.

It's often useful to provide `eval_set = [(X_val, y_val)]` when fitting a compatible model (such as LGBM or [TabNet](https://github.com/dreamquark-ai/tabnet), for example) to utilize early stopping and avoid overfitting. Therefore, during training, each fold's validation part is passed to corresponding model's `.fit()` method (if possible).

Using an ensemble of models trained during cross-validation is a typical part of solving tabular tasks. Usually the process requires a custom loop - not really hard to write or copy from someone else's notebook. But this module aims to essentially narrow it down to **fit**-**predict**.

### Main methods and attributes
 -
    ##### .fit(X, y, *data_args, data_wrapper=None,                   eval_training_set=False, **base_fit_kwargs)
    
  
```
	X, y: array-like, compatible with sklearn-like splitter

  *data_args : array-like, compatible with sklearn-like splitter
      additional fit data parameters, e.g. weights.

  data_wrapper : callable, optional
      applied after splitting to [X, y] + list(data_args)
      e.g. for catboost:
          lambda x, y, w: Pool(x, y, weight=w, cat_features = cat_feats)
      If None (default), models receive data for fitting as
      (X, y, *data_args)

  eval_training_set : bool, optional
      if True, adds train part of each split to eval_set list

  **base_fit_kwargs: kwargs to pass to base_estimator's fit method
      e.g. (verbose=100, plot=True)
```

 - ##### .models
	 List of trained models instances. Can be used, for example, to apply alternative ensembling method, or feature importance exploration.

-
    ##### .get_oof_predictions()

     Returns tuple (oof_true, oof_pred) to compare or pass into sklearn metrics, e.g.:
     `mean_squared_error(*cvm_reg.get_oof_predictions())`,
     where oof_pred is similar to what [cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn-model-selection-cross-val-predict) would return. This may not be suitable depending on metric, as the outputs of different models are bunched together. An alternative is this:
     

 - ##### .oof_res_df
	 **pd.DataFrame** storing out-of-fold results during training. Columns:
	 `[idx_split, model_id, true, pred]`, where model_id can be used to separate splits. In case of classification with probability estimates, `.oof_res_df` also includes **single** `proba` column with either probability of positive class (for binary) or the **highest** probability amongst classes (if multiclass). Full probas are available in a similar **.oof_proba_df** with columns `[idx_split, model_id, pr0, pr1, ...]`

## Development
This project has not fully formed yet, so significant changes in API might happen in the future.

Any contributions are welcome.
