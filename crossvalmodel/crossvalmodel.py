import pandas as pd
import numpy as np
import sklearn
from tqdm.auto import tqdm
import copy
import datetime
import scipy
import inspect


__all__ = [
    'CrossValModel',
    'CrossValRegressor',
    'CrossValClassifier',
]


class CrossValModel:
    """
    Cross-validation wrapper preserving trained models for prediction.

    Base class, to be sublassed.
    Use CrossValRegressor and CrossValClassifier instead.
    """

    def __init__(self, base_estimator, cv_split, verbosity):
        self.base_estimator = copy.deepcopy(base_estimator)
        self.cv_split = cv_split
        self.verbosity = verbosity
        self._init_attributes()

    def _init_attributes(self):
        self.is_fit = False
        self.models = []
        self.oof_res_df = pd.DataFrame()
        self.oof_proba_df = pd.DataFrame()
        self.best_iteration_ = None

    def fit(self, X, y, *data_args, data_wrapper=None,
            eval_training_set=False, **base_fit_kwargs):
        """
        Cross-validate: fit several models on data according to splits.

        Parameters
        ----------
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

        Returns
        -------
        model: CrossValRegressor or CrossValClassifier
        """
        self._init_attributes()

        # delete ouside eval set because it will be made by cv_split
        base_fit_kwargs.pop('eval_set', None)
        self._alert('base_estimator fitting kwargs:', base_fit_kwargs)

        try:
            cvm_splits = self.cv_split.split(X, y)
            n_splits = self.cv_split.get_n_splits()
        except AttributeError:
            cvm_splits = self.cv_split
            n_splits = len(cvm_splits)

        fit_signature = inspect.signature(self.base_estimator.fit)
        provide_eval_set = 'eval_set' in fit_signature.parameters

        data = [X, y] + list(data_args)

        for model_id, (train_ids, val_ids) in enumerate(tqdm(cvm_splits, total=n_splits)):
            self._alert(f'\n{datetime.datetime.now()} Fold {model_id}, getting train and val sets')

            # pandas/numpy indexing
            data_tr, data_val = [], []
            for d in data:
                d_tr, d_v = (d.iloc[train_ids], d.iloc[val_ids]) if \
                            isinstance(d, pd.core.generic.NDFrame) else \
                            (d[train_ids], d[val_ids])
                data_tr.append(d_tr)
                data_val.append(d_v)

            (X_tr, _), (X_v, y_v) = data_tr[:2], data_val[:2]
            self._alert('train and val shapes:', X_tr.shape, X_v.shape)

            if data_wrapper is not None:
                data_tr = data_wrapper(*data_tr)
                data_val = data_wrapper(*data_val)
            else:
                data_tr, data_val = map(tuple, (data_tr, data_val))

            self._fit_single_split(
                model_id, data_tr, data_val, val_ids, X_v, y_v,
                provide_eval_set, eval_training_set,
                **base_fit_kwargs)

        self.oof_res_df.sort_values(by='idx_split',
                                       ignore_index=True, inplace=True)
        self.oof_proba_df.sort_values(by='idx_split',
                                          ignore_index=True, inplace=True)

        try:
            self.best_iteration_ = np.mean([m.best_iteration_ for m in self.models])
        except AttributeError:
            pass
        self.is_fit = True

        return self

    def _fit_single_split(self, model_id, data_tr, data_val, val_ids, X_v, y_v,
                          provide_eval_set, eval_training_set,
                          **base_fit_kwargs):
        est = copy.deepcopy(self.base_estimator)
        self._alert(datetime.datetime.now(), 'fitting')

        fold_fit_kwargs = base_fit_kwargs.copy()
        if provide_eval_set:
            eval_set = [data_tr, data_val] if eval_training_set else [data_val]
            fold_fit_kwargs['eval_set'] = eval_set

        if isinstance(data_tr, (tuple, list)):
            data_shapes = [d.shape if hasattr(d, 'shape') else '???'
                           for d in data_tr]
            self._alert(f'fit tuple of len: {len(data_tr)}, shapes:',
                        *data_shapes)
            est.fit(*data_tr, **fold_fit_kwargs)
        else:
            self._alert(f'fit {type(data_tr)}')
            est.fit(data_tr, **fold_fit_kwargs)

        self._alert(datetime.datetime.now(), 'fit over')

        self.models.append(est)

        fold_res = pd.DataFrame(data={'idx_split': val_ids})
        for data_obj in (y_v, X_v):
            if isinstance(data_obj, pd.core.generic.NDFrame):
                fold_res['idx_orig'] = data_obj.index
                break
        fold_res = fold_res.assign(model_id=model_id,
                                   true=np.array(y_v))
        fold_probas = fold_res.loc[:, :'true']

        try:
            # classification with probability
            y_v_proba = est.predict_proba(X_v)
            fold_res['pred'] = self.models[0].classes_[np.argmax(y_v_proba, axis=-1)]
            if y_v_proba.shape[1] <= 2:
                fold_res['proba'] = y_v_proba[:, -1]
            else:
                fold_res['proba'] = y_v_proba.max(axis=-1)

            tmp_probas_df = pd.DataFrame(
                data=y_v_proba,
                columns=['pr_' + str(ci) for ci in range(y_v_proba.shape[-1])],
                index=fold_res.index,
                )
            fold_probas = pd.concat((fold_probas, tmp_probas_df), axis=1)

            self._alert(datetime.datetime.now(), 'proba over')
        except AttributeError:
            # regression and classification w/o probability
            y_v_pred = est.predict(X_v)
            fold_res['pred'] = np.array(y_v_pred)
            self._alert(datetime.datetime.now(), 'predict over')

        if self.oof_res_df.empty:
            self.oof_res_df.reindex(columns=fold_res.columns)
            self.oof_proba_df.reindex(columns=fold_probas.columns)
        self.oof_res_df = self.oof_res_df.append(fold_res, ignore_index=True)
        self.oof_proba_df = self.oof_proba_df.append(fold_probas, ignore_index=True)

    def _alert(self, *message, alert_level=1, **kwargs):
        if self.verbosity >= alert_level:
            print(*message, **kwargs)

    def get_oof_predictions(self):
        """
        Get OOF probabilities for metric calculation.

        Returns
        -------
        Tuple (oof_true, oof_pred) to pass into sklearn metrics, e.g.:

        mean_squared_error(*cvm_reg.get_oof_predictions())
        """
        return (self.oof_res_df['true'], self.oof_res_df['pred'])

    def get_params(self, **kwargs):
        try:
            return self.base_estimator.get_params(**kwargs)
        except AttributeError:
            self._alert('base_estimator has no "get_params" method')

    def set_params(self, **params):
        self.base_estimator.set_params(**params)


class CrossValRegressor(CrossValModel):
    def __init__(self, base_estimator, cv_split, verbosity=0):
        """
        Cross-validation wrapper preserving trained regressors for prediction.

        Parameters
        ----------
        base_estimator : model with sklearn-like API

        cv_split : either sklearn-like splitter (e.g. KFold())
                   or iterable of indices
        verbosity : bool or int
            0 - silent, 1 and above - debugging alerts
        """
        super().__init__(base_estimator, cv_split, verbosity)
        self.__name__ = 'CrossValRegressor'

    def predict(self, X):
        """
        Predict regression for X: simple mean of each model's predicton.

        Parameters
        ----------
        X : array-like, same features as X passed to fit method.

        Returns
        -------
        y: ndarray
            Predicted values - np.mean of predictions by all models.
        """
        if not self.is_fit:
            raise sklearn.exceptions.NotFittedError()
        all_models_pred = [model.predict(X) for model in self.models]
        return np.stack(all_models_pred, axis=-1).mean(axis=-1, keepdims=False)


class CrossValClassifier(CrossValModel):
    def __init__(self, base_estimator, cv_split, verbosity=0):
        """
        Cross-validation wrapper preserving trained classifiers for prediction.

        Parameters
        ----------
        base_estimator : model with sklearn-like API

        cv_split : either sklearn-like splitter (e.g. KFold())
        or iterable of indices

        verbosity : bool or int
            0 - silent, 1 and above - debugging alerts
        """
        super().__init__(base_estimator, cv_split, verbosity)
        self.__name__ = 'CrossValClassifier'

    def predict_proba(self, X):
        """
        Predict class probabilities for X: mean of each model's predict_proba.

        Parameters
        ----------
        X : array-like, same features as X passed to fit method.

        Returns
        -------
        p: ndarray
            Predicted values - np.mean of predictions by all models.
        """
        if not self.is_fit:
            raise sklearn.exceptions.NotFittedError()
        all_models_proba = [model.predict_proba(X) for model in self.models]
        return np.stack(all_models_proba, axis=-1).mean(axis=-1)

    def predict(self, X, calc_probas=True):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like, same features as X passed to fit method.

        calc_probas : bool, optional
            If True, predicts class with the largest average probability output
            by all models.
            If False, just takes mode of all predictions.

        Returns
        -------
         y: ndarray
            Predicted class.
        """
        if not self.is_fit:
            raise sklearn.exceptions.NotFittedError()

        if calc_probas:
            probas = self.predict_proba(X)
            # probably might work wrong if models get different label sets
            return self.models[0].classes_[np.argmax(probas, axis=-1)]
        else:
            all_models_preds = [model.predict(X) for model in self.models]
            return scipy.stats.mode(np.stack(all_models_preds, axis=-1), axis=1)[0]

    def get_oof_proba(self, squeeze_binary=True):
        """
        Get OOF probabilities for metric calculation.

        Parameters
        ----------
        squeeze_binary : bool, optional
            For binary classification, return proba just for positive class.

        Returns
        -------
        Tuple (oof_true, oof_proba) to pass into sklearn metrics, e.g.:

        roc_auc_score(*cvm_clf.get_oof_proba())
        """
        oof_proba = self.oof_proba_df.loc[:, 'pr_0':]
        if oof_proba.shape[1] == 2 and squeeze_binary:
            oof_proba = oof_proba['pr_1']
        return self.oof_proba_df['true'], oof_proba
