import datetime
from sklearn.pipeline import Pipeline
from .crossvalmodel import CrossValModel


def estimator_perform(est, Xy_fit, Xy_eval_sets, score_metric, **kwargs):
    start = datetime.datetime.now()
    metric_name = score_metric.__name__
    if isinstance(est, Pipeline):
        final_est = est.steps[-1][1]
        final_est_name = est.steps[-1][0]
        data_prep = est[:-1]
        data_prep.fit(Xy_fit[0])
        Xy_eval_sets = [(data_prep.transform(X), y) for X, y in Xy_eval_sets]
        eval_set_dict = {final_est_name + '__' + 'eval_set': Xy_eval_sets}
    else:
        class DummyTransformer:
            def fit(self):
                return self

            def transform(self, data):
                return data
        final_est = est
        data_prep = DummyTransformer()
        eval_set_dict = {'eval_set': Xy_eval_sets}

    try:
        est.fit(*Xy_fit, **eval_set_dict, **kwargs)
    except TypeError:
        est.fit(*Xy_fit, **kwargs)

    now = datetime.datetime.now()
    fit_time = now - start
    print(f"{now} fit over in {fit_time}")

    if isinstance(final_est, CrossValModel):
        try:
            cv_metric = score_metric(*final_est.get_oof_proba())
            sep_models_scores = [
                [score_metric(y, proba_squeeze(m.predict_proba(X)))
                 for m in final_est.models] for X, y in Xy_eval_sets]
        except (AttributeError, ValueError):
            cv_metric = score_metric(*final_est.get_oof_predictions())
            sep_models_scores = [
                [score_metric(y, m.predict(X))
                 for m in final_est.models] for X, y in Xy_eval_sets]
        print(f'CVM OOF {metric_name}:', cv_metric)
        print(f'CVM val {metric_name} (sep. models):', sep_models_scores)
    else:
        cv_metric, sep_models_scores = [], []

    try:
        score_metrics = [score_metric(y, proba_squeeze(est.predict_proba(X)))
                         for X, y in Xy_eval_sets]
    except (AttributeError, ValueError):
        score_metrics = [score_metric(y, est.predict(X)) for X, y in Xy_eval_sets]

    print(f'Final model {metric_name}:', score_metrics)

    try:
        best_iter = final_est.best_iteration_
    except AttributeError:
        best_iter = '-'

    now = datetime.datetime.now()
    total_time = now - start
    print('Total time:', total_time)

    res = {'est_name': type(est).__name__,
           f'cvm_oof_{metric_name}': cv_metric,
           f'cvm_val_{metric_name}_sep_mod': sep_models_scores,
           f'final_val_{metric_name}': score_metrics,
           'est_params': str(est.get_params()),
           'best_iter': best_iter,
           'fit_time': fit_time,
           'total_time': total_time,
           'X_fit_shape': Xy_fit[0].shape,
           'final_est_features_in': data_prep.transform(Xy_fit[0][:2]).shape[1],
           'X_fit[:2]': Xy_fit[0][:2],
           }

    return res


def est_performance_output(est_perf, show_est_params=True, show_X=False):
    res = '\n\n'.join((':\n'.join(map(str, kv)) for kv in est_perf.items() \
                       if not((kv[0]=='est_params' and not show_est_params) or
                              (kv[0]=='X_fit[:2]' and not show_X))
                       ))
    return res


def proba_squeeze(pr):
    return pr[:, 1] if pr.shape[1] == 2 else pr


def get_catboost_sum_models(models):
    import catboost
    from catboost import sum_models, to_regressor, to_classifier

    models_avrg = sum_models(models,  weights=[1.0/len(models)] * len(models))
    if isinstance(models[0], catboost.CatBoostClassifier):
        return to_classifier(models_avrg)
    else:
        return to_regressor(models_avrg)
