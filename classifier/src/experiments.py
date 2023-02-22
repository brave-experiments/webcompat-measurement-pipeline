import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import (auc, roc_curve)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from src.config import *
from src.util import *
from src.transformers import (DropCorrelatedFeatures, DropMajorityNullFeatures)


def split_features_and_label(df):
    X = df.loc[:, df.columns != LABEL_NAME].to_numpy()
    y = df[LABEL_NAME].to_numpy()

    return X, y


def build_model(clf):
    return Pipeline([
        ('drop_nulls', DropMajorityNullFeatures(threshold=0.66)),
        ('drop_correlates', DropCorrelatedFeatures(threshold=0.8)),
        ('scale', StandardScaler()),
        ('model', clf)
    ])


def get_feature_importance(df, model):
    X_df = df.loc[:, df.columns != LABEL_NAME]
    feature_names = X_df.columns
    if 'feature_selection' in model.named_steps.keys():
        cols = model['feature_selection'].get_support(indices=True)
        feature_names = X_df.iloc[:,cols].columns

    features_df = pd.DataFrame((feature_names, model['model'].feature_importances_)).T
    features_df.columns = ['feature', 'importance']
    return features_df.set_index('feature').sort_values(by='importance', ascending=False)


def get_avg_feature_importance(feature_importances):
    return pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)


def tune(clf, grid_params, X, y, n_inner_folds=3, opt_steps=32):
    model = build_model(clf)

    search = BayesSearchCV(model, grid_params, scoring='average_precision', n_iter=opt_steps, n_points=4, n_jobs=4, cv=n_inner_folds, refit=False, verbose=1)
    search.fit(X, y)

    return search.best_params_, search.best_score_


def train(clf, params, X, y):
    model = build_model(clf)
    model.set_params(**params)
    model.fit(X, y)
    
    return model


def run_experiment(clf, grid_params, df, n_outer_folds=3, n_inner_folds=3, opt_steps=32, save_results=False):
    X, y = split_features_and_label(df)
    print('X.shape', X.shape, 'y.shape', y.shape)

    outer_cv = StratifiedKFold(n_splits=n_outer_folds)

    y_holdout_list, y_proba_list = [], []
    best_params_list, best_score_list = [], []
    feature_importances = []

    start_time = time.time()

    # Outer CV
    for ix_fold, (ix_trainval, ix_holdout) in enumerate(outer_cv.split(X, y)):
        X_trainval, y_trainval = X[ix_trainval], y[ix_trainval]
        X_holdout, y_holdout = X[ix_holdout], y[ix_holdout]
        
        # Inner CV for tuning per outer fold
        best_params, best_score = tune(clf, grid_params, X_trainval, y_trainval, n_inner_folds=n_inner_folds, opt_steps=opt_steps)
        best_params_list.append(best_params)
        best_score_list.append(best_score)
        print('fold', ix_fold, 'best score', best_score, 'best params', best_params)

        # Train on all data w/ best params
        fitted_model = train(clf, best_params, X_trainval, y_trainval)
        feature_importances.append(get_feature_importance(df, fitted_model))

        # Predict hold-out data
        y_proba = fitted_model.predict_proba(X_holdout)[:,1]
        y_holdout_list.append(y_holdout)
        y_proba_list.append(y_proba)

    elapsed_time = time.time() - start_time
    print('elapsed_time', round(elapsed_time/60, 2), 'mins')

    config = {
        'datetime': time.time(),
        'grid_params': grid_params,
        'clf': clf,
        'n_inner_folds': n_inner_folds,
        'n_outer_folds': n_outer_folds,
        'df_shape': df.shape,
    }

    results = {
        'avg_feature_importances_outer_cv': get_avg_feature_importance(feature_importances),
        'best_scores_outer_cv': best_score_list,
        'best_params_outer_cv': best_params_list,
        'y_holdout_list': y_holdout_list,
        'y_proba_list': y_proba_list,
    }

    if save_results:
        save_as_pickle(config, 'config.pickle')
        save_as_pickle(results, 'results.pickle')

    return config, results


def __calc_auc(y_val, y_proba):
    roc_fpr, roc_tpr, _ = roc_curve(y_val, y_proba)
    return auc(roc_fpr, roc_tpr)


def cv_auc_scores(clf, X, y, n_folds=3):
    y_val_list, y_proba_list = [], []

    cv = StratifiedKFold(n_splits=n_folds)
    for ix_train, ix_val in cv.split(X, y):
        X_train, y_train = X[ix_train], y[ix_train]
        X_val, y_val = X[ix_val], y[ix_val]

        fitted_model = train(clf, {}, X_train, y_train)
        y_proba = fitted_model.predict_proba(X_val)[:,1]

        y_val_list.append(y_val)
        y_proba_list.append(y_proba)
        
    return [__calc_auc(y_val, y_proba) for (y_val, y_proba) in tuple(zip(y_val_list, y_proba_list))]


def __compute_baseline_for_feature_importance(clf, df, n_folds):
    X, y = split_features_and_label(df)
    return cv_auc_scores(clf, X, y, n_folds=n_folds)


def __compute_leave_one_out_feature_importance(clf, df, n_folds):
    X, y = split_features_and_label(df)
    return cv_auc_scores(clf, X, y, n_folds=n_folds)


def run_leave_one_feature_out(clf, df, n_folds=5):
    start_time = time.time()

    baseline_scores = __compute_baseline_for_feature_importance(clf, df, n_folds)

    feature_scores = {}
    X_df = df.loc[:, df.columns != LABEL_NAME]
    for feature in tqdm(X_df.columns.to_list()):
        cols = df.columns[df.columns != feature]

        X, y = split_features_and_label(df[cols])

        auc_scores = __compute_leave_one_out_feature_importance(clf, df[cols], n_folds)
        feature_scores = {**feature_scores, **{feature: auc_scores}}

    elapsed_time = time.time() - start_time
    print('elapsed_time', round(elapsed_time/60, 2), 'mins')
    
    return baseline_scores, feature_scores


def run_leave_one_feature_group_out(clf, df, feature_groups, n_folds=3):
    start_time = time.time()

    baseline_scores = __compute_baseline_for_feature_importance(clf, df, n_folds)

    feature_group_scores = {}
    for feature_group in feature_groups:
        feature_group_cols = [col for col in df.columns if feature_group in col]
        cols = df.columns[~df.columns.isin(feature_group_cols)]

        auc_scores = __compute_leave_one_out_feature_importance(clf, df[cols], n_folds)
        feature_group_scores = {**feature_group_scores, **{feature_group: auc_scores}}

    elapsed_time = time.time() - start_time
    print('elapsed_time', round(elapsed_time/60, 2), 'mins')
    
    return baseline_scores, feature_group_scores


def run_sample_complexity(clf, df, t_sizes, n_folds=5):
    X, y = split_features_and_label(df)

    start_time = time.time()

    model = build_model(clf)
    print('t_sizes', t_sizes)
    
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=t_sizes, cv=n_folds, verbose=0, n_jobs=-1)

    elapsed_time = time.time() - start_time

    print('elapsed_time', round(elapsed_time/60, 2), 'mins')

    return train_sizes, train_scores, test_scores
