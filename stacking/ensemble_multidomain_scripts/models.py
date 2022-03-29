
import pickle
from pathlib import Path
from logging import getLogger

import joblib

import sklearn.linear_model
import sklearn.svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
import lightgbm
import torch
import optuna

from data_util import normalize_score, inverse_normalize_score

logger = getLogger(__name__)

K_CV = 5
HP_OPT_TRIALS = 100
HP_OPT_SEED = 0
LinearSVR_OPT_MAX_ITER = 1000
KernelSVR_OPT_MAX_ITER = 100000

class Ridge:

    def __init__(self, params=None):

        if params is None:
            # use hyperparameters tuned by w2v_small
            self.params =  {
                'alpha': 36.315622558743634,
            }

        else:
            self.params = params

    def train(self, train_X, train_y, val_X, val_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        val_X_sc = X_scaler.transform(val_X)

        ridge = sklearn.linear_model.Ridge(**(self.params))

        ridge.fit(train_X_sc, train_y_sc)

        pred_y = inverse_normalize_score(ridge.predict(val_X_sc))

        val_mse = mean_squared_error(val_y, pred_y)

        logger.info('Val MSE: {:f}'.format(val_mse))

        self.ridge = ridge
        self.X_scaler = X_scaler


    def predict(self, X, df):
        """
        Calculate results and insert them in pd.DataDrame columns
        """
        X_sc = self.X_scaler.transform(X)

        pred_y = inverse_normalize_score(self.ridge.predict(X_sc))

        df['pred_mos'] = pred_y.ravel()

        return df

    def save_model(self, out_dir: Path):

        joblib.dump(self.ridge, out_dir / 'model.joblib')
        joblib.dump(self.X_scaler, out_dir / 'X_scaler.joblib')

    def load_model(self, model_dir: Path):

        self.ridge = joblib.load(model_dir / 'model.joblib')
        self.X_scaler = joblib.load(model_dir / 'X_scaler.joblib')

    def optimize_hp(self, train_X, train_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        param_distributions = {
            'alpha': optuna.distributions.LogUniformDistribution(1e-5, 1e+5)
        }

        model = sklearn.linear_model.Ridge()
        scoring = make_scorer(mean_squared_error, greater_is_better=False)

        optuna_search = optuna.integration.OptunaSearchCV(model, param_distributions,
                                                            cv=K_CV,
                                                            n_trials=HP_OPT_TRIALS,
                                                            random_state=HP_OPT_SEED,
                                                            scoring=scoring,
                                                            verbose=0)

        optuna_search.fit(train_X_sc, train_y_sc)

        return optuna_search.best_params_


class LinearSVR:

    def __init__(self, params=None, stage='stage1'):

        if params is None:
            # use hyperparameters tuned by w2v_small
            self.params =  {
                'C': 0.01982058833734277,
                'epsilon': 0.23072531432463972,
            }

        else:
            self.params = params

        self.max_iter = 10000
        if stage == 'stage1':
            self.opt_max_iter = self.max_iter
        else:
            self.opt_max_iter = LinearSVR_OPT_MAX_ITER

    def train(self, train_X, train_y, val_X, val_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        val_X_sc = X_scaler.transform(val_X)

        svr = sklearn.svm.LinearSVR(max_iter=self.max_iter, **self.params)

        svr.fit(train_X_sc, train_y_sc)

        pred_y = inverse_normalize_score(svr.predict(val_X_sc))

        val_mse = mean_squared_error(val_y, pred_y)

        logger.info('Val MSE: {:f}'.format(val_mse))

        self.svr = svr
        self.X_scaler = X_scaler


    def predict(self, X, df):
        """
        Calculate results and insert them in pd.DataDrame columns
        """
        X_sc = self.X_scaler.transform(X)

        pred_y = inverse_normalize_score(self.svr.predict(X_sc))

        df['pred_mos'] = pred_y.ravel()

        return df

    def save_model(self, out_dir: Path):

        joblib.dump(self.svr, out_dir / 'model.joblib')
        joblib.dump(self.X_scaler, out_dir / 'X_scaler.joblib')

    def load_model(self, model_dir: Path):

        self.svr = joblib.load(model_dir / 'model.joblib')
        self.X_scaler = joblib.load(model_dir / 'X_scaler.joblib')

    def optimize_hp(self, train_X, train_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        param_distributions = {
            'C': optuna.distributions.LogUniformDistribution(1e-5, 1e+5),
            'epsilon': optuna.distributions.UniformDistribution(0, 1),
        }

        model = sklearn.svm.LinearSVR(max_iter=self.opt_max_iter)
        scoring = make_scorer(mean_squared_error, greater_is_better=False)

        optuna_search = optuna.integration.OptunaSearchCV(model, param_distributions,
                                                            cv=K_CV,
                                                            n_trials=HP_OPT_TRIALS,
                                                            random_state=HP_OPT_SEED,
                                                            scoring=scoring,
                                                            verbose=0)

        optuna_search.fit(train_X_sc, train_y_sc)

        return optuna_search.best_params_


class KernelSVR:

    def __init__(self, params=None, stage='stage1'):

        if params is None:
            # use hyperparameters tuned by w2v_small
            self.params =  {
                'C': 4.483354499092266,
                'epsilon': 0.2177054099781604,
                'gamma': 0.0006981540829311363,
            }

        else:
            self.params = params

        # self.max_iter = 10000
        if stage == 'stage1':
            self.opt_max_iter = -1
        else:
            self.opt_max_iter = KernelSVR_OPT_MAX_ITER


    def train(self, train_X, train_y, val_X, val_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        val_X_sc = X_scaler.transform(val_X)

        svr = sklearn.svm.SVR(kernel='rbf', **self.params)

        svr.fit(train_X_sc, train_y_sc)

        pred_y = inverse_normalize_score(svr.predict(val_X_sc))

        val_mse = mean_squared_error(val_y, pred_y)

        logger.info('Val MSE: {:f}'.format(val_mse))

        self.svr = svr
        self.X_scaler = X_scaler


    def predict(self, X, df):
        """
        Calculate results and insert them in pd.DataDrame columns
        """
        X_sc = self.X_scaler.transform(X)

        pred_y = inverse_normalize_score(self.svr.predict(X_sc))

        df['pred_mos'] = pred_y.ravel()

        return df

    def save_model(self, out_dir: Path):

        joblib.dump(self.svr, out_dir / 'model.joblib')
        joblib.dump(self.X_scaler, out_dir / 'X_scaler.joblib')

    def load_model(self, model_dir: Path):

        self.svr = joblib.load(model_dir / 'model.joblib')
        self.X_scaler = joblib.load(model_dir / 'X_scaler.joblib')



    def optimize_hp(self, train_X, train_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        param_distributions = {
            'C': optuna.distributions.LogUniformDistribution(1e-5, 1e+5),
            'epsilon': optuna.distributions.UniformDistribution(0, 1),
            'gamma': optuna.distributions.LogUniformDistribution(1e-5, 1e+5),
        }

        model = sklearn.svm.SVR(max_iter=self.opt_max_iter)
        scoring = make_scorer(mean_squared_error, greater_is_better=False)

        optuna_search = optuna.integration.OptunaSearchCV(model, param_distributions,
                                                            cv=K_CV,
                                                            n_trials=HP_OPT_TRIALS,
                                                            random_state=HP_OPT_SEED,
                                                            scoring=scoring,
                                                            verbose=0)

        optuna_search.fit(train_X_sc, train_y_sc)

        return optuna_search.best_params_


class RandomForest:

    def __init__(self, params=None):

        if params is None:
            self.params =  {
                'n_estimators': 100,
                'max_depth': 1000,
            }

        else:
            self.params = params

    def train(self, train_X, train_y, val_X, val_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        val_X_sc = X_scaler.transform(val_X)

        rf = sklearn.ensemble.RandomForestRegressor(**self.params)

        rf.fit(train_X_sc, train_y_sc)

        pred_y = inverse_normalize_score(rf.predict(val_X_sc))

        val_mse = mean_squared_error(val_y, pred_y)

        logger.info('Val MSE: {:f}'.format(val_mse))

        self.rf = rf
        self.X_scaler = X_scaler


    def predict(self, X, df):
        """
        Calculate results and insert them in pd.DataDrame columns
        """
        X_sc = self.X_scaler.transform(X)

        pred_y = inverse_normalize_score(self.rf.predict(X_sc))

        df['pred_mos'] = pred_y.ravel()

        return df

    def save_model(self, out_dir: Path):

        joblib.dump(self.rf, out_dir / 'model.joblib')
        joblib.dump(self.X_scaler, out_dir / 'X_scaler.joblib')

    def load_model(self, model_dir: Path):

        self.rf = joblib.load(model_dir / 'model.joblib')
        self.X_scaler = joblib.load(model_dir / 'X_scaler.joblib')


    def optimize_hp(self, train_X, train_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        param_distributions = {
            'n_estimators': optuna.distributions.IntUniformDistribution(1, 100),
            'max_depth': optuna.distributions.IntLogUniformDistribution(1, 1000),
            'max_features': optuna.distributions.CategoricalDistribution(['auto', 'sqrt', 'log2']),
        }

        model = sklearn.ensemble.RandomForestRegressor()
        scoring = make_scorer(mean_squared_error, greater_is_better=False)

        optuna_search = optuna.integration.OptunaSearchCV(model, param_distributions,
                                                            cv=K_CV,
                                                            n_trials=HP_OPT_TRIALS,
                                                            random_state=HP_OPT_SEED,
                                                            scoring=scoring,
                                                            verbose=0)

        optuna_search.fit(train_X_sc, train_y_sc)

        return optuna_search.best_params_


class LightGBM:

    def __init__(self, params=None):

        if params is None:
            # use hyperparameters tuned by w2v_small
            self.params =  {
                'lambda_l1': 0.03708013547428929,
                'lambda_l2': 3.1884740170707856e-07,
                'num_leaves': 220,
                'feature_fraction': 0.6747205882024254,
                'bagging_fraction': 0.9367956222111139,
                'bagging_freq': 2,
                'min_child_samples': 92,
                'max_depth': 10
            }

        else:
            self.params = params

    def train(self, train_X, train_y, val_X, val_y):

        train_set = lightgbm.Dataset(train_X, train_y)
        valid_set = lightgbm.Dataset(val_X, val_y, reference=train_set)

        lgb_model = lightgbm.train(
            params = self.params,
            train_set = train_set,
            valid_sets = [train_set, valid_set],
            num_boost_round = 10000,
            early_stopping_rounds = 10,
        )

        pred_y = lgb_model.predict(val_X, num_iteration=lgb_model.best_iteration)

        val_mse = mean_squared_error(val_y, pred_y)

        logger.info('Val MSE: {:f}'.format(val_mse))

        self.lgb_model = lgb_model


    def predict(self, X, df):
        """
        Calculate results and insert them in pd.DataDrame columns
        """
        pred_y = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)

        df['pred_mos'] = pred_y.ravel()

        return df

    def save_model(self, out_dir: Path):

        with open(out_dir / 'model.pkl', 'wb') as f:
            pickle.dump(self.lgb_model, f)


    def load_model(self, model_dir: Path):

        self.lgb_model = pickle.load(open(model_dir / 'model.pkl', 'rb'))


    def optimize_hp(self, train_X, train_y):

        X_scaler = StandardScaler()
        train_X_sc = X_scaler.fit_transform(train_X)
        train_y_sc = normalize_score(train_y)

        lgb_train = optuna.integration.lightgbm.Dataset(train_X_sc, train_y_sc)

        lgbm_params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
        }        

        folds = sklearn.model_selection.KFold(n_splits=K_CV, shuffle=True, random_state=HP_OPT_SEED)

        tuner_cv = optuna.integration.lightgbm.LightGBMTunerCV(
            lgbm_params, lgb_train,
            num_boost_round=1000,
            early_stopping_rounds=100,
            # verbose_eval=20,
            folds=folds,
            optuna_seed=HP_OPT_SEED,
        )

        tuner_cv.run()

        return tuner_cv.best_params
