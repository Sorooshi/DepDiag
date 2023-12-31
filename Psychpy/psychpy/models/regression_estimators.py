import time
import pickle
import numpy as np
import xgboost as xgb
from sklearn.svm import SVR
from skopt import BayesSearchCV
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import psychpy.common.utils as util
import plotly.express as px
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, \
    ExpSineSquared, ConstantKernel, RBF


class RegressionEstimators:

    def __init__(self, x, y, cv, data, estimator_name, configs, ):
        self.x = x  # np.ndarray, a pre-processed matrix of features/random variables.
        self.y = y  # np.ndarray, not pre-processed vector of target variables.
        self.cv = cv  # CV sklearn instance, stratified KFolds Cross_Validation generator with/without shuffles.
        self.data = data  # Dict of dicts, containing repeated train and test splits, (x, y np arrays).
        self.estimator_name = estimator_name.lower()  # str, name of estimator to select the method.
        self.configs = configs  # configuration dict, as namespace, to pass storing path, etc.

        self.estimator = None
        self.tuning_estimator = None
        self.params = defaultdict()
        if self.configs.run == 1:
            self.tuned_params = defaultdict()
        else:
            self.tuned_params = self.load_saved_tuned_params()
            print(
                "tuned params:\n",
                self.tuned_params,
                "\n"
            )

        self.results = defaultdict(defaultdict)

    def instantiate_tuning_estimator_and_parameters(self, ):

        # Simplest learning method(s):
        if self.estimator_name == "l_reg":
            self.tuning_estimator = LinearRegression()

            # define search space
            self.params = defaultdict()
            self.params["fit_intercept"] = Categorical([True, False])

            print(
                "Linear Regressor."
            )

        # Support Vector machine method(s):
        elif self.estimator_name == "sv_reg":
            self.tuning_estimator = SVR()

            # define search space
            self.params = defaultdict()
            self.params["kernel"] = Categorical(["linear", "poly", "rbf", "sigmoid", ])
            self.params['degree'] = Integer(1, 3)
            self.params['C'] = Real(1e-1, 4.0, 'log-uniform')
            self.params['gamma'] = Real(1e-1, 2.0, 'log-uniform')
            self.params["epsilon"] = Real(1e-1, 2.0, 'log-uniform')

            print(
                "C-Support Vector Regression."
            )

        # KNN method(s):
        elif self.estimator_name == "knn_reg":
            self.tuning_estimator = KNeighborsRegressor()

            # define search space
            self.params = defaultdict()
            self.params["n_neighbors"] = Integer(1, 10, )
            self.params["p"] = Real(1, 5, "uniform")

            print(
                "KNearest Neighbor Regressor."
            )

        # Bayesian Ridge:
        elif self.estimator_name == "br_reg":
            self.tuning_estimator = BayesianRidge()

            # define search space
            self.params = defaultdict()
            self.params["n_iter"] = Integer(1e2, 2e4, "uniform")
            self.params["alpha_1"] = Real(1e-8, 1e-2, "uniform")
            self.params["alpha_2"] = Real(1e-8, 1e-2, "uniform")
            self.params["lambda_1"] = Real(1e-8, 1e-2, "uniform")
            self.params["lambda_2"] = Real(1e-8, 1e-2, "uniform")
            self.params["fit_intercept"] = Categorical([True, False])

            print(
                "Instantiate Bayesian Ridge Regressor."
            )

        # Ensemble learning method(s):
        elif self.estimator_name == "rf_reg":
            self.tuning_estimator = RandomForestRegressor(verbose=0, )

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )

            print(
                "Random Forest Regressor."
            )

        elif self.estimator_name == "gb_reg":
            self.tuning_estimator = GradientBoostingRegressor(verbose=0, )

            # define search space
            self.params = defaultdict()
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )
            self.params["alpha"] = Real(1e-1, 9e-1, "uniform")

            print(
                "Gradient Boosting Regressor."
            )

        elif self.estimator_name == "ab_reg":
            self.tuning_estimator = AdaBoostRegressor()

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            print(
                "Adaboost Regressor."
            )

        elif self.estimator_name == "xgb_reg":
            self.tuning_estimator = xgb.XGBRegressor()

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            self.params["max_depth"] = Integer(1, 100, "uniform")
            print(
                "XGBoost Regressor."
            )

        # Gaussian Process method(s):
        elif self.estimator_name == "_gp_reg_":
            self.tuning_estimator = GaussianProcessRegressor()
            # Previously we faced some issue due to limits of
            #   GP due the dataset size, and thus for now I won't consider it.

            kernel_rbf = RBF(
                length_scale=1.0, length_scale_bounds=(1e-1, 10.0)
            )

            kernel_rational_quadratic = RationalQuadratic(
                length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15)
            )

            kernel_exp_sin_squared = ExpSineSquared(
                length_scale=1.0,
                periodicity=3.0,
                length_scale_bounds=(0.1, 10.0),
                periodicity_bounds=(1.0, 10.0),
            )

            kernel = ConstantKernel(
                constant_value=1.0, constant_value_bounds=(0.0, 10.0)
            ) * RBF(
                length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(
                length_scale=2.0, length_scale_bounds=(0.0, 10.0))

            # define search space
            self.params = defaultdict()
            self.params["kernel"] = Categorical([kernel, None])  # how to define?
            # Categorical([kernel_rbf, kernel_rational_quadratic, kernel_exp_sin_squared])
            print(
                "Gaussian Process Regressor."
            )

        # Neural Networks method(s):
        elif self.estimator_name == "mlp_reg":
            self.tuning_estimator = MLPRegressor(
                shuffle=False,
                verbose=False,
            )

            # define search space
            self.params = defaultdict()
            self.params["hidden_layer_sizes"] = (2, 200,)
            self.params["activation"] = Categorical(["logistic", "tanh", "relu"])
            self.params["solver"] = Categorical(["lbfgs", "sgd", "adam"])
            # self.params["alpha"] = Real(1e-6, 1e-2, "uniform")
            # self.params["learning_rate"] = Categorical(["constant", "invscaling", "adaptive"])
            # self.params["learning_rate_init"] = Real(1e-4, 1e-1, "uniform")
            self.params["max_iter"] = Integer(100, 50000, "uniform")

            print(
                "Multi Layer Perceptron Regressor."
            )

        else:
            assert False, "Undefined regression model."

        return None  # self.tuning_estimator, self.params

    def instantiate_train_test_estimator(self, ):

        # Simplest learning method(s):
        if self.estimator_name == "l_reg":
            self.estimator = LinearRegression(**self.tuned_params)
            print(
                "Instantiate Linear Regressor."
            )

        # Support Vector machine method(s):
        elif self.estimator_name == "sv_reg":
            self.estimator = SVR(**self.tuned_params)
            print(
                "Instantiate C-Support Vector Regression."
            )

        # KNN method(s):
        elif self.estimator_name == "knn_reg":
            self.estimator = KNeighborsRegressor(**self.tuned_params)
            print(
                "Instantiate KNearest Neighbor Regressor."
            )

        # Bayesian Ridge:
        elif self.estimator_name == "br_reg":
            self.estimator = BayesianRidge(**self.tuned_params)
            print(
                "Instantiate Bayesian Ridge Regressor."
            )

        # Ensemble learning method(s):
        elif self.estimator_name == "rf_reg":
            self.estimator = RandomForestRegressor(**self.tuned_params)

            print(
                "Instantiate Random Forest Regressor."
            )

        elif self.estimator_name == "gb_reg":
            self.estimator = GradientBoostingRegressor(**self.tuned_params)
            print(
                "Instantiate Gradient Boosting Regressor."
            )

        elif self.estimator_name == "ab_reg":  # does not support 2d y
            self.estimator = AdaBoostRegressor(**self.tuned_params)
            print(
                "Instantiate Adaboost Regressor."
            )

        elif self.estimator_name == "xgb_reg":
            self.estimator = xgb.XGBRegressor(**self.tuned_params)
            print(
                "Instantiate XGBoost Regressor."
            )

        # Gaussian Process method(s):
        elif self.estimator_name == "gp_reg":
            self.estimator = GaussianProcessRegressor(**self.tuned_params)
            # Previously we faced some issue due to limits of
            #   GP due dataset size, and thus for now I won't consider it
            print(
                "Instantiate Gaussian Process Regressor."
            )

        # Neural Networks method(s):
        elif self.estimator_name == "mlp_reg":
            self.estimator = MLPRegressor(**self.tuned_params)
            print(
                "Instantiate Multi-Layer Perceptron Regressor."
            )

        else:
            assert False, "Undefined regression model."

        return None  # self.estimator

    def tune_hyper_parameters(self, ):
        """ estimator sklearn estimator, estimator dict of parameters. """

        print("CV hyper-parameters tuning for " + self.estimator_name)

        # define the search
        search = BayesSearchCV(
            estimator=self.tuning_estimator,
            search_spaces=self.params,
            n_jobs=1, cv=self.cv,
            scoring="r2",
            optimizer_kwargs={'base_estimator': 'GP'},
            verbose=1,
        )

        # perform the search
        search.fit(X=self.x, y=self.y, )

        # report the best result
        print("best score:", search.best_score_)
        print("best params:", search.best_params_)
        self.tuned_params = search.best_params_

        return None  # self.tuned_params, self.estimator

    def train_test_tuned_estimator(self,):

        """ returns of dict of dicts, containing y_test and y_pred per each repeat. """

        print(
            "Training and testing of " + self.estimator_name
        )

        old_score = - np.inf

        for k, v in self.data.items():

            self.results[k] = defaultdict()

            start = time.time()
            self.estimator.fit(v["x_train"], v["y_train"])
            y_test = v["y_test"]
            x_test = v["x_test"]
            y_pred = self.estimator.predict(x_test)
            y_pred_prob = None
            end = time.time()

            print(
                "k:", k,
                "x_train:", v["x_train"].shape,
                "y_train:", v["y_train"].shape,
                "x_test:", x_test.shape,
                "y_test:", y_test.shape,
            )
            self.results[k]["y_test"] = y_test
            self.results[k]["x_test"] = x_test
            self.results[k]["y_pred"] = y_pred
            self.results[k]["y_pred_prob"] = y_pred_prob
            self.results[k]["exe_time"] = end - start

            # to save the best results model and plots
            score = r2_score(y_test, y_pred)
            print(
                f"repeat{k}, score {score} \n"
                f"y_true: {y_test} \n"
                f"y_pred: {y_pred}"
            )

            if score > old_score:
                old_score = score

                _ = util.save_model(
                    path=self.configs.models_path,
                    model=self.estimator,
                    specifier=self.configs.specifier,
                )

            # run.finish()

        return None  # self.results

    def save_params_results(self,):
        # save tuned_params
        util.save_a_dict(
            a_dict=self.tuned_params,
            name=self.configs.specifier,
            save_path=self.configs.params_path,
        )

        # save results
        util.save_a_dict(
            a_dict=self.results,
            name=self.configs.specifier,
            save_path=self.configs.results_path,
        )

        return None

    def load_saved_tuned_params(self,):
        saved_tuned_params = util.load_a_dict(
            name=self.configs.specifier,
            save_path=self.configs.params_path
        )
        return saved_tuned_params

    def print_results(self, ):

        results = util.load_a_dict(
            name=self.configs.specifier,
            save_path=self.configs.results_path,
        )

        util.print_the_evaluated_results(
            results,
            self.configs.learning_method,
        )

        return None
