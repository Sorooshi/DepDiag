import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from types import SimpleNamespace

from psychpy.data.psychiatric_data import PsychiatricData
from psychpy.data.preprocess import preprocess_data


np.set_printoptions(suppress=True, precision=3, linewidth=120)


def args_parser(arguments):

    _pp = arguments.pp.lower()
    _tag = arguments.tag.lower()
    _run = arguments.run
    _data_name = arguments.data_name.lower()
    _estimator_name = arguments.estimator_name.lower()
    _project = arguments.project
    _target_is_org = arguments.target_is_org
    _to_shuffle = arguments.to_shuffle
    _n_clusters = arguments.n_clusters
    _to_exclude_at_risk = arguments.to_exclude_at_risk
    _data_type = arguments.data_type.lower()

    return _pp, _tag, _run, _data_name, _estimator_name, _project, \
        _target_is_org, _to_shuffle, _n_clusters, _to_exclude_at_risk, _data_type


configs = {
    "models_path": Path("/Users/aleksandrakoptseva/Desktop/Psychiatric_Disorders-dev/Psychpy/psychpy/models"),
    "results_path": Path("/Users/aleksandrakoptseva/Desktop/Psychiatric_Disorders-dev/Psychpy/results"),
    "figures_path": Path("/Users/aleksandrakoptseva/Desktop/Psychiatric_Disorders-dev/Psychpy/figures"),
    "params_path": Path("/Users/aleksandrakoptseva/Desktop/Psychiatric_Disorders-dev/Psychpy/params"),
    "data_dir": Path("/Users/aleksandrakoptseva/Desktop/Psychiatric_Disorders-dev/Datasets"),
    "n_repeats": 10,
    "n_splits": 5,
}

configs = SimpleNamespace(**configs)

if not configs.models_path.exists():
    configs.models_path.mkdir()

if not configs.results_path.exists():
    configs.results_path.mkdir()

if not configs.figures_path.exists():
    configs.figures_path.mkdir()

if not configs.params_path.exists():
    configs.params_path.mkdir()

if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project", type=str, default="DC",
        help="Project name for WandB project initialization."
    )

    parser.add_argument(
        "--data_name", type=str, default="s_pers",
        help="data_name: str, consists of two parts XY, and Z separated with an underscore. "
             "The first part (XY) should be an string consists of at most two English alphabets such that each of the "
             "alphabet represents one of the four possible groups: D for depression, S for schizophrenia, C for control"
             " and B for both having depression and schizophrenia." 
             " Duplicated identical letters, e.g. CC or SS are not supported and will stops the program."
             " The second part, Z, can be one of the four following (stimuli) cases:"
             "      \"pic\" or \"intrs\" or \"pers\" or \"all\" for extracting all stimuli together."
             "This argument determines which group of participants with which (sub)set of stimuli are being analysed!"
             "For instance, d_pic, c_instr, s_all, sc_all Etc."
    )

    parser.add_argument(
        "--estimator_name", type=str, default="l_cls",
        help="None case sensitive first letter abbreviated name of an estimator proceeds "
             "  one of the three following suffixes separated with the underscore."
             "  Possible suffixes are: regression := reg, "
             "  classification := cls, clustering := clu"
             "      E.g., Random Forest Regressor := rf_reg, or "
             "      Random Forest Classifiers := rf_cls "
             "Note: First letter of the methods' name should be used for abbreviation."
    )

    parser.add_argument(
        "--run", type=int, default=1,
        help="Run the model or load the saved"
             " weights and reproduce the results."
    )

    parser.add_argument(
        "--pp", type=str, default="mm",
        help="Data preprocessing method:"
             " MinMax/Z-Scoring/etc."
    )

    parser.add_argument(
        "--tag", type=str, default="warmup",
        help="W&B tag will be used to filter some of runs"
             "of the same set of experiments if needed."
    )

    parser.add_argument(
        "--n_clusters", type=int, default=3,
        help="Number of clusters/classes/discrete target values."
    )

    parser.add_argument(
        "--target_is_org", type=int, default=1,
        help="Whether to use not preprocessed target values or not."
    )

    parser.add_argument(
        "--to_shuffle", type=int, default=1,
        help="Whether to shuffle data during CV or not."
             "  Only setting it to one (shuffle=1) will shuffle data."
    )

    parser.add_argument(
        "--to_exclude_at_risk", type=int, default=0,
        help="Whether to exclude at-risk class from experiments or not."
             "  Only setting it to one (to_exclude_at_risk=1) will exclude this class. "
    )

    parser.add_argument(
        "--data_type", type=str, default="1d_features",
        help="Type of data to work with. The following types are supported:"
             " - 1D_features"
             " - audios"
             " - audios_dtw"  # equal length audio file using DTW
             " - spectrograms"
             " - MFCCs"
             " - Chromagrams"
             " - Etc..."
    )

    args = parser.parse_args()

    pp, tag, run, data_name, estimator_name, project, \
        target_is_org, to_shuffle, n_clusters, to_exclude_at_risk, data_type = args_parser(arguments=args)

    print(
        "configuration: \n",
        "  data_name:", data_name, "\n",
        "  data_type:", data_type, "\n",
        "  estimator:", estimator_name, "\n",
        "  shuffle_data:", to_shuffle, "\n",
        "  pre-processing:", pp, "\n",
        "  run:", run, "\n",
        "  n_clusters:", n_clusters, "\n",
    )

    if data_type == "csv":
        print(f" Processing {data_type}")

        psd = PsychiatricData(
            n_repeats=configs.n_repeats,
        )

        group_names = data_name.split("_")[0]
        stimulus = data_name.split("_")[-1]
        df_data_to_use = pd.read_csv(
            "/Users/aleksandrakoptseva/Desktop/Psychiatric_Disorders-dev/Datasets/" + data_name
        )

        print("data: \n", df_data_to_use.head())

        indicators = ["name"]
        targets = ["depression.symptoms"]

        x_org, y_org = psd.get_onehot_features_targets(
            data=df_data_to_use,
            c_features=None,
            indicators=indicators,
            targets=targets,
            type=stimulus  # change to type of features
        )

        x_org = x_org.drop(columns=['name', 'depression.symptoms'])
        x = preprocess_data(x=x_org, pp=pp)  # only x is standardized
        y = y_org["depression.symptoms"].values
        assert not len(targets) > 1, f"more consideration is required to encode the target values {y.shape}!"

        if estimator_name.split("_")[-1] == "cls":
            learning_method = "classification"
            from psychpy.models.classification_estimators import ClassificationEstimators

        elif estimator_name.split("_")[-1] == "reg":
            learning_method = "regression"
            from psychpy.models.regression_estimators import RegressionEstimators
        else:
            assert False, "Undefined estimator and thus undefined target values."

        if to_shuffle == 1:
            to_shuffle = True
            group = learning_method + "-" + "shuffled"
        else:
            to_shuffle = False
            group = learning_method + "-" + "not-shuffled"

        # Adding some details for the sake of clarity in storing and visualization
        configs.run = run
        configs.project = project
        configs.group = group
        configs.tag = tag
        specifier = data_name + "-" + estimator_name + "--shuffled:" + str(to_shuffle)
        configs.specifier = specifier
        configs.data_name = data_name
        configs.name_wb = data_name+": "+specifier
        configs.learning_method = learning_method
        configs.n_clusters = n_clusters

        cv = psd.get_stratified_kfold_cv(
            n_splits=configs.n_splits,
            to_shuffle=to_shuffle,
        )
        data = psd.get_stratified_train_test_splits(
            x=x, y=y,
            labels=y_org[targets].values,
            to_shuffle=to_shuffle,
            n_splits=configs.n_repeats
        )
        print(y)
        # print("data:", data)

        # Classification methods: tune and fit
        if learning_method == "classification" and run == 1:

            cls_est = ClassificationEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            cls_est.instantiate_tuning_estimator_and_parameters()
            cls_est.tune_hyper_parameters()
            cls_est.instantiate_train_test_estimator()
            cls_est.train_test_tuned_estimator()
            cls_est.save_params_results()
            cls_est.print_results()

        # Classification methods: fit with tuned params
        elif learning_method == "classification" and run == 2:

            cls_est = ClassificationEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )
            cls_est.instantiate_train_test_estimator()
            cls_est.train_test_tuned_estimator()
            cls_est.save_params_results()
            cls_est.print_results()

        # Classification methods: print the saved results
        elif learning_method == "classification" and run == 3:

            cls_est = ClassificationEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            cls_est.print_results()
        elif learning_method == "regression" and run == 1:

            reg_est = RegressionEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            reg_est.instantiate_tuning_estimator_and_parameters()
            reg_est.tune_hyper_parameters()
            reg_est.instantiate_train_test_estimator()
            reg_est.train_test_tuned_estimator()
            reg_est.save_params_results()
            reg_est.print_results()

            # Clustering methods: fit with tuned params
        elif learning_method == "regression" and run == 2:

            reg_est = RegressionEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            reg_est.instantiate_train_test_estimator()

            reg_est.train_test_tuned_estimator()

            reg_est.save_params_results()

            reg_est.print_results()
            # Clustering methods: print the saved results
        elif learning_method == "regression" and run == 3:

            reg_est = RegressionEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            reg_est.print_results()
        # Add regression methods for run=1, run=2 and run=3 here:
        # >>>>
        # Here
        # <<<
        # Clustering methods: tune and fit
        elif learning_method == "clustering" and run == 1:

            clu_est = ClusteringEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            clu_est.instantiate_tuning_estimator_and_parameters()
            clu_est.tune_hyper_parameters()
            clu_est.instantiate_fit_test_estimator()
            clu_est.fit_test_tuned_estimator()
            clu_est.save_params_results()
            clu_est.print_results()

        # Clustering methods: fit with tuned params
        elif learning_method == "clustering" and run == 2:

            clu_est = ClusteringEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            clu_est.instantiate_fit_test_estimator()

            clu_est.fit_test_tuned_estimator()

            clu_est.save_params_results()

            clu_est.print_results()
        # Clustering methods: print the saved results
        elif learning_method == "clustering" and run == 3:

            clu_est = ClusteringEstimators(
                x=x, y=y, cv=cv, data=data,
                estimator_name=estimator_name,
                configs=configs,
            )

            clu_est.print_results()
    else:
        assert False, "Unsupported data type!"

    # Adding some details for the sake of clarity in storing and visualization
    # configs.run = run
    # specifier = data_name + "-" + data_type + "-" + estimator_name + "-shuffled:" + str(to_shuffle)
    # configs.specifier = specifier
    # configs.data_name = data_name
    # configs.learning_method = learning_method
    # configs.n_clusters = n_clusters
