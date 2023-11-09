import os
import numpy as np
import librosa as lb
import pandas as pd
import librosa.display
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split


FEATURES_E = [
    "name", "pitch_mean", "pitch_stddevNorm", "loudness_mean", "loudness_stddevNorm", "spectralFlux_mean", "spectralFlux_stddevNorm",
    "mfcc1_mean", "mfcc1_stddevNorm", "mfcc2_mean", "mfcc2_stddevNorm", "mfcc3_mean", "mfcc3_stddevNorm", "mfcc4_mean", "mfcc4_stddevNorm",
    "jitter_mean", "jitter_stddevNorm", "shimmer_mean", "shimmer_stddevNorm", "HNR_mean", "HNR_stddevNorm", "H1-H2_mean",
    "H1-H2_stddevNorm", "H1-A3_mean", "H1-A3_stddevNorm", "F1frequency_mean", "F1frequency_stddevNorm", "F1bandwidth_mean",
    "F1bandwidth_stddevNorm", "F1amplitude_mean", "F1amplitude_stddevNorm", "F2frequency_mean", "F2frequency_stddevNorm",
    "F2bandwidth_mean", "F2bandwidth_stddevNorm", "F2amplitude_mean", "F2amplitude_stddevNorm", "F3frequency_mean",
    "F3frequency_stddevNorm", "F3bandwidth_mean", "F3bandwidth_stddevNorm", "F3amplitude_mean", "F3amplitude_stddevNorm",
    "alphaRatioV_mean", "alphaRatioV_stddevNorm", "hammarbergIndexV_mean", "hammarbergIndexV_stddevNorm", "slopeV0-500_mean",
    "slopeV0-500_stddevNorm", "slopeV500-1500_mean", "slopeV500-1500_stddevNorm", "spectralFluxV_mean",
    "spectralFluxV_stddevNorm", "mfcc1V_mean", "mfcc1V_stddevNorm", "mfcc2V_mean", "mfcc2V_stddevNorm", "mfcc3V_mean",
    "mfcc3V_stddevNorm", "mfcc4V_mean", "mfcc4V_stddevNorm", "alphaRatioUV_mean", "hammarbergIndexUV_mean", "slopeUV0-500_mean",
    "slopeUV500-1500_mean", "spectralFluxUV_mean", "loudnessPeaksPerSec", "VoicedSegmentsPerSec", "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec", "MeanUnvoicedSegmentLength", "StddevUnvoicedSegmentLength", "equivalentSoundLevel_dBp",
    "depression.symptoms"
]

FEATURES_M = [
    "name", "pitch_mean", "pitch_stddevNorm", "loudness_mean", "loudness_stddevNorm", "jitter_mean", "jitter_stddevNorm",
    "shimmer_mean", "shimmer_stddevNorm", "HNR_mean", "HNR_stddevNorm","H1-H2_mean", "H1-H2_stddevNorm", "H1-A3_mean",
    "H1-A3_stddevNorm", "F1frequency_mean", "F1frequency_stddevNorm", "F1bandwidth_mean", "F1bandwidth_stddevNorm",
    "F1amplitude_mean", "F1amplitude_stddevNorm", "F2frequency_mean", "F2frequency_stddevNorm", "F2amplitude_mean",
    "F2amplitude_stddevNorm", "F3frequency_mean", "F3frequency_stddevNorm", "F3amplitude_mean", "F3amplitude_stddevNorm",
    "alphaRatioV_mean", "alphaRatioV_stddevNorm", "hammarbergIndexV_mean", "hammarbergIndexV_stddevNorm", "slopeV0-500_mean",
    "slopeV0-500_stddevNorm", "slopeV500-1500_mean", "slopeV500-1500_stddevNorm", "alphaRatioUV_mean", "hammarbergIndexUV_mean",
    "slopeUV0-500_mean", "slopeUV500-1500_mean", "loudnessPeaksPerSec", "VoicedSegmentsPerSec", "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec", "MeanUnvoicedSegmentLength", "StddevUnvoicedSegmentLength", "depression.symptoms"
]


class PsychiatricData:
    """ Loading various data set(s) depending on the type of disorder and type of data."""

    def __init__(self,
                 # n_splits: int = 5,
                 n_repeats: int = 10,
                 # to_shuffle: bool = True,
                 data_dir=Path("/Users/aleksandrakoptseva/Desktop/Psychiatric_Disorders-dev/Datasets")
                 ):

        self.data_dir = data_dir
        # self.n_splits = n_splits
        self.n_repeats = n_repeats
        # self.to_shuffle = to_shuffle

        # Variables initialization
        self.ids = list()

        self.features = None  # columns names
        self.x = pd.DataFrame()  # features/random variables (either shuffled or not)
        self.y = pd.DataFrame()  # targets variables/predictions (in correspondence to x)
        self.x_dum_test_pp_df = pd.DataFrame()  # independent/real-world test data

        self.stratified_kFold_cv = None
        self.stratified_train_test_splits = defaultdict(defaultdict)

        # an xlsx file containing the participant IDs, demographic data,
        # symptoms severity of each psychiatric disorder etc.
        self.participants = pd.DataFrame()

        self.depression_only = pd.DataFrame()
        self.schizophrenia_only = pd.DataFrame()
        self.having_both = pd.DataFrame()
        self.control_group = pd.DataFrame()

        # path to audio (.wav) files
        self.audio_files_paths = Path(
            os.path.join(self.data_dir, "wav files")
        )
        # path to participants info file
        self.participants_info_path = Path(
            os.path.join(
                self.data_dir, "PsychiatricDiscourse_participant_data.xlsx"
            )
        )

    def get_onehot_features_targets(self, data, c_features=None, indicators=None, targets=None, type='e'):
        """ Returns x, y, pd.DataFrames, of features and targets values respectively. """
        if c_features:
            data = pd.get_dummies(
                data=data, columns=c_features
            )

        if not indicators:
            indicators = ["ID", ]
        if not targets:
            targets = ["Labels_1", "Labels_2"]
        if type == 'e':
            self.features = [f for f in FEATURES_E]
        elif type == 'm':
            self.features = [f for f in FEATURES_M]

        self.x = data.loc[:, self.features]
        self.y = data.loc[:, targets]

        return self.x, self.y

    def get_stratified_kfold_cv(self, n_splits, to_shuffle):

        """ Returns a CV object to be used in Bayesian/Grid/Random
        search optimization to tune the estimator(s) hyper-parameters.
        """
        self.stratified_kFold_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=to_shuffle,
        )

        return self.stratified_kFold_cv

    def get_stratified_train_test_splits(self, x, y, labels, to_shuffle=True, n_splits=10):
        """ Returns dict containing repeated train and test splits.
                Repeat numbers are separated from the rest of strinds in the key with a single dash "-".
        """

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=to_shuffle
        )

        repeat = 0
        for train_index, test_index in skf.split(x, labels):  # labels := y.Group: to provide correct stratified splits
            repeat += 1
            k = str(repeat)
            self.stratified_train_test_splits[k] = defaultdict(list)
            self.stratified_train_test_splits[k]["x_train"] = x[train_index]
            self.stratified_train_test_splits[k]["x_test"] = x[test_index]
            self.stratified_train_test_splits[k]["y_train"] = y[train_index]
            self.stratified_train_test_splits[k]["y_test"] = y[test_index]

        return self.stratified_train_test_splits

    def get_1d_features_data(self, data_name):
        """ gets 1D_FEATURES of a group (depression, schizophrenia, control) per stimulus or for all"""

        self.participants = self.get_participants_info()

        data_name = data_name.lower()
        stimulus = data_name.split("_")[-1]

        if data_name.split("_")[0] == "d":
            print(f"Depression data {data_name} is being processed.")
            group_info = self.get_depression_only()

        elif data_name.split("_")[0] == "s":
            print(f"Schizophrenia data {data_name} is being processed.")
            group_info = self.get_schizophrenia_only()
            
        elif data_name.split("_")[0] == "c":
            print(f"Control group data {data_name} is being processed.")
            group_info = self.get_control_group()
            
        elif data_name.split("_")[0] == "b":
            print(f"Having both, i.e., depression and schizophrenia data {data_name} is being processed.")
            group_info = self.get_having_both()
        
        else:
            assert False, f"ill-defined data_name{data_name}!"

        # remove the file manually for generating new xlsx file.
        if not os.path.exists(os.path.join(
                self.data_dir, data_name + "_1d_features.xlsx")):

            paths, inconsistencies = self.get_all_files_paths_of_a_group_per_stimulus(
                group_info=group_info, stimulus=stimulus
            )
            inconsistencies_ = ["-".join(i.split("-")[:-1]) for i in inconsistencies]

            audios, sampling_rates = self.get_all_audio_files(paths)
            data = self.extract_1d_voice_features_of_all_audio(
                data_name=data_name, group_info=group_info, audios=audios,
                s_rates=sampling_rates, paths=paths, inconsistencies=inconsistencies_
            )
            data.to_excel(
                os.path.join(
                    self.data_dir, data_name + "_1d_features.xlsx"),
                # index=False,
            )
            return data
        else:
            data = pd.read_excel(
                os.path.join(
                    self.data_dir, data_name + "_1d_features.xlsx"),
                header=0, index_col=0
            )
            return data

    def extract_1d_voice_features_of_all_audio(self, data_name, group_info, audios, s_rates, paths, inconsistencies):

        means, stds, medians, p_ids = [], [], [], []
        for audio, sr, p in zip(audios, s_rates, paths):
            mean, std, median = self.get_1d_voice_features_of_an_audio(audio=audio, sr=sr)
            means.append(mean), stds.append(std), medians.append(median)
            pp = p.parts[-1].split("-")
            p_id = "-".join(pp[:2])
            p_ids.append(p_id)

        means = np.asarray(means)
        stds = np.asarray(stds)
        medians = np.asarray(medians)

        print(
            f"means.shape: {means.shape}, "
            f"stds.shape: {stds.shape}, "
            f"medians.shape: {medians.shape}, "
            f"No. p_ids: {len(p_ids)}, "
            f"group.shape: {group_info.shape}"
        )

        data = pd.DataFrame(p_ids, columns=["ID"])
        print("inconsistencies:\n", inconsistencies)

        group_info = group_info[~group_info["ID"].isin(inconsistencies)]
        print("Group: \n", group_info.head())

        data["sex"] = group_info["sex"].values
        data["age"] = group_info["age"].values
        data["education.level"] = group_info["education.level"].values
        data["education.years"] = group_info["education.years"].values

        if data_name.split("_")[0] == "d":
            data["Labels"] = group_info["depression.symptoms"].values
        elif data_name.split("_")[0] == "s":
            data["Labels"] = group_info["thought.disorder.symptoms"].values
        elif data_name.split("_")[0] == "c":
            data["Labels"] = np.repeat(0, len(group_info))
        else:
            data["Labels_1"] = group_info["depression.symptoms"].values
            data["Labels_2"] = group_info["thought.disorder.symptoms"].values

        for f in range(len(FEATURES)):
            data.insert(loc=f + 1, column=FEATURES[f] + "-ave", value=means[:, f], allow_duplicates=True)
            data.insert(loc=f + 1, column=FEATURES[f] + "-std", value=stds[:, f], allow_duplicates=True)
            data.insert(loc=f + 1, column=FEATURES[f] + "-med", value=medians[:, f], allow_duplicates=True)

        return data

    def get_all_audio_files(self, files_path):
        audios = []
        sampling_rates = []
        for file_path in files_path:
            a, s = self.get_an_audio_file(file_path)
            audios.append(a)
            sampling_rates.append(s)
        return audios, sampling_rates

    def get_all_files_paths_of_a_group_per_stimulus(self, group_info, stimulus):
        files_path = []
        inconsistencies = []
        for p_id in group_info.ID:  # group info := participants_info, p_id := participant_id
            files_path_of_an_id = self.get_files_path_of_an_id(participant_id=p_id)
            try:
                tmp = self.get_the_file_path_of_the_stimulus(files_path_of_an_id, stimulus)
                if tmp:
                    files_path.append(tmp)
                else:
                    print(f"{p_id} has missing audio for {stimulus}")
                    inconsistencies.append(p_id + "-" + stimulus)
            except NameError:
                print(f"Missing audio files for {p_id}")
                inconsistencies.append(p_id)
                
        return files_path, inconsistencies

    def get_all_files_paths_of_a_group_for_all_stimuli(self, group_info):
        files_path = []
        inconsistencies = []
        for p_id in set(group_info.ID):  # group info := participants_info, p_id := participant_id
            try:
                files_path_of_an_id = self.get_files_path_of_an_id(participant_id=p_id)
                # print("files_path_of_an_id: \n", files_path_of_an_id)
                if files_path_of_an_id:
                    files_path += files_path_of_an_id
                    # print("files_path:\n", files_path)
                if len(files_path_of_an_id) < 3:
                    print(f"Missing audio files for {p_id} with some stimuli")
                    inconsistencies.append(p_id)
            except NameError:
                print(f"Missing audio files for {p_id}")
                inconsistencies.append(p_id)

        return files_path, inconsistencies

    def get_files_path_of_an_id(self, participant_id):
        """returns list of paths to all audio files of a participant."""
        return list(self.audio_files_paths.glob(participant_id + "*.wav"))

    def get_participants_info(self):
        self.participants = pd.read_excel(self.participants_info_path)
        return self.participants

    def get_depression_only(self):
        return self.participants.loc[
            (self.participants['thought.disorder.symptoms'] == 0.) &
            (self.participants['depression.symptoms'] != 0.)
            ]

    def get_schizophrenia_only(self):
        return self.participants.loc[
            (self.participants['depression.symptoms'] == 0.) &
            (self.participants['thought.disorder.symptoms'] != 0.)
        ]

    def get_having_both(self):
        return self.participants.loc[
            (self.participants['depression.symptoms'] != 0.) &
            (self.participants['thought.disorder.symptoms'] != 0.)
        ]

    def get_control_group(self):
        return self.participants.loc[
            (self.participants['depression.symptoms'] == 0.) &
            (self.participants['thought.disorder.symptoms'] == 0.)
        ]

    @staticmethod
    def get_the_file_path_of_the_stimulus(files_path_of_id, stimulus):
        """stimulus: str, one the three following items: 'pers', 'pic', 'instr'. """
        for file_path in files_path_of_id:
            if stimulus in file_path.parts[-1].split("-"):
                return file_path

    @staticmethod
    def get_an_audio_file(file_path):
        signal, sr = lb.load(file_path, sr=22050)
        signal, _ = lb.effects.trim(y=signal, top_db=15)
        return signal, sr

    @staticmethod
    def get_1d_voice_features_of_an_audio(audio, sr):

        mean, std, median = [], [], []
        # functions with sampling rate
        features_fn_i = [
            lb.feature.spectral_centroid,
            lb.feature.spectral_bandwidth,
            lb.feature.spectral_rolloff,
        ]

        for fn in features_fn_i:
            res = fn(y=audio, sr=sr)
            mean.append(res.mean())
            std.append(res.std())
            median.append(np.median(res))

        # functions without sampling rate (sr)
        features_fn_ii = [
            lb.feature.spectral_flatness,
            lb.feature.zero_crossing_rate,
            lb.feature.rms,
        ]

        for fn in features_fn_ii:
            res = fn(y=audio,)
            mean.append(res.mean())
            std.append(res.std())
            median.append(np.median(res))

        return mean, std, median

    @staticmethod
    def plot_an_audio_file(signal, sr, title):

        plt.figure(figsize=(20, 5))
        librosa.display.waveshow(signal, sr=sr)
        plt.title('Waveplot ' + title, fontdict=dict(size=18))
        plt.xlabel('Time', fontdict=dict(size=15))
        plt.ylabel('Amplitude', fontdict=dict(size=15))
        plt.show()

        return None






