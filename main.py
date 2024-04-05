# This is a sample Python script.
from src.core.readData.read_data import ReadData
from src.core.featuresReshape.data_reshape import data_reshape, split_features_and_target
from src.core.featuresReshape.features_scaler import standard_scaler
from src.core.featuresReshape.features_select import regression
from src.core.featuresReshape.reduce_dimension import reduce_dimension
from src.utils.train_test_features_targets import train_test_features_targets
from src.core.model.logistic_regression import logistic_rgs
import pandas as pd
import numpy as np
from src.core.model.save_and_load.save_and_load import save_model
import warnings

warnings.filterwarnings("ignore")


def print_hi(name):
    data_L, data_X = ReadData().read_data()
    data_all = data_reshape(data_L, data_X).execute()
    features, target = split_features_and_target(data_all)

    features_SS, scaler = standard_scaler(features)
    # features_SS_slct = regression(features_SS, target).ridge()
    # print(features_SS_slct.shape)
    features_lda, lda = reduce_dimension(1, features_SS).lda(target)
    features_targets_4 = train_test_features_targets(features_lda, target, 0.3)
    print(features_lda.shape)
    score, log = logistic_rgs(features_targets_4)
    save_model(features, target, scaler, lda, log)
    print(score)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
