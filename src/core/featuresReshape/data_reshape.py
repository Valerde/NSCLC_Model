import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene
from sklearn.utils import shuffle


class data_reshape:

    def __init__(self, data_L, data_X):
        self.data_L = data_L
        self.data_X = data_X

    def remove_non_num_features(self, data):
        data_ret = pd.DataFrame()
        columns = data.columns
        for col in columns:
            try:
                df = data[col].astype(np.float64)
                data_ret = pd.concat([data_ret, df], axis=1)
            except:
                pass
            continue
        return data_ret

    def t_test(self, data_1, data_2):
        # 方差齐性
        index_ = []
        for col in data_1.columns:
            if levene(data_1[col], data_2[col])[1] > 0.05:
                if ttest_ind(data_1[col], data_2[col])[1] < 0.05:
                    index_.append(col)
            else:
                if ttest_ind(data_1[col], data_2[col], equal_var=False)[1] < 0.05:
                    index_.append(col)

        data_1_T = data_1[index_]
        data_2_T = data_2[index_]
        return data_1_T, data_2_T

    '''
    經過此處理,數據為亂序的,可轉換為數值的
    '''

    def data_concat(self, data_1, data_2):
        data = pd.concat([data_1, data_2])
        data = shuffle(data)
        return data

    def execute(self):
        features_L = self.remove_non_num_features(self.data_L)
        features_X = self.remove_non_num_features(self.data_X)
        features_L, features_X = self.t_test(features_L, features_X)
        return self.data_concat(features_L, features_X)

#python 不支持方法重載
    # def execute(self, data_1, data_2):
    #     features_L = self.remove_non_num_features(data_1)
    #     features_X = self.remove_non_num_features(data_2)
    #     features_L, features_X = self.t_test(features_L, features_X)
    #     return self.data_concat(features_L, features_X)


def split_features_and_target(data):
    target = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    return features, target
