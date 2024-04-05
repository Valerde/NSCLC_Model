import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV

from sklearn.feature_selection import SelectFromModel


class regression:
    def __init__(self, features_SS, target):
        self.features_SS = features_SS
        self.target = target

    def lasso(self):
        alphas_ = np.logspace(-2, 0, 300)

        lassocv = LassoCV(alphas=alphas_, cv=10, max_iter=100000).fit(self.features_SS, self.target)

        # features = features[self.features_SS.columns[lassocv.coef_ != 0]]

        features_SS = self.features_SS[:, pd.DataFrame(self.features_SS).columns[lassocv.coef_ != 0]]

        return features_SS

    def ridge(self):
        alphas_ = np.logspace(-2, 0, 300)
        ridge_cv = RidgeCV(alphas=alphas_)

        # 使用 SelectFromModel 进行特征选择
        sfm = SelectFromModel(ridge_cv)
        sfm.fit(self.features_SS, self.target)

        # 获取选择的特征索引
        selected_features_idx = sfm.get_support(indices=True)
        features_SS = pd.DataFrame(self.features_SS).iloc[:, selected_features_idx]

        return features_SS
