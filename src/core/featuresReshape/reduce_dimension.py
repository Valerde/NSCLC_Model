import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

from core.featuresReshape.data_reshape import data_reshape


class reduce_dimension:
    def __init__(self, n_components, features_SS):
        self.n_components = n_components
        self.features_SS = features_SS

    def pca(self):
        pca = PCA(n_components=self.n_components)
        features_reduced = pca.fit_transform(self.features_SS)
        return features_reduced, pca

    def kernel_pca(self, kernel="linear"):
        pca = KernelPCA(n_components=self.n_components, kernel=kernel)
        features_reduced = pca.fit_transform(self.features_SS)
        return features_reduced, pca

    def lda(self, target):
        if min(target.nunique() - 1, self.features_SS.shape[1]) < self.n_components:
            print('warning: n_components cannot be larger than min(n_features, n_classes - 1)\n' +
                  'already set n_component largest')
            self.n_components = min(target.nunique() - 1, self.features_SS.shape[1])
        lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        features_lda = lda.fit_transform(self.features_SS, target)
        np.save('output/features_name.npy', lda.feature_names_in_)
        return features_lda, lda
