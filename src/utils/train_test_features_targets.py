from sklearn.model_selection import train_test_split


class train_test_features_targets:
    def __init__(self, features_all, target, test_size):
        self.train_features, self.test_features, \
        self.train_target, self.test_target = train_test_split(features_all,
                                                               target,
                                                               test_size=test_size)

