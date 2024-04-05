# 集成算法
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import StackingClassifier


# log_clf = LogisticRegression(penalty='l2',max_iter = 10000000,n_jobs=-1)
# svc_linear = SVC(kernel = "linear",max_iter=-1)
# svc_poly = SVC(kernel="poly",degree=3,coef0=0)
# svc_rbf = SVC(kernel="rbf",degree=3,coef0=0)
# tree_clf = DecisionTreeClassifier()

class ensemble:
    def __init__(self, feature_target_4, *models):
        self.model = models
        self.feature_target_4 = feature_target_4
        self.model_estimators = [(chr(97 + i), val) for i, val in enumerate(self.model)]

    def voting(self, voting_type):
        voting_clf = VotingClassifier(estimators=self.model_estimators, voting=voting_type)

        voting_clf.fit(self.feature_target_4.train_features, self.feature_target_4.train_target)
        model_ = self.model
        model_.append(voting_clf)
        for clf in model_:
            clf.fit(self.feature_target_4.train_features, self.feature_target_4.train_target)
            print(clf.__class__.__name__,
                  clf.score(self.feature_target_4.test_features, self.feature_target_4.test_target))
        return voting_clf.score(self.feature_target_4.test_features, self.feature_target_4.test_target), voting_clf

    def stacking(self, final_estimator):
        # log_clf = LogisticRegression(penalty='l2', max_iter=10000000, n_jobs=-1)
        # svc_linear = SVC(kernel="linear", max_iter=-1, probability=True)
        # svc_poly = SVC(kernel="poly", degree=3, coef0=0, probability=True)
        # svc_rbf = SVC(kernel="rbf", degree=3, coef0=0, probability=True) TODO handle probability
        # tree_clf = DecisionTreeClassifier()
        #
        # log_ensemble = LogisticRegression()

        stk_clf = StackingClassifier(estimators=self.model_estimators, final_estimator=final_estimator)

        stk_clf.fit(self.feature_target_4.train_features, self.feature_target_4.train_target)

        score = stk_clf.score(self.feature_target_4.test_features, self.feature_target_4.test_target)
        print(score)
        return score, stk_clf
