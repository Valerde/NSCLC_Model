# 邏輯回歸
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model import LogisticRegression


def logistic_rgs(feature_target_4):
    log = LogisticRegression(penalty='l2', max_iter=10000000, n_jobs=-1)

    log.fit(feature_target_4.train_features, feature_target_4.train_target)
    test_predict = log.predict(feature_target_4.test_features)
    print(test_predict)
    logic_score = log.score(feature_target_4.test_features, feature_target_4.test_target)
    return logic_score, log
