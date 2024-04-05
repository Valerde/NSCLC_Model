# svm
from sklearn.svm import SVC


def svc(feature_target_4, kernel="linear", degree=3, gamma='scale', coef0=0.0):
    svc = SVC(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    svc.fit(feature_target_4.train_features, feature_target_4.train_target)
    test_predict = svc.predict(feature_target_4.test_features)
    print(test_predict)
    score = svc.score(feature_target_4.test_features, feature_target_4.test_target)
    print(score)
    return score, svc
