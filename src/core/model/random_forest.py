# randomForest
from sklearn.ensemble import RandomForestClassifier


def random_forest(feature_target_4):
    random_forest_clf = RandomForestClassifier(n_estimators=20)
    model_rf = random_forest_clf.fit(feature_target_4.train_features,
                                     feature_target_4.train_target)
    score_rf = model_rf.score(feature_target_4.test_features, feature_target_4.test_target)
    print(score_rf)
    predict = model_rf.predict(feature_target_4.test_features)
    print(predict)
    return score_rf, model_rf
