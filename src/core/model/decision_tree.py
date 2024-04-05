from sklearn.tree import DecisionTreeClassifier


def decision_tree(feature_target_4):
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(feature_target_4.train_features, feature_target_4.train_target)
    test_predict = tree_clf.predict(feature_target_4.test_features)
    print(test_predict)
    tree_score = tree_clf.score(feature_target_4.test_features, feature_target_4.test_target)
    return tree_score, tree_clf
