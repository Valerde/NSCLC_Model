from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, RepeatedKFold
import pandas as pd


def repeated_kfold(model, all_features, target, n_splits=10, n_repeats=10):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    sum_score = 0

    for train_index, test_index in rkf.split(all_features):
        X_train = pd.DataFrame(all_features).iloc[train_index]
        X_test = pd.DataFrame(all_features).iloc[test_index]
        # print(train_index)
        # print(test_index)
        y_train = pd.DataFrame(target).iloc[train_index]
        y_test = pd.DataFrame(target).iloc[test_index]
        model_svm = model.fit(X_train, y_train)
        score_svm = model_svm.score(X_test, y_test)
        sum_score += score_svm

    print(sum_score / 100)
    return sum_score
