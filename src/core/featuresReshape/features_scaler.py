from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def standard_scaler(features):
    transfer = StandardScaler()
    original_columns = features.columns.tolist()
    features_SS = transfer.fit_transform(features)
    transformed_df = pd.DataFrame(features_SS, columns=original_columns)
    return transformed_df,transfer


def min_max_scaler(features):
    transfer = MinMaxScaler()
    features_SS = transfer.fit_transform(features)
    return features_SS
