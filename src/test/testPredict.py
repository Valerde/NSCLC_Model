import numpy as np
import pandas as pd
from src.core.model.save_and_load.save_and_load import load_model
from src.core.featuresReshape.data_reshape import data_reshape
from src.core.featuresReshape.features_scaler import standard_scaler

data = pd.read_excel('../../output/features/cc_nj_67_0_20240406144210.xlsx', index_col=0)

features = data_reshape(None, None).remove_non_num_features(data)
features_name = np.load('../../output/features_name.npy', allow_pickle=True)
# print(features_name)
s = data[data.columns.intersection(features_name)]

model = load_model('../../output/model.pkl')

predict = model.predict(s)
# print(s)
print('预测结果依次为:')
print(int(predict[0]))
