import pandas as pd
from src.core.readData.read_data import ReadData

data1, data2 = ReadData('../../input/XA_total.csv', '../../input/LA_total.csv').read_data()
columns = data1.columns
_str_array = [s.replace('-', '') for s in data1.columns]
columns1 = [s.replace('_', '') for s in _str_array]
tmp = data1.iloc[0:, 0:]

result = pd.DataFrame(data=tmp.values, columns=columns1)  # 将格式变回dataframe，才能写入excel文件
result.index = data1.index  # 转换dataframe index 行索引
xlsx_name = 'XA_total_' + '.xlsx'
xlsx_obj = pd.ExcelWriter(xlsx_name)
result.to_excel(xlsx_obj)
xlsx_obj._save()
print('計算結束')

_str_array = [s.replace('-', '') for s in data2.columns]
columns2 = [s.replace('_', '') for s in _str_array]
tmp = data2.iloc[0:, 0:]

result = pd.DataFrame(data=tmp.values, columns=columns1)  # 将格式变回dataframe，才能写入excel文件
result.index = data2.index  # 转换dataframe index 行索引
xlsx_name = 'LA_total_' + '.xlsx'
xlsx_obj = pd.ExcelWriter(xlsx_name)
result.to_excel(xlsx_obj)
xlsx_obj._save()
print('計算結束')
