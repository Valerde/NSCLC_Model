import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import datetime
from datetime import datetime

import sys

image_path = sys.argv[1]
label_path = sys.argv[2]


def ExtractFeatures(image_path, label_path, extractor, file_name):
    index = []  # 用于存储标签作为index
    patient_info = image_path.split('/')[-1].split('.')[0]
    index.append(patient_info)
    image = sitk.ReadImage(image_path)  # %读取图片
    label = sitk.ReadImage(label_path)  # %读取图片对应的label
    features = extractor.execute(image, label)  # 特征提取
    features_dict = features.items()  # 将特征转换成字典
    features_df = pd.DataFrame(list(features_dict)).T  # 将字典转化成dataFrame对象,并转置
    features_df.columns = features_df.iloc[0, :]  # 对df对象做处理，主要是表格格式化
    _str_array = [s.replace('-', '') for s in features_df.columns]
    features_df.columns = [s.replace('_', '') for s in _str_array]
    features_df.drop(index=[0], inplace=True)  # 同上

    result = pd.DataFrame(data=features_df.iloc[0:, 0:], columns=features_df.columns)  # 将格式变回dataframe，才能写入excel文件
    result.index = [patient_info]  # 转换dataframe index 行索引
    formatted_time = datetime.now().strftime("%Y%m%d%H%M%S")
    xlsx_name = file_name + patient_info + '_' + formatted_time + '.xlsx'
    xlsx_obj = pd.ExcelWriter(xlsx_name)
    result.to_excel(xlsx_obj)
    xlsx_obj._save()
    return xlsx_name


def executor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableAllFeatures()
    result = ExtractFeatures(image_path, label_path, extractor=extractor, file_name='./output/features/')
    return result
