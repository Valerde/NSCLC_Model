import matplotlib.pyplot as plt
import six
import os  # needed navigate the system to get the input data
import numpy as np
import radiomics
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # This module is used for interaction with pyradiomics

# from radiomics import getTestCase
# import csv
# %matplotlib inline
# import matplotlib.pyplot as plt


# settings = {}
# settings['binWidth'] = 25  # 5
# settings['sigma'] = [3, 5]
# settings['Interpolator'] = sitk.sitkBSpline
# settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
# settings['voxelArrayShift'] = 1000  # 300
# settings['normalize'] = True
# settings['normalizeScale'] = 100
# extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor = featureextractor.RadiomicsFeatureExtractor()
print('Extraction parameters:\n\t', extractor.settings)

extractor.enableImageTypeByName('LoG')
extractor.enableImageTypeByName('Wavelet')
extractor.enableAllFeatures()
# extractor.enableFeaturesByName(
#     firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean',
#                 'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
#                 'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
# extractor.enableFeaturesByName(
#     shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2',
#            'Sphericity', 'SphericalDisproportion', 'Maximum3DDiameter', 'Maximum2DDiameterSlice',
#            'Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength',
#            'Elongation', 'Flatness'])
# 上边两句我将一阶特征和形状特征中的默认禁用的特征都手动启用，为了之后特征筛选
print('Enabled filters:\n\t', extractor.enabledImagetypes)

'''
特征提取方法
root_path:要提取特征的文件地址
extractor:特征提取器，需提前定义
file_name:特征保留文件名，.xlsx格式，无需后缀
'''


def ExtractFeatures(root_path, extractor, file_name):
    image_list = os.listdir(root_path)  # 将改地址下的图片名称序列化，该方法得到的list顺序与文件夹中文件的顺序不一定相同
    image_list = sorted(image_list)
    print(image_list)
    result = None  # 创建一个空对象，用处存储之后的特征数据
    columns = None  # 用于处存储特征名
    index = []  # 用于存储标签作为index
    for i in range(0, len(image_list), 2):
        index.append(image_list[i].split('.')[0])
        image_path = os.path.join(root_path, image_list[i + 1])  # 组合出每张图片的地址
        label_path = os.path.join(root_path, image_list[i])  # 组合出每张图片对应的label地址
        image = sitk.ReadImage(image_path)  # %读取图片
        label = sitk.ReadImage(label_path)  # %读取图片对应的label
        # features = extractor.execute(image)  # 特征提取
        features = extractor.execute(image, label)  # 特征提取
        features_dict = features.items()  # 将特征转换成字典
        features_df = pd.DataFrame(list(features_dict)).T  # 将字典转化成dataFrame对象,并转置
        features_df.columns = features_df.iloc[0, :]  # 对df对象做处理，主要是表格格式化
        _str_array = [s.replace('-', '') for s in features_df.columns]
        features_df.columns = [s.replace('_', '') for s in _str_array]
        features_df.drop(index=[0], inplace=True)  # 同上
        if i == 0:  # 第一张图片，直接赋值
            result = features_df
            columns = features_df.columns
        else:
            result = np.append(result, features_df, axis=0)  # 在追加features_df时，result的类型会从dataframe变成ndarray
    result = pd.DataFrame(data=result[0:, 0:], columns=columns)  # 将格式变回dataframe，才能写入excel文件
    result.index = index  # 转换dataframe index 行索引
    xlsx_name = file_name + '.xlsx'
    xlsx_obj = pd.ExcelWriter(xlsx_name)
    result.to_excel(xlsx_obj)
    xlsx_obj._save()
    print('計算結束')


print(os.getcwd())
root_path = '/root/IdeaProjects/model/input/images'
# images_list = os.listdir(root_path)
# images_path = os.path.join(root_path, images_list[0])
# images = sitk.ReadImage(images_path)
# plt.imshow(sitk.GetArrayFromImage(images)[12, :, :], cmap="gray")
ExtractFeatures(root_path=root_path, extractor=extractor, file_name='../../../output/test1')
