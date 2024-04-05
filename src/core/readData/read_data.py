import pandas as pd
import numpy as np

'''
L表示無效,
X表示有效
'''


class ReadData:
    X_location = 'input/XA_total_.xlsx'
    L_location = 'input/LA_total_.xlsx'

    def __init__(self, X_location=None, L_location=None):
        if X_location is not None:
            self.X_location = X_location
        if L_location is not None:
            self.L_location = L_location

    def read_data(self):
        # if(self.L_location.end)
        data_L = pd.read_excel(self.L_location, index_col=0)
        data_X = pd.read_excel(self.X_location, index_col=0)
        return data_X, data_L
