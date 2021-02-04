#
import numpy as np
from apps.kwm.ds.kwm_ds import KwmDs
from apps.kwm.xgb_engine import XgbEngine

class KwmApp(object):
    WORK_RAW_DATA = 1
    WORK_TXT_DATA =2
    WORK_CSV_DATA = 3

    def __init__(self):
        self.refl = 'apps.kwm.KwmApp'

    def startup(self, args={}):
        print('晶圆缺陷检测应用WDD v0.0.2')
        mode = KwmApp.WORK_CSV_DATA
        ds = KwmDs()
        if KwmApp.WORK_RAW_DATA == mode:
            X, y = ds.generate_ds()
            print('保存数据集到文本文件：')
            np.savetxt('./datas/kwm_x.txt', X, fmt='%.18e', delimiter=',', newline='\n', encoding='utf-8')
            np.savetxt('./datas/kwm_y.txt', y, fmt='%d', delimiter=',', newline='\n', encoding='utf-8')
        elif KwmApp.WORK_TXT_DATA == mode:
            X, y = ds.load_processed_ds()
            print('X: type={0}; shape={1};'.format(type(X), X.shape))
            print('y: type={0}; shape={1};'.format(type(y), y.shape))
            clss = np.unique(y)
            print('classes:{0};'.format(clss))
            # self.save_csv(X, y) # 为XGBoost算法准备数据集
        elif KwmApp.WORK_CSV_DATA == mode:
            self.run_xgboost()
        print('^_^ The End ^_^')

    def run_xgboost(self):
        print('XGBoost算法应用')
        xgbe = XgbEngine('./datas/kwm_xgb.csv', 59, 8)
        xgbe.startup(XgbEngine.MODE_TRAIN)

    def save_csv(self, X, y):
        y = y.reshape(y.shape[0], 1)
        data = np.concatenate((X, y), axis=1)
        print('data: {0};'.format(data.shape))
        #np.savetxt('./datas/kwm_xgb.csv', data, delimiter=',', newline='\n', encoding='utf-8')
        with open('./datas/kwm_xgb.csv', 'w', encoding='utf-8') as fd:
            for row in data:
                for idx in range(0, 59):
                    fd.write('{0},'.format(row[idx]))
                fd.write('{0}\n'.format(int(row[59])))
