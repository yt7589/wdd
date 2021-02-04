#
import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn.model_selection as skms
from xgboost.callback import early_stop #import train_test_split


class XgbEngine(object):
    MODE_TRAIN = 1
    MODE_RUN = 2

    def __init__(self, data_path, feature_num, num_class):
        self.name = 'apps.kwm.XgbEngine'
        self.bst = None
        self.data_path = data_path
        self.feature_num = feature_num
        self.num_class = num_class
        self.model_file = './work/xgb.bst'

    def startup(self, mode):
        X_train, X_test, y_train, y_test = self.load_ds()
        if XgbEngine.MODE_TRAIN == mode:
            self.bst = self.build_model(X_train, X_test, y_train, y_test)
        else:
            self.load_model()
        xgb_test=xgb.DMatrix(X_test, label=y_test)
        pred = self.predict(xgb_test)
        accuracy = self.evaluate_model(X_test, y_test, pred)
        print('测试集上精度为：{0};'.format(accuracy))
        self.save_model()

    def load_ds(self):
        data = pd.read_csv(self.data_path, header=None, 
                    sep=',',converters={self.feature_num : lambda x:int(x)})
        data.rename(columns={self.feature_num:'lable'}, inplace=True)
        X=data.iloc[:,:self.feature_num-1]
        Y=data.iloc[:,self.feature_num]
        return skms.train_test_split(X, Y, test_size=0.25, random_state=100)

    def build_model(self, X_train, X_test, y_train, y_test):
        xgb_train=xgb.DMatrix(X_train, label=y_train)
        xgb_test=xgb.DMatrix(X_test, label=y_test)
        params={
            'objective':'multi:softmax',
            'eta':0.1,
            'max_depth':5,
            'num_class': self.num_class
        }
        watchlist=[(xgb_train,'train'),(xgb_test,'test')]
        num_round = 600
        eval_set = (X_test, y_test)
        evals_result = ({'eval_metric': 'mlogloss'})
        early_stopping_rounds = 60
        return xgb.train(params, xgb_train, num_round, watchlist, 
                    early_stopping_rounds=early_stopping_rounds, 
                    evals_result=evals_result)

    def predict(self, X):
        return self.bst.predict(X)

    def evaluate_model(self, X_test, y_test, pred):
        error_rate=np.sum(pred!=y_test)/y_test.shape[0]
        accuracy=1-error_rate
        return accuracy

    def save_model(self):
        self.bst.save_model(self.model_file)

    def load_model(self):
        self.bst = xgb.Booster()
        self.bst.load_model(self.model_file)

