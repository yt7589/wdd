#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

class KwmDs(object):
    def __init__(self):
        self.refl = 'apps.kwm.ds.KwmDs'

    def pre_analysize(self):
        df = pd.read_pickle("./datas/LSWMD.pkl")
        print(df.info())