#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import gridspec
# 引入特征类
from apps.kwm.region_feature import RegionFeature
from apps.kwm.radon_transform_feature import RadonTransformFeature
from apps.kwm.geometry_feature import GeometryFeature

class KwmDs(object):
    def __init__(self):
        self.refl = 'apps.kwm.ds.KwmDs'

    def load_processed_ds(self):
        X = np.loadtxt('./datas/kwm_x.txt', dtype=np.float, delimiter=',', encoding='utf-8')
        y = np.loadtxt('./datas/kwm_y.txt', dtype=np.int, delimiter=',', encoding='utf-8')
        return X, y

    def generate_ds(self):
        # 载入数据集
        df_withpattern = self.load_ds()
        # 获取区域特征df_withpattern['fea_reg']
        region_feature = RegionFeature()
        df_withpattern['fea_reg'] = region_feature.extract_feature(df_withpattern=df_withpattern)
        region_feature.show_region_densitys(df_withpattern=df_withpattern)
        # 获取Radon变换特征
        rtf = RadonTransformFeature()
        df_withpattern['rtf_cub_mean'], df_withpattern['rtf_cub_std'] = rtf.extract_feature(df_withpattern=df_withpattern)
        # 获取几何特征
        geom_feat = GeometryFeature()
        df_withpattern['fea_geom'] = df_withpattern.waferMap.apply(geom_feat.extract_feature)
        # 生成数据集
        df_all = df_withpattern.copy()
        a = [df_all.fea_reg[i] for i in range(df_all.shape[0])] #13
        b = [df_all.rtf_cub_mean[i] for i in range(df_all.shape[0])] #20
        c = [df_all.rtf_cub_std[i] for i in range(df_all.shape[0])] #20
        d = [df_all.fea_geom[i] for i in range(df_all.shape[0])] #6
        fea_all = np.concatenate((np.array(a), np.array(b), np.array(c), np.array(d)), axis=1)
        label=[df_all.failureNum[i] for i in range(df_all.shape[0])]
        label=np.array(label)
        return fea_all, label

    def load_ds(self):
        df = pd.read_pickle("./datas/LSWMD.pkl")
        print('数据集基本情况：v0.0.2')
        df.info()
        #self.draw_wafer_index_distribution(df)
        df = df.drop(['waferIndex'], axis = 1)
        #self.check_sample_dim(df)
        df = self.proc_missing_vals(df)
        df_withpattern = self.count_failure_type(df)
        #self.draw_first_100_samples(df, df_withpattern=df_withpattern)
        #self.draw_failure_types(df_withpattern=df_withpattern)
        self.draw_typical_failure_type(df_withpattern)
        return df_withpattern


    def draw_wafer_index_distribution(self, df):
        uni_Index = np.unique(df.waferIndex, return_counts=True)
        plt.bar(uni_Index[0],uni_Index[1], color='blue', align='center', alpha=0.5)
        plt.title('wafer Index distribution')
        plt.xlabel('index #')
        plt.ylabel('frequency')
        plt.xlim(0, 26)
        plt.ylim(30000, 34000)
        plt.show()

    def find_dim(self, x):
        dim0 = np.size(x,axis=0)
        dim1 = np.size(x,axis=1)
        return dim0,dim1
        
    def check_sample_dim(self, df):
        df['waferMapDim'] = df.waferMap.apply(self.find_dim)
        print('max: {0}; min: {1};'.format(max(df.waferMapDim), min(df.waferMapDim)))
        uni_waferDim = np.unique(df.waferMapDim, return_counts=True)
        print(uni_waferDim[0].shape[0])

    def proc_missing_vals(self, df):
        df['failureNum'] = df.failureType
        df['trainTestNum'] = df.trianTestLabel
        mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
        mapping_traintest={'Training':0,'Test':1}
        print('总样本数：{0};'.format(df.shape[0]))
        return df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

    def count_failure_type(self, df):
        df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
        df_withlabel =df_withlabel.reset_index()
        df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
        df_withpattern = df_withpattern.reset_index()
        df_nonpattern = df[(df['failureNum']==8)]
        print('有标签样本数：{0}; 缺陷样本数：{0}; 正常样本数：{1};'.format(
            df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]
        ))
        fig = plt.figure(figsize=(20, 4.5)) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5]) 
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        no_wafers=[df.shape[0]-df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]
        colors = ['silver', 'orange', 'blue']
        explode = (0.1, 0, 0)  # explode 1st slice
        labels = ['no-label','label&pattern','label&non-pattern']
        ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
        labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
        ax2.bar(uni_pattern[0],uni_pattern[1]/df_withpattern.shape[0], color='blue', align='center', alpha=0.9)
        ax2.set_title("failure type frequency")
        ax2.set_ylabel("% of pattern wafers")
        ax2.set_xticklabels(labels2)
        plt.show()
        return df_withpattern

    def draw_first_100_samples(self, df, df_withpattern):
        fig, ax = plt.subplots(nrows = 2, ncols = 10, figsize=(20, 20))
        ax = ax.ravel(order='C')
        for i in range(20):
            img = df_withpattern.waferMap[i]
            ax[i].imshow(img)
            ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
            ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.tight_layout()
        plt.show()

    def draw_failure_types(self, df_withpattern):
        x = [0,1,2,3,4,5,6,7]
        labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
        for k in x:
            fig, ax = plt.subplots(nrows = 1, ncols = 10, figsize=(18, 12))
            ax = ax.ravel(order='C')
            for j in [k]:
                img = df_withpattern.waferMap[df_withpattern.failureType==labels2[j]]
                for i in range(10):
                    ax[i].imshow(img[img.index[i]])
                    ax[i].set_title(df_withpattern.failureType[img.index[i]][0][0], fontsize=10)
                    ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
        plt.tight_layout()
        plt.show()

    def draw_typical_failure_type(self, df_withpattern):
        x = [9,340, 3, 16, 0, 25, 84, 37]
        labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
        fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
        ax = ax.ravel(order='C')
        for i in range(8):
            img = df_withpattern.waferMap[x[i]]
            ax[i].imshow(img)
            ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=24)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.tight_layout()
        plt.show() 