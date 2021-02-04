#
import numpy as np
import matplotlib.pyplot as plt

class RegionFeature(object):
    def __init__(self):
        self.name = 'apps.kwm.RegionFeature'

    def extract_feature(self, df_withpattern):
        return df_withpattern.waferMap.apply(self.find_regions)

    def show_region_densitys(self, df_withpattern):
        x = [9,340, 3, 16, 0, 25, 84, 37]
        labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

        fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
        ax = ax.ravel(order='C')
        for i in range(8):
            ax[i].bar(np.linspace(1,13,13),df_withpattern.fea_reg[x[i]])
            ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=15)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.tight_layout()
        plt.show() 

    def draw_regions(self):
        # illustration of 13 regions
        an = np.linspace(0, 2*np.pi, 100)
        plt.plot(2.5*np.cos(an), 2.5*np.sin(an))
        plt.axis('equal')
        plt.axis([-4, 4, -4, 4])
        plt.plot([-2.5, 2.5], [1.5, 1.5])
        plt.plot([-2.5, 2.5], [0.5, 0.5 ])
        plt.plot([-2.5, 2.5], [-0.5, -0.5 ])
        plt.plot([-2.5, 2.5], [-1.5,-1.5 ])

        plt.plot([0.5, 0.5], [-2.5, 2.5])
        plt.plot([1.5, 1.5], [-2.5, 2.5])
        plt.plot([-0.5, -0.5], [-2.5, 2.5])
        plt.plot([-1.5, -1.5], [-2.5, 2.5])
        plt.title(" Devide wafer map to 13 regions")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def cal_den(self, x):
        return 100*(np.sum(x==2)/np.size(x))  

    def find_regions(self, x):
        rows=np.size(x,axis=0)
        cols=np.size(x,axis=1)
        ind1=np.arange(0,rows,rows//5)
        ind2=np.arange(0,cols,cols//5)
        # 边缘区域定义
        reg1=x[ind1[0]:ind1[1],:]
        reg3=x[ind1[4]:,:]
        reg4=x[:,ind2[0]:ind2[1]]
        reg2=x[:,ind2[4]:]
        # 中间9个区域定义
        reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
        reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
        reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
        reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
        reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
        reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
        reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
        reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
        reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
        # 对每个区域求出其密度：所有元素平方和除以元素个数
        fea_reg_den = []
        fea_reg_den = [self.cal_den(reg1), self.cal_den(reg2), self.cal_den(reg3), 
                    self.cal_den(reg4), self.cal_den(reg5), self.cal_den(reg6),
                    self.cal_den(reg7), self.cal_den(reg8), self.cal_den(reg9),
                    self.cal_den(reg10), self.cal_den(reg11), self.cal_den(reg12),
                    self.cal_den(reg13)]
        return fea_reg_den