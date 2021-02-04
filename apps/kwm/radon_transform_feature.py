#
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr # radon
import scipy #import interpolate

class RadonTransformFeature(object):
    def __init__(self):
        self.name = ''

    def extract_feature(self, df_withpattern):
        print('Radon Transform Feature extractor')
        df_withpattern_copy = df_withpattern.copy()
        df_withpattern_copy['new_waferMap'] =df_withpattern_copy.waferMap.apply(self.change_val)
        # 为什么操作的是waferMap而不是new_waferMap?????????????????????
        self.draw_raw_rtf(df_withpattern_copy)
        df_withpattern_copy['rtf_cub_mean'] = df_withpattern_copy.waferMap.apply(self.cubic_inter_mean)
        self.draw_rtf_cub(df_withpattern_copy, 1)
        df_withpattern_copy['rtf_cub_std'] = df_withpattern_copy.waferMap.apply(self.cubic_inter_std)
        self.draw_rtf_cub(df_withpattern_copy, 2)
        return df_withpattern_copy['rtf_cub_mean'], df_withpattern_copy['rtf_cub_std']


    def change_val(self, img):
        img[img==1] =0  
        return img

    def cubic_inter_mean(self, img):
        return self.cubic_interplate(img, np.mean)

    def cubic_inter_std(self, img):
        return self.cubic_interplate(img, np.std)

    def cubic_interplate(self, img, proc):
        theta = np.linspace(0., 180., max(img.shape), endpoint=False)
        sinogram = sktr.radon(img, theta=theta)
        x_Row = proc(sinogram, axis = 1)
        x = np.linspace(1, x_Row.size, x_Row.size)
        y = x_Row
        f = scipy.interpolate.interp1d(x, y, kind = 'cubic')
        xnew = np.linspace(1, x_Row.size, 20)
        ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
        return ynew

    def draw_raw_rtf(self, df_withpattern_copy):
        x = [9,340, 3, 16, 0, 25, 84, 37]
        labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
        fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
        ax = ax.ravel(order='C')
        for i in range(8):
            img = df_withpattern_copy.waferMap[x[i]]
            theta = np.linspace(0., 180., max(img.shape), endpoint=False)
            sinogram = sktr.radon(img, theta=theta)
            ax[i].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
            ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=15)
            ax[i].set_xticks([])
        plt.tight_layout()
        plt.show()

    def draw_rtf_cub(self, df_withpattern_copy, mode):
        x = [9,340, 3, 16, 0, 25, 84, 37]
        labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
        fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
        ax = ax.ravel(order='C')
        for i in range(8):
            if 1 == mode:
                ax[i].bar(np.linspace(1,20,20),df_withpattern_copy.rtf_cub_mean[x[i]])
            else:
                ax[i].bar(np.linspace(1,20,20),df_withpattern_copy.rtf_cub_std[x[i]])
            ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_xlim([0,21])   
            ax[i].set_ylim([0,1])
        plt.tight_layout()
        plt.show() 