#
import numpy as np
import skimage as sk #import measure
import scipy # import stats

class GeometryFeature(object):
    def __init__(self):
        self.name = 'apps.kwm.GeometryFeature'

    def cal_dist(self, img,x,y):
        dim0=np.size(img,axis=0)    
        dim1=np.size(img,axis=1)
        dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
        return dist  

    def extract_feature(self, img):
        norm_area=img.shape[0]*img.shape[1]
        norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
        
        img_labels = sk.measure.label(img, neighbors=4, connectivity=1, background=0)

        if img_labels.max()==0:
            img_labels[img_labels==0]=1
            no_region = 0
        else:
            info_region = scipy.stats.mode(img_labels[img_labels>0], axis = None)
            no_region = info_region[0][0]-1       
        
        prop = sk.measure.regionprops(img_labels)
        prop_area = prop[no_region].area/norm_area
        prop_perimeter = prop[no_region].perimeter/norm_perimeter 
        
        prop_cent = prop[no_region].local_centroid 
        prop_cent = self.cal_dist(img,prop_cent[0],prop_cent[1])
        
        prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
        prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
        prop_ecc = prop[no_region].eccentricity  
        prop_solidity = prop[no_region].solidity  
        
        return prop_area,prop_perimeter,prop_majaxis,prop_minaxis,prop_ecc,prop_solidity