#
from apps.kwm.ds.kwm_ds import KwmDs

class KwmApp(object):
    def __init__(self):
        self.refl = 'apps.kwm.KwmApp'

    def startup(self, args={}):
        print('晶圆缺陷检测应用WDD v0.0.1')
        ds = KwmDs()
        ds.pre_analysize()