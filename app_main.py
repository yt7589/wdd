#
from apps.kwm.kwm_app import KwmApp

def main(args={}):
    print('晶圆缺陷检测 v0.0.1')
    app = KwmApp()
    app.startup()

if '__main__' == __name__:
    main()