#
import numpy as np
import pandas as pd

def main(args={}):
    df = pd.DataFrame({'A':np.array([100, 200, 300, 400, 500]), 'B':['a', 'b', 'c', 'd', 'e'],'C':[1, 2, 3, 4, 5]})
    print('type: {0};'.format(type(df['A'])))
    print(df[ (df['A']>=150) & (df['A']<300) ])

if '__main__' == __name__:
    main()