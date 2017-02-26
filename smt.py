# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 01:57:42 2017

@author: Chella Rm
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.text import Text
from matplotlib.finance import candlestick2_ohlc
import numpy as np
import pandas as pd
from mpl_interaction import PanAndZoom


df = pd.read_csv('table.csv',index_col=0)
df.columns =["PX_OPEN","PX_HIGH","PX_LOW","PX_LAST"]



# np.random.seed(0)
# dates = pd.date_range('20160101',periods=7)
# df = pd.DataFrame(np.reshape(1+np.random.random_sample(42)*0.1,(7,6)),index=dates,columns=["PX_BID","PX_ASK","PX_LAST","PX_OPEN","PX_HIGH","PX_LOW"])
# df['PX_HIGH']+=.1
# df['PX_LOW']-=.1




fig, ax1 = plt.subplots()
pan_zoom = PanAndZoom(fig)
ax1.set_title('click on points', picker=20)
ax1.set_ylabel('ylabel', picker=20, bbox=dict(facecolor='red'))

(lines,polys) = candlestick2_ohlc(ax1, df['PX_OPEN'],df['PX_HIGH'],df['PX_LOW'],df['PX_LAST'],width=0.4)
lines.set_picker(True)
polys.set_picker(True)

def onpick1(event):
#    if event.mouseevent.button == 1:
#        index = event.ind
#        print df.ix[index]
    if event.dblclick:
        print "SHIT"
#     if isinstance(event.artist, (Text)):
#         text = event.artist
#         print 'You clicked on the title ("%s")' % text.get_text()
#     elif isinstance(event.artist, (LineCollection, PolyCollection)):   
#         thisline = event.artist
#         mouseevent = event.mouseevent
#         ind = event.ind[0]
#         print 'You clicked on item %d' % ind
#         print 'Day: ' + df.index[ind]#.normalize().to_datetime().strftime('%Y-%m-%d')
#         for p in ['PX_OPEN','PX_OPEN','PX_HIGH','PX_LOW']:
#             print p + ':' + str(df[p][ind])    
#         print('x=%d, y=%d, xdata=%f, ydata=%f' %
#           ( mouseevent.x, mouseevent.y, mouseevent.xdata, mouseevent.ydata))



fig.canvas.mpl_connect('button_press_event', onpick1)
plt.show()