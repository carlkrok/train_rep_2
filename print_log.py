import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data1 = pd.read_csv('log_monday.csv', index_col = False)
data1.columns = ['epoch', 'loss','mean_squared_error','val_loss','val_mean_squared_error']

data2 = pd.read_csv('log_tuesday.csv', index_col = False)
data2.columns = ['epoch', 'loss','mean_squared_error','val_loss','val_mean_squared_error']

x = np.arange(0,400,1)

y1_1 = data1['loss']
y1_1 = np.array(y1_1)

y2_1 = data1['val_loss']
y2_1 = np.array(y2_1)

y1_2 = data2['loss']
y1_2  = np.array(y1_2 )

y2_2  = data2['val_loss']
y2_2  = np.array(y2_2 )

y1 = np.append(y1_1,y1_2)
y2 = np.append(y2_1,y2_2)


plt.plot(x,y1,'-r',x,y2,'-b')
plt.show()
