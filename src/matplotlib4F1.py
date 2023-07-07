import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
y1=np.array([10,13,5,40,30,60,70,12,55,25])
x1=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x2=range(0,10)
y2=[5,8,0,30,20,40,50,10,40,15]
plt.plot(x1, y1, label='Frist line', linewidth=3, color='r', marker='o',
         markerfacecolor='blue', markersize=12)
plt.plot(x2, y2, label='second line')

# 平滑选项
# x1_smooth = np.linspace(x1.min(), x1.max(), 300) #300 represents number of points to make between T.min and T.max
# y1_smooth = make_interp_spline(x1, y1)(x1_smooth)
# plt.plot(x1_smooth, y1_smooth)


plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()