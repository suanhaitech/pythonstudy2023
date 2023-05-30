"""
用marker 参数来定义标记的形状；此外，设置标记的大小、内外颜色
"""
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r', mfc = 'b')
plt.show()
