import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import segyio


segyFile = sys.argv[1]

with segyio.open(segyFile, 'r+', ignore_geometry=True) as f:
        x = np.zeros((len(f.trace[0]),len(f.trace)))
        for i in range(len(f.trace)):
                #print(f.trace[i][0])
                x[:,i] = f.trace[i]

minx = np.min(x)
maxx = np.max(x)
plt.imshow(x, aspect='auto', norm=colors.Normalize(vmin=minx/100, vmax=maxx/100))
plt.xlabel("Receivers")
plt.ylabel("sample")
plt.show()
