import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import inspect

fig, ax = plt.subplots(1, 2, figsize=[12, 6])

r = 8
im = ps.generators.cylinders(shape=[200, 200, 200], r=r, ncylinders=100)
ax[0].imshow(ps.visualization.sem(im, axis=2), cmap=plt.cm.bone)
ax[0].axis(False)

r = 16
im = ps.generators.cylinders(shape=[200, 200, 200], r=r, ncylinders=100)

ax[1].imshow(ps.visualization.sem(im, axis=2), cmap=plt.cm.bone)
ax[1].axis(False)
plt.show()
