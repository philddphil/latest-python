import matplotlib.pyplot as plt
import numpy as np
import os

p2 = r'C:\Users\Philip\Documents\LabVIEW\Projects\Data\Cam pics'

os.chdir(p2)

files = os.listdir(p2)

im_size = np.shape(np.loadtxt(files[1]))
sz = (im_size[0], im_size[1], len(files))

ims = np.zeros(sz)
fig = plt.figure()

for i1 in range(len(files)):
    print(i1)
    im = np.loadtxt(files[i1])
    ims[:, :, i1] = im
    plt.imshow(im)

im_sum = np.sum(ims, 2)
plt.imshow(im_sum)
plt.show()
print(np.shape(ims))
