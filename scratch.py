import numpy as np
import matplotlib.pyplot as plt

originalImg = np.load("originalIm.npy")
img_64 = np.load("64x64.npy")
img_224 = np.load("224x224.npy")
img_8bit = np.load("8bit.npy")

x = np.zeros((224, 224, 3))  # size of image that VGG16 takes
x[:, :, 0] = img_8bit
x[:, :, 1] = img_8bit
x[:, :, 2] = img_8bit

fig = plt.figure()
ax1 = fig.add_subplot(1, 5, 1)
ax1.imshow(originalImg)
ax2 = fig.add_subplot(1, 5, 2)
ax2.imshow(img_64)
ax3 = fig.add_subplot(1, 5, 3)
ax3.imshow(img_224)
ax4 = fig.add_subplot(1, 5, 4)
ax4.imshow(img_8bit)
ax5 = fig.add_subplot(1,5,5)
ax5.imshow(x)
plt.show()

(115, "ARP1052_1201_LCC.img")
(753, "ARP0239_1001_LCC.img")
(995,"FK0101000_1025_LCC.img")