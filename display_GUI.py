import tkinter as tk
from PIL import Image, ImageTk
import openImage
import segmentImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import scipy.ndimage
import os

root = tk.Tk()

imageFilePath = "C:\\Users\jones\Desktop\MammData\Test-Images-GE-format\M2000210_1612_LCC.img"

img = openImage.load_image(imageFilePath)
imgRaw, LorR = segmentImage.segment(imageFilePath, img)
imageName = os.path.basename(imageFilePath)

img = imgRaw.astype(np.uint8)  # normalize to 0-255 for visualization using tkinter

f = Figure()
a = f.add_subplot(111)
a.imshow(imgRaw, cmap='gray')
if LorR =='R':
    a.set_title("Right Breast")
else:
    a.set_title("Left Breast")
a.xaxis.set_visible(False)
a.yaxis.set_visible(False)


canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
canvas._tkcanvas.pack(side="top", fill="both", expand=1)

root.mainloop()

