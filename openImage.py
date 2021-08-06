
import matplotlib.pyplot as plt
import numpy as np


# file = open("C:\\Users\jones\Desktop\MammData\Test-Images-GE-format\M2000210_1612_LCC.img", 'rb')

def load_image(fileName):
    file = open(fileName, 'rb')

    # key is the ID
    # value[0] is the starting byte index
    # value[1] is the length of the key in bytes
    headerVariableNames = {
        'siteid': [0 ,16],
        'xcrop': [16,8],
        'ycrop': [24,8],
        'filename': [32,16],
        'headersize': [48,8],
        'bits':[56,4],
        'rows':[60,8],
        'columns':[68,8],
        'pixelsize': [76, 12],
        'spotsize': [88,4],
        'examnum':[92,8],
        'imagenum':[100,8],
        'imageinthisexam':[108,4],
        'view': [ 112,16],
        'lastname':[128,32],
        'firstname': [160,16],
        'midinit':[176,2],
        'dob': [178,16],
        'ssnum': [194,16],
        'examdate': [210,16],
        'bodypart': [226,16],
        'recordnum': [242,8],
        'padding': [249,6]
    }

    header = {}

    for key, value in headerVariableNames.items():
        file.seek(value[0])
        header[key]= str(file.read(value[1]))

    nrow = int(header['rows'][2:5])
    ncolumn = int(header['columns'][2:5])
    headerSize = 256 ## intialzing this here since it is left blank in the image header

    ## load the image, subsample by 2
    m = 256
    file.seek(m) #jump to the first byte after the header

    img_raw = np.zeros((nrow,ncolumn))
    index = 256

    for i in range(nrow):
        twoRows = file.read(ncolumn*2)
        index+=(ncolumn*2)
        file.seek(index)
        k = 0
        for j in range(ncolumn):
            img_raw[i][j] = 256 * twoRows[k] + twoRows[k+1]
            k+=2

    img = 4095 -img_raw
    img = np.clip(img, 0, 5000) #negative pixels become 0

    # plt.imshow(img,cmap='gray')
    # plt.show()

    return img

