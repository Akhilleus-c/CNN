# Eksikleri var.

import numpy as np
from PIL import Image
import time
def mesure_time(function):
    def wrapper(img, fltr):
        nonlocal function
        innit = time.time()
        b = function(img, fltr)
        final = time.time()
        print(f"{(final - innit)} - time passed")
        return b
    return wrapper


def color_changer(path):
    global normalization
    img = Image.open(path).convert("L")
    img.save("ymar.jpg")
    img = np.array(img)



    def normalization(img:np.array):
        ma = img.max()
        mi = img.min()
        img = (img - mi) / (ma - mi)
        return img

    return normalization(img)


img = color_changer(r"C:\Users\bb\Zac\resm1.png")

fltr = np.random.random(9).reshape(3, 3)

def relU(arr:np.array):
    arr[arr < 0] = 0
    return arr

@mesure_time
def filterize(img:np.array, filter:np.array):
    b = filter.sum() - 1
    l = []
    row = -1
    column = -1
    flag = 0
    while True:
        if flag == 2: break
        row += 1
        lr = row + 3
        column = -1
        while True:
                column += 1
                imgsframes = img[row:lr, column:column + 3]
                if imgsframes.size != 9:
                     flag += 1
                     break
                result = (imgsframes * filter).sum() - b
                l.append(float(result))
                flag = 0
    result = np.array(l).reshape(718, 1278)
    print(result)
    # l.clear()
    # cols = 1278
    # ros = 718
    # for rs in range(0, cols, 2):
    #     for cs in range(0, ros, 2): 
    #         a = arr[rs:rs + 2, cs:cs + 2].max()
    #         if a.size != 4:
    #             break
    #         l.append(arr[rs:rs + 2, cs:cs + 2].max())
            




print(filterize(img, fltr))
