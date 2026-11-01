import numpy as np
from PIL import Image
import generalUseFunctions

def color_changer(path):
    img = Image.open(path).convert("L")
    img.save("ymar.jpg")
    img = np.array(img)
    return generalUseFunctions.normalization(img)


@generalUseFunctions.mesure_time
def filterize(img:np.array, filter:np.array):
    b = filter.sum() - 1
    l = []
    row = -1
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
    l.clear()
    flag = 0
    row = -1
    while True:
        if flag == 2: break
        row += 2
        lr = row + 2
        column = -1
        while True:
                column += 2
                resltfrs = result[row:lr, column:column + 2]
                if resltfrs.size != 4:
                     flag += 1
                     break
                l.append(float(resltfrs.max()))
                flag = 0
    
    final = np.array(l)
    final = generalUseFunctions.relU(final)
    return final
    

if __name__ == "__main__":
    img = color_changer(r"C:\Users\zeker\Projeler\resm1.jpg")
    fltr = np.random.random(9).reshape(3, 3)
    print(filterize(img, fltr))
