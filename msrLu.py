import numpy as np
import rawpy
import cv2
import matplotlib.pyplot as plt
import glob

lensFolderPath = './lens/*'
noLensFolderPath = './nolens/*'
pixels = 50 #グラフ化するピクセル幅 偶数

lensFolder = glob.glob(lensFolderPath)
noLensFolder = glob.glob(noLensFolderPath)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 8), tight_layout=True) #図の準備
xaxis = np.arange(pixels)

lensGrays = np.array([])
noLensGrays = np.array([])

def smoothing(grayImgList):
    resultImg = np.zeros(len(grayImgList[0])*len(grayImgList[0][0]))
    resultImg = np.reshape(resultImg, (len(grayImgList[0]), len(grayImgList[0][0])))
    for i in range(len(grayImgList)):
        resultImg += grayImgList[i]
    resultImg = resultImg / len(grayImgList)
    return resultImg

for i in range(len(lensFolder)):
    with rawpy.imread(lensFolder[i]) as f:
        rwimg = f.postprocess(no_auto_scale=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.raw, no_auto_bright=True, gamma=(1,1), user_flip=0, output_bps=16)
    gimg = cv2.cvtColor(rwimg, cv2.COLOR_RGB2GRAY) #グレースケール化
    lensGrays = np.append(lensGrays, gimg)
lensGrays = np.reshape(lensGrays, (len(lensFolder), len(gimg), len(gimg[0])))

lensGray = smoothing(lensGrays)
lensCenter = lensGray[len(lensGray)//2][len(lensGray[0])//2 -pixels//2:len(lensGray[0])//2 +pixels//2]

axs[0].set_title('Lens')
axs[0].imshow(lensGray, cmap=plt.cm.Greys_r)
axs[2].plot(xaxis, lensCenter, color='blue', label='Lens') 

for i in range(len(noLensFolder)):
    with rawpy.imread(noLensFolder[i]) as f:
        rwimg = f.postprocess(no_auto_scale=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.raw, no_auto_bright=True, gamma=(1,1), user_flip=0, output_bps=16)
    gimg = cv2.cvtColor(rwimg, cv2.COLOR_RGB2GRAY) #グレースケール化
    noLensGrays = np.append(noLensGrays, gimg)
noLensGrays = np.reshape(noLensGrays, (len(noLensFolder), len(gimg), len(gimg[0])))

noLensGray = smoothing(noLensGrays)
noLensCenter = noLensGray[len(noLensGray)//2][len(noLensGray[0])//2 -pixels//2:len(noLensGray[0])//2 +pixels//2]

axs[1].set_title('noLens')
axs[1].imshow(noLensGray, cmap=plt.cm.Greys_r)
axs[2].plot(xaxis, noLensCenter, color='red', label='noLens')

plt.savefig('result.png')