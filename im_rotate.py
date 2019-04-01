import cv2
import numpy as np
import time


def avg(img):
    d = 0
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img
    for i in vis0:
        d += sum(i)

    return d/(32*16)

def getHighestValue(nsI,ssI,wsI,esI):
    n = avg(nsI)
    s = avg(ssI)
    w = avg(wsI)
    e = avg(esI)
    return max([n,s,w,e])

def imageRotation(gI, sI):
    height, width = sI.shape[:2] # 这里重现，遇到问题，是按照算法原理理解而修改的。
    if width == height:
        nsI = sI[0: width, 0:int(height/2)]
        ssI = sI[0: width, int(height/2): height]
        wsI = sI[0: int(width / 2), 0: height]
        esI = sI[int(width / 2): width, 0: height] # 这里论文有误。
        highestMean = getHighestValue(nsI,ssI,wsI,esI)
        if highestMean == avg(nsI):
            sI = np.rot90(sI, 2)
        elif highestMean == avg(wsI):
            sI = np.rot90(sI, 3)
        elif highestMean == avg(esI):
            sI = np.rot90(sI)

    else:
        if width < height:
            sI = np.rot90(sI, 1)
        nsI = sI[0: width, 0: int(height / 2)]
        ssI = sI[0: width, int(height / 2): height]
        if avg(nsI) < avg(ssI): # 符号问题
            sI = np.rot90(sI, 2)
    return sI


def pHashBase(imgfile):
    img_list = []
    # 加载并调整图片为32x32灰度图片
    img0 = cv2.imread(imgfile, 0)

    img = cv2.resize(img0, (32, 32), interpolation=cv2.INTER_CUBIC)

    # 旋转图像
    img = imageRotation(img0, img)

    # cv2.imwrite('c.jpg', img)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(vis0)
    cv2.imwrite('a.jpg', vis0) # 保存原始压缩图片
    cv2.imwrite('b.jpg', vis1) # 保存频率图片
    # vis1.resize(32, 32)

    # 裁切图像
    vis1 = vis1[0:12, 0:12]

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 1]), 2) for x in range(0, 12 * 12, 1)])

def hammingDistPro(s1, s2):
    # assert len(s1) == len(s2)
    return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 1. / (len(s1))



if __name__ == '__main__':

    time1 = time.time()
    HASH1 = pHashBase("rotate/12_450.png")
    HASH2 = pHashBase("rotate/12_450.png_270.png")
    n = hammingDistPro(HASH1, HASH2)
    print('感知哈希算法相似度：', n, "-----time=", (time.time() - time1))
