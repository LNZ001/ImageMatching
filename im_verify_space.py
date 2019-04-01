# 计算像素空间的利用率，进而估计99%的情况下，1000w张图大概需要多大的矩阵空间.——数据集过小，难以估计
import cv2
import numpy as np
import time

def pHash_verify(imgfile, k):
    img_list = []
    # 加载并调整图片为32x32灰度图片
    img0 = cv2.imread(imgfile, 0)
    img = cv2.resize(img0, (32, 32), interpolation=cv2.INTER_CUBIC)

    # 旋转图像
    # imageRotation(img0, img)

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
    vis1 = vis1[0:k, 0:k]

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 1]), 2) for x in range(0, k * k, 1)])

def hammingDistPro(s1, s2):
    # assert len(s1) == len(s2)
    return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 1. / (len(s1))

def binaryTo(b):
    sum=0
    for i in range(len(b)):
        sum += int(b[len(b)-1-i])*pow(2,i)
    return sum

if __name__ == '__main__':


    # 99张原始图像 + 99*4张微博压缩图像
    # 99张一轮

    # 99% (2~3)

    for ki in range(2):

        k = ki + 2
        count = 0
        sum_list = list2 = [0] * 65535

        print("========== k=", k,"===========")

        # 99
        for i in range(99):
            # print("==========图" + str(i+1) + "========")

            HASH_orgin = pHash_verify("WB2019/Database/" + str(i + 1) + "_450.png", k)
            sum_list[binaryTo(HASH_orgin)] += 1

        q = pow(2, k*k)
        # max = 0
        # for i in range(q):
        #     if sum_list[i] > max:
        #         max = sum[i]
        maxC = max(sum_list)

        p = 99 / (maxC * q)
        print("k=", k, "时，利用率为：", p)



