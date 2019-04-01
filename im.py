import cv2
import numpy as np
import time


# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
            # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值感知算法
def dHash(img):
    # 缩放8*8
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# Hash值对比
# def cmpHash(hash1, hash2):
#     n = 0
#     # hash长度不同则返回-1代表传参出错
#     if len(hash1) != len(hash2):
#         return -1
#     # 遍历判断
#     for i in range(len(hash1)):
#         # 不相等则n计数+1，n最终为相似度
#         if hash1[i] != hash2[i]:
#             n = n + 1
#     return 1 - n / len(hash1)


def pHash(imgfile):
    img_list = []
    # 加载并调整图片为32x32灰度图片
    img0 = cv2.imread(imgfile, 0)
    img = cv2.resize(img0, (32, 32), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('c.jpg', img0)
    # cv2.imwrite('d.jpg', img)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(vis0)
    cv2.imwrite('a.jpg', vis0) #保存原始压缩图片
    cv2.imwrite('b.jpg', vis1) #保存频率图片
    # vis1.resize(32, 32)

    # 裁切图像
    vis1 = vis1[0:8, 0:8]

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 1]), 2) for x in range(0, 8 * 8, 1)])

def avg(img):
    d = 0
    for i in range(img):
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
    # 加载并调整图片为32x32灰度图片
    img0 = cv2.imread(imgfile, 0)
    img = cv2.resize(img0, (32, 32), interpolation=cv2.INTER_CUBIC)

    # 旋转图像
    # img = imageRotation(img0, img)

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
    # img1 = cv2.imread("Database_0_9/Database/6_450.png")
    # img2 = cv2.imread("Database_0_9/Database_y1/3.jpg")
    # time1 = time.time()
    # hash1 = aHash(img1)
    # hash2 = aHash(img2)
    # n = hammingDistPro(hash1, hash2)
    # print('均值哈希算法相似度：', n, "-----time=", (time.time() - time1))
    # time1 = time.time()
    # hash1 = dHash(img1)
    # hash2 = dHash(img2)
    # n = hammingDistPro(hash1, hash2)
    # print('差值哈希算法相似度：', n, "-----time=", (time.time() - time1))

    # time1 = time.time()
    # HASH1 = pHashBase("Database_0_9/Database/1_450.png")
    # HASH2 = pHashBase("Database_0_9/Database_y4/1.jpg")
    # n = hammingDistPro(HASH1, HASH2)
    # print('感知哈希算法相似度：', n, "-----time=", (time.time() - time1))

    # 99张原始图像 + 99*4张微博压缩图像
    # 99张一轮
    sum_n = 0.
    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.
    sum_4 = 0.

    for i in range(99):
        print("==========图" + str(i+1) + "========")
        # 压缩1~4次后与原图比较
        for j in range(4):
            HASH_orgin = pHash("WB2019/Database/" + str(i+1) + "_450.png")
            HASH_after = pHash("WB2019/Database_y" + str(j+1) + "/" + str(i+1) + ".jpg")
            n = hammingDistPro(HASH_orgin, HASH_after)
            print('第'+ str(j+1) + '轮——感知哈希算法相似度：', n)

            HASH_orgin_b = pHashBase("WB2019/Database/" + str(i+1) + "_450.png")
            HASH_after_b = pHashBase("WB2019/Database_y" + str(j+1) + "/" + str(i+1) + ".jpg")
            m = hammingDistPro(HASH_orgin_b, HASH_after_b)
            print('第'+ str(j+1) + '轮——改良的感知哈希算法相似度：', m)

            c = m - n
            sum_n += c

            if j == 0:
                sum_1 += c
            elif j == 1:
                sum_2 += c
            elif j ==2:
                sum_3 += c
            elif j == 3:
                sum_4 += c

    for i in range(4):
        print("第" + str(i+1) + "轮 改良识别率累计：", eval("sum_" + str(i+1)))

    print("改良识别率累计:", sum_n)
