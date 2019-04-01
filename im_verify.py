# 验证相似度随裁切矩阵大小的变化。
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


if __name__ == '__main__':


    # 99张原始图像 + 99*4张微博压缩图像
    # 99张一轮

    # 不同的超参数裁切矩阵验证
    for ki in range(10):
        sum_n = 0.
        sum_1 = 0.
        sum_2 = 0.
        sum_3 = 0.
        sum_4 = 0.
        k = ki + 2

        for i in range(99):
            # print("==========图" + str(i+1) + "========")
            # 压缩1~4次后与原图比较
            for j in range(4):
                HASH_orgin = pHash_verify("WB2019/Database/" + str(i+1) + "_450.png", 8)
                HASH_after = pHash_verify("WB2019/Database_y" + str(j+1) + "/" + str(i+1) + ".jpg", 8)
                n = hammingDistPro(HASH_orgin, HASH_after)
                # print('第'+ str(j+1) + '轮——感知哈希算法相似度：', n)

                HASH_orgin_b = pHash_verify("WB2019/Database/" + str(i+1) + "_450.png", k)
                HASH_after_b = pHash_verify("WB2019/Database_y" + str(j+1) + "/" + str(i+1) + ".jpg", k)
                m = hammingDistPro(HASH_orgin_b, HASH_after_b)
                # print('第'+ str(j+1) + '轮——改良的感知哈希算法相似度：', m)

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

        print("改良识别率累计:(", k, "*", k, ")", sum_n)

        # 测试说明随着裁切矩阵的增大，相似度下降。