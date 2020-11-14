import cv2
import matplotlib.pyplot as plt
import numpy as np

#计算像素值出现概率
def get_probability(img):
    prob = np.zeros(256)
    for rv in img:
        for cv in rv:
            prob[cv] += 1
    m,n = img.shape
    prob = prob/(m * n)       #求概率值
    return prob
#根据像素概率将原始图像直方图均衡化
def equalization(img, prob):
    prob = np.cumsum(prob)  # 求累计概率
    pixel_trans = [int(255 * prob[i]+0.5) for i in range(256)]  # 像素值映射
    m,n = img.shape
    for j in range(m):
        for k in range(n):
            img[j,k] = pixel_trans[img[j,k]]
    return img

if __name__ == '__main__':
    im= cv2.imread("E:\\image processing\\tiger.jpg", 0)  # 读取灰度图
    plt.subplot(2, 2, 1), plt.imshow(im, cmap='gray')
    plt.title('(a)initial picture'), plt.xticks([]), plt.yticks([])
    prob = get_probability(im)
    plt.subplot(2, 2, 2), plt.bar([i for i in range(256)], prob, width=1)
    plt.title('(b)initial histogram')
    im = equalization(im, prob)
    prob = get_probability(im)
    plt.subplot(2, 2, 4), plt.bar([i for i in range(256)], prob, width=1)
    plt.title('(c)histogram after equalization')
    plt.subplot(2, 2, 3), plt.imshow(im, cmap='gray')
    plt.title('(d)picture after equalization'), plt.xticks([]), plt.yticks([])
    plt.show()

