#图像的空域平滑、锐化
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

#求均值
def get_average(img,x,y,step):
    ans=0
    for i in range(x-int(step/2),x+int(step/2)+1):
        for j in range(y-int(step/2),y+int(step/2)+1):
            ans+=img[i][j]            #求像素周围像素的均值
    return int(ans/step/step)         #返回均值
#空间平滑，均值滤波
def average_filter(img,step):
    m,n=img.shape                     #图像长和宽的像素数
    a = np.zeros((m, n), dtype="uint8")
    for i in range(m):                #解决图像边缘像素问题
        for j in range(n):
            if i-int(step/2)<0 or i+int(step/2)>=m:
                a[i][j] = img[i][j]
            elif j-int(step/2)<0 or j+int(step/2)>=n:
                a[i][j] = img[i][j]
            else:
                a[i][j] = get_average(img,i,j,step)
    return a

#空间锐化,拉普拉斯算子
def laplacian_filter(img):
    img0=np.array(img)
    m,n=img0.shape
    img1=np.zeros((m,n))
    for i in range(2,m-1):
        for j in range(2,n-1):
            img1[i,j]=abs(int(img[i+1,j])+int(img[i-1,j])+int(img[i,j-1])+int(img[i,j+1])-int(4*img[i,j]))  # 拉普拉斯算子锐化图像，用二阶微分
    img_laplace=np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            img_laplace[i][j] = img_laplace[i][j] + img[i][j]   #将原始图像和拉普拉斯图像叠加在一起的简单方法可以保护拉普拉斯锐化处理的效果，同时又能复原背景信息
    return img1,img_laplace

#添加椒盐噪声
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
#添加高斯噪声
def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

im=cv2.imread('E:\\image processing\\tiger.jpg',0)    #读取图片并取灰度
im0= sp_noise(im,0.02)
im1=gasuss_noise(im)
im2=average_filter(im0,3)     #三位均值滤波器
im3=average_filter(im0,5)     #五位均值滤波器
im4,im5=laplacian_filter(im)   #拉普拉斯算子锐化
im6=average_filter(im1,3)     #三位均值滤波器
im7=average_filter(im1,5)     #五位均值滤波器
plt.subplot(3, 3, 1), plt.imshow(im , cmap='gray')
plt.title('(a)initial picture'), plt.xticks([]), plt.yticks([])   #原始图片
plt.subplot(3, 3, 2), plt.imshow(im4, cmap='gray')
plt.title('(b)laplacian filter'), plt.xticks([]), plt.yticks([])     #拉普拉斯算子锐化
plt.subplot(3, 3, 3), plt.imshow(im5, cmap='gray')
plt.title('(c)laplacian filter+initial img'), plt.xticks([]), plt.yticks([])     #空间锐化结果+原图
plt.subplot(3, 3, 4), plt.imshow(im0, cmap='gray')
plt.title('(d)img with salt-pepper noise'), plt.xticks([]), plt.yticks([])   #加噪图片
plt.subplot(3, 3, 5), plt.imshow(im2 , cmap='gray')
plt.title('(e)average filter 3'), plt.xticks([]), plt.yticks([])     #空间平滑结果1
plt.subplot(3, 3, 6), plt.imshow(im3, cmap='gray')
plt.title('(f)average filter 5'), plt.xticks([]), plt.yticks([])     #空间平滑结果2
plt.subplot(3, 3, 7), plt.imshow(im1, cmap='gray')
plt.title('(g)img with gasuss noise'), plt.xticks([]), plt.yticks([])   #加噪图片
plt.subplot(3, 3, 8), plt.imshow(im6 , cmap='gray')
plt.title('(h)average filter 3'), plt.xticks([]), plt.yticks([])     #空间平滑结果1
plt.subplot(3, 3, 9), plt.imshow(im7, cmap='gray')
plt.title('(i)average filter 5'), plt.xticks([]), plt.yticks([])     #空间平滑结果2
plt.show()               #显示实验结果