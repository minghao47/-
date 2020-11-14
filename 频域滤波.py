import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

#频域锐化，高斯高通滤波器
def Gaussian_High_Filter(img,d):
    fft = np.fft.fft2(img)       #进行傅里叶变换
    fft_shift = np.fft.fftshift(fft)   #将图像中的低频部分移动到图像的中心
    def transfer_func(d):
        transfer_func = np.zeros(img.shape)
        s1 = np.log(np.abs(fft_shift))
        center = tuple(map(lambda x:(x-1)/2,s1.shape))
        m,n=transfer_func.shape
        for i in range(m):
            for j in range(n):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center,(i,j))
                transfer_func[i,j] = 1-np.exp(-(dis**2)/(2*(d**2)))
        return transfer_func
    d_matrix = transfer_func(d)
    ifft = np.fft.ifftshift(fft_shift*d_matrix) #将频谱的零频点移动到频谱图的左上角位置
    ifft_shift = np.fft.ifft2(ifft)  #傅里叶反变换
    new_img = np.abs(ifft_shift)
    return new_img

def Image_Enhencement(img0,img1):#图像增强
    m,n=img0.shape
    img2 = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            img2[i,j]=img0[i,j]+img1[i,j]
    return img2    #原始图像+滤波结果


#频域平滑，高斯低通滤波器
def Gaussian_Low_Filter(image,d):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(f)
    def transfer_func(d):
        transfer_func = np.zeros(image.shape)
        center = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfer_func.shape[0]):
            for j in range(transfer_func.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center,(i,j))
                transfer_func[i,j] = np.exp(-(dis**2)/(2*(d**2)))
        return transfer_func
    d_matrix = transfer_func(d)
    ifft = np.fft.ifftshift(fft_shift * d_matrix)  # 将频谱的零频点移动到频谱图的左上角位置
    ifft_shift = np.fft.ifft2(ifft)  # 傅里叶反变换
    new_img = np.abs(ifft_shift)
    return new_img

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
def gasuss_noise(image, mean=0, var=0.01):
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

img = cv2.imread('E:\\image processing\\tiger.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
s1 = np.log(np.abs(fshift))
s3 = np.log(np.abs(f))
plt.subplot(1,2,2),plt.imshow(s1,'gray')
plt.title('Frequency Domain2')
plt.subplot(1,2,1),plt.imshow(s3,'gray')
plt.title('Frequency Domain')
plt.show()

img_d1 = Gaussian_High_Filter(img,10)
img_d2 = Gaussian_High_Filter(img,30)
img_d3 = Gaussian_High_Filter(img,50)
img_e1 = Image_Enhencement(img,img_d1)
img_e2 = Image_Enhencement(img,img_d2)
img_e3 = Image_Enhencement(img,img_d3)
plt.subplot(3,3,1), plt.imshow(img_d1,cmap="gray"),plt.axis("off"),plt.title('(a)sharpening(D0=10) ')
plt.subplot(3,3,2), plt.imshow(img_d2,cmap="gray"),plt.axis("off"),plt.title('(b)sharpening(D0=30) ')
plt.subplot(3,3,3), plt.imshow(img_d3,cmap="gray"),plt.axis("off"),plt.title('(c)sharpening(D0=50) ')
plt.subplot(3,3,4)
plt.axis("off")
plt.imshow(img_e1,cmap="gray")
plt.title('(d)enhencement(D0=10)')
plt.subplot(3,3,5)
plt.axis("off")
plt.title('(e)enhencement(D0=30)')
plt.imshow(img_e2,cmap="gray")
plt.subplot(3,3,6)
plt.axis("off")
plt.title("(f)enhencement(D0=50)")
plt.imshow(img_e3,cmap="gray")

img_d1 = Gaussian_Low_Filter(img,10)
img_d2 = Gaussian_Low_Filter(img,30)
img_d3 = Gaussian_Low_Filter(img,50)
plt.subplot(3,3,7)
plt.axis("off")
plt.imshow(img_d1,cmap="gray")
plt.title('(h)smoothing(D0=10)')
plt.subplot(3,3,8)
plt.axis("off")
plt.title('(i)smoothing(D0=30)')
plt.imshow(img_d2,cmap="gray")
plt.subplot(3,3,9)
plt.axis("off")
plt.title("(j)smoothing(D0=50)")
plt.imshow(img_d3,cmap="gray")
plt.show()


im0 = sp_noise(img,0.02)
plt.subplot(2,2,1)
plt.axis("off")
plt.title("(a)img with salt-pepper noise")
plt.imshow(im0,cmap="gray")

im1 = Gaussian_Low_Filter(im0,50)
plt.subplot(2,2,2)
plt.axis("off")
plt.title("(b)smoothing")
plt.imshow(im1,cmap="gray")

im2 =gasuss_noise (img,0.02)
plt.subplot(2,2,3)
plt.axis("off")
plt.title("(c)img with gasuss noise")
plt.imshow(im2,cmap="gray")

im3 = Gaussian_Low_Filter(im2,50)
plt.subplot(2,2,4)
plt.axis("off")
plt.title("(d)smoothing")
plt.imshow(im3,cmap="gray")

plt.show()