import cv2
import numpy as np
import matplotlib.pyplot as plt
    
def ukloniSum(img, r, prag):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if (x-256)**2 + (y-256)**2 > r**2 and (img[x,y]) > prag:
                img[x,y] = 0
                print(x,y)
    return img
    
if __name__ == '__main__':
    sumSlika = cv2.imread("./slika_1.png")
    sumSlika = cv2.cvtColor(sumSlika, cv2.COLOR_BGR2GRAY)
    plt.imshow(sumSlika, cmap='gray')
    plt.title("Pocetna slika")
    plt.show()
    
    ft = np.fft.fft2(sumSlika)
    ft = np.fft.fftshift(ft)
    
    complexModuo = ft / np.abs(ft)
    magPre = np.log(np.abs(ft))
    plt.imshow(magPre, cmap='gray')
    plt.title("Magnituda pre uklanjanja suma")
    plt.show()

    magPosle = ukloniSum(magPre, 10, 0.8 * np.max(magPre))
    plt.imshow(magPosle, cmap='gray')
    plt.title("Magnituda posle uklanjanja suma")
    plt.show()
    
    bezSuma = complexModuo * np.exp(magPosle)
    bezSuma = np.abs(np.fft.ifft2(bezSuma))
    plt.imshow(bezSuma, cmap='gray')
    plt.title("Finalna slika")
    plt.savefig("finalnaslika")
    plt.show()
    