import cv2
import numpy as np
import matplotlib.pyplot as plt

def pronadjiKrugove(slika):
    visina, sirina = slika.shape[:2]
    krugovi = []
    for i in range(visina):
        for j in range(sirina):
            if slika[i,j] == 255 and proveriKrug(i, j, krugovi):
                pom = i + 1
                br = 0
                while br < 10:
                    if slika[pom, j] == 0:
                        br += 1
                    else:
                        br = 0
                    pom += 1
                r = int((pom - 10 - i) / 2)
                krugovi.append((i + r, j, r))

    return krugovi
         
def proveriKrug(i, j, krugovi):
    for k in krugovi:
        if (i - k[0])**2 + (j - k[1])**2 <= (1.15*k[2])**2:
           return False
    
    return True

def pronadjiNovcic(krugovi):
    for k in krugovi:
        cy, cx, r = k
        br = 0
        for i in range(cy - r, cy + r):
            for j in range(cx - r, cx + r):
                if (i - cy)**2 + (j - cx)**2 <= (0.9*k[2])**2 and popunjenaSlika[i,j] == 0:
                    br += 1
                if br > 10:
                    return k
    return None
    
def napraviMasku(slika, trazeniNovcic):
    visina, sirina = slika.shape[:2]
    maska = np.zeros((visina, sirina), dtype=np.uint8)
    cy, cx, r = trazeniNovcic
    
    for i in range(cy - r, cy + r):
        for j in range(cx - r, cx + r):
            if (i - cy)**2 + (j - cx)**2 <= (0.95*r)**2:
                maska[i,j] = 255
    return maska
        
if __name__ == '__main__':
    ulaznaSlika = cv2.imread("coins.png")
    plt.imshow(ulaznaSlika)
    plt.title("Ulazna slika")
    plt.show()
    ulaznaSlikaGray = cv2.cvtColor(ulaznaSlika, cv2.COLOR_BGR2GRAY)
    
    _, pragovanaSlika = cv2.threshold(ulaznaSlikaGray, 165, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(pragovanaSlika, cmap='gray')
    plt.title("Pragovana slika")
    plt.show()
    
    popunjenaSlika = cv2.morphologyEx(pragovanaSlika, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
    plt.imshow(popunjenaSlika, cmap='gray')
    plt.title("Popunjena slika")
    plt.show()
    
    krugovi = pronadjiKrugove(popunjenaSlika)
    trazeniNovcic = pronadjiNovcic(krugovi)
    maska = napraviMasku(popunjenaSlika, trazeniNovcic)
    plt.imshow(maska, cmap='gray')
    plt.title("Maska")
    plt.show()
    
    finalnaSlika = cv2.bitwise_and(ulaznaSlika, ulaznaSlika, mask=maska)
    finalnaSlikaRGB = cv2.cvtColor(finalnaSlika, cv2.COLOR_BGR2RGB)
    plt.imshow(finalnaSlikaRGB)
    plt.title("Finalna slika")
    plt.show()  