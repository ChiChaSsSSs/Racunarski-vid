import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import imutils

def findEdgePixel(img) -> tuple:
    h, w = img.shape[0:2]
    for i in range(0, h):
        for j in range(0, w):
            # Optimizacija da ne idem kroz deo slike koji sigurno ne sadrzi okvir
            if j > w - 1440:
                break
            # Ako su svi pikseli u matrici 2x3 takvi da im je zbir bela boja onda je to trazeni piksel
            if int(img[i, j]) + int(img[i, j + 1]) + int(img[i + 1, j]) + int(img[i + 1, j + 1]) + int(img[i, j + 2]) + int(img[i + 1, j + 2]) > 1020:
                return(i, j)

# Funkcija koja vrsi skaliranja ulazne slike i vraca ulaznu sliku u razlicitim formatima
# Ovo je neophodno kako bi mogli da detektujemo trazene objekte u razlicitim velicinama      
def pyramidOfImages(img, scale=2, minSize=(180,180)):
    yield img
    while True:
        w = int(img.shape[1] / scale) # Racunamo novu sirinu slike na osnovu koeficijenta skaliranja
        img = imutils.resize(img, width=w) # Srazmerno smanjujemo sliku
        if img.shape[0] < minSize[1] or img.shape[1] < minSize[0]: # Kada dodjemo do minimalno smanjene slike prekidamo smanjivanje
            break
        yield img

# Funkcija za implementaciju sliding prozora
# Prolazimo kroz celu sliku img, po visini i sirini, povecajuci se za odabran korak
# U svakom prolazu uzimamo samo onaj deo slike odredjen velicinom prozora
def slidingWindow(img, stepSize, windowSize):
    for y in range(0, img.shape[0], stepSize):
	    for x in range(0, img.shape[1], stepSize):
		    yield (x, y, img[y:y + windowSize[1], x:x + windowSize[0]])

if __name__ == "__main__":
    img = cv.imread("nekemacke.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Konvertuje se slika u gray da bi lakse detektovao pocetak okvira
    cv.imshow("Entry image", img)
    cv.waitKey(0)

    edgePixel = findEdgePixel(gray) # Pronalazim gornje levo teme okvira
    croppedImage = img[edgePixel[0]: edgePixel[0] + 720, edgePixel[1]: edgePixel[1] + 1440]
    cv.imshow("Cropped image", croppedImage)
    cv.waitKey(0)
    
    rows = open('synset_words.txt').read().strip().split("\n") # Otvaramo fajl sa imenima zivotinja
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows] # Izdvajamo imena zivotinja
    
    for resizedImage in pyramidOfImages(croppedImage): # Prolazimo kroz sve skalirane slike
        for (x, y, window) in slidingWindow(resizedImage,stepSize=180,windowSize=(180,180)): # Prolazimo kroz svaki sliding window
            if window.shape[0] != 180 or window.shape[1] != 180: # Ako trenutni prozor ne odgovara velicini sliding prozora
                continue
            scale=croppedImage.shape[0]//resizedImage.shape[0] # Trenutno skaliranje
            
            clone = resizedImage.copy()
            
            blob = cv.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123)) # Iz trenutne slike window se kreira niz podataka koji se moze proslediti neuronskoj mrezi
            net = cv.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel') # Ucitava se neuronska mreza
            net.setInput(blob) # Prosledjivanje podataka ucitanoj mrezi
            preds = net.forward() # Pokrece se neuronska mreza unapred i racuna se izlaz kao verovatnoca poklapanja sa razlicitim slikama
            idxs = np.argsort(preds[0])[::-1][:5] # Sortiraju se i selektuju 5 klasa koje najverovatnije zadovoljavaju uslov
            
            cv.rectangle(clone,(x,y),(x+180,y+180),(204,255,255))
            #cv.imshow('Resized image',clone)
            #cv.waitKey(0)
            
            for (i, idx) in enumerate(idxs): # Prolazimo kroz sve selektovane klase
                if i == 0 and preds[0][idx]>=0.7: # Ovime selektujemo samo one predikcije vece od 0.75 kako bismo bili sigurni da detektujemo dobar objekat na slici
                        if 'cat' in classes[idx]: # Ako je u selektovanim klasama macka
                            print("MACKA")
                            text = "CAT"
                            cv.putText(croppedImage, text, (x*scale+5, y*scale+25),  cv.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                            cv.rectangle(croppedImage,(x*scale,y*scale),((x+180)*scale,(y+180)*scale),(0,0,255))
                        elif 'dog' in classes[idx]: # Ako je u selektovanim klasama pas
                            print("PAS")
                            text = "DOG"
                            cv.putText(croppedImage, text, (x*scale+5, y*scale+25),  cv.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
                            cv.rectangle(croppedImage,(x*scale,y*scale),((x+180)*scale,(y+180)*scale),(0,255,255))
                            
cv.imwrite('output.jpg',croppedImage)
cv.imshow("Final image" ,croppedImage)
cv.waitKey(0)           
    
    

    
    
    