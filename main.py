import cv2
import matplotlib.pyplot as plt
import numpy as np



def click(event, x, y, flags, param):
    global gx, gy
    if event == cv2.EVENT_LBUTTONDOWN:
        gx = x
        gy = y
        return 0

def gauss(image):       
    filter_gauss = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype = np.uint32)
    image = np.array(image, dtype = np.uint32)
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if x - 1 < 0 or y - 1 < 0 or x + 1 >= image.shape[1] or y + 1 >= image.shape[0]:
                continue
            
            a = image[y-1:y+2,x-1:x+2] * filter_gauss
            image[y, x] = np.sum(a) / np.sum(filter_gauss)
    image = np.array(image, dtype = np.uint8)
    return image

def main():
    name_title = ['zero','one','two','three','four','five','six','seven','eight','nine']
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)
    anh = []
    nPositionX = 80
    nPositionY = 33
    global gx, gy
    gx = nPositionX
    gy = nPositionY
    nGridX = 1
    nGridY = 1
    nPaddingX = 0
    nPaddingY = 0
    nGridHeight = 35
    nGridWidth = 35
    index = 142
    index_anh = 0
    threshold = 150
    image = cv2.imread('D:/data/image_0.jpg')
    image = cv2.resize(image, (720, 480))
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     
    number = 0
    # gray = gauss(gray)
    idx = gray > threshold
    gray[idx == True] = 255
    gray[idx == False] = 0
    original_gray = gray.copy()
    isGaussed = False
    crop = []
    while (True):           
        image = original.copy() 
        
        nPositionX = gx
        nPositionY = gy        
        
        for x in range(nGridX):
            for y in range(nGridY):        
                px = nPositionX + x * (nGridWidth + nPaddingX)
                py = nPositionY + y * (nGridHeight + nPaddingY)
                cv2.rectangle(image, (px, py), (px + nGridWidth, py + nGridHeight), (0, 0, 255), 1)
                crop = gray[py:py+nGridHeight,px:px+nGridWidth].copy()                
              
        k = cv2.waitKey(1) & 0xFF
        process_image = False

        cv2.putText(image, "Thresh={0}".format(threshold), (10, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (209, 80, 0, 255), 1)
        cv2.putText(image, "So={0}".format(number), (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (209, 80, 0, 255), 1)
        
        cv2.imshow("Image", image)
        cv2.imshow("Gray", gray)
                
        if k == ord('q'):
            break
        elif k == ord('p'):
            index_anh +=1
            if index_anh > 17:
                index_anh = 17
            image = cv2.imread('D:/data/image_' + str(index_anh) + '.jpg')
            image = cv2.resize(image, (720, 480))
            original = image.copy()
            process_image = True
        elif k == ord('o'):
            index_anh -=1
            if index_anh < 0:
                index_anh = 0
            image = cv2.imread('D:/data/image_' + str(index_anh) + '.jpg')
            image = cv2.resize(image, (720, 480))
            original = image.copy()
            process_image = True
        elif k == ord('t'):
            process_image = True
            threshold -= 1
        elif k == ord('y'):
            threshold += 1
            process_image = True
        elif k == ord('g'):
            if isGaussed:
                gray = original_gray.copy()
            else:
                gray = gauss(gray)            
            isGaussed = not isGaussed
        elif k in [ord(str(num)) for num in range(10)]:
            number = k - ord('0')
            pass
        elif k == ord('s'):
            dir = 'D:/data/' + str(number) + '/' + str(index) + '.png'
            cv2.imwrite(dir, crop)
            index+=1

        if process_image:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)                    
            # gray = gauss(gray)
            idx = gray > threshold
            gray[idx == True] = 255
            gray[idx == False] = 0
            original_gray = gray.copy()
    print("Exit")

if __name__ == "__main__":
    main()