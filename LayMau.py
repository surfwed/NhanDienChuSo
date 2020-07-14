import cv2
import matplotlib.pyplot as plt
import numpy as np



def click(event, x, y, flags, param):
    global gx, gy
    if event == cv2.EVENT_LBUTTONDOWN:
        gx = x
        gy = y
        return 0

def main():

    global gx, gy
    nPositionX = 80
    nPositionY = 33
    gx = nPositionX
    gy = nPositionY
    nGridX = 1
    nGridY = 1
    nPaddingX = 0
    nPaddingY = 0
    nGridHeight = 49
    nGridWidth = 76
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)

    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    index_anh = 0
    v1 = 9
    index = 401
    while True:
        image = cv2.imread('D:/data/image_' + str(index_anh) + '.jpg')
        image = cv2.resize(image, (1080, 720))
        original_image = image.copy()

        image = original_image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        # ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY_INV, v1, 2)

        nPositionX = gx
        nPositionY = gy        
        
        for x in range(nGridX):
            for y in range(nGridY):        
                px = nPositionX + x * (nGridWidth + nPaddingX)
                py = nPositionY + y * (nGridHeight + nPaddingY)
                cv2.rectangle(image, (px, py), (px + nGridWidth, py + nGridHeight), (0, 0, 255), 1)              
        
        
        thresh = cv2.erode(thresh, np.ones((1,1), dtype = np.uint8), iterations = 3)
        thresh = cv2.dilate(thresh, np.ones((1,1), dtype = np.uint8), iterations = 3)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_grille = None
        maxArea = 0
        cnts = []
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2 )
        # for c in contours:
        #     area = cv2.contourArea(c)
        #     if area > 2000 and area < 10000:
        #         peri = cv2.arcLength(c, True)
        #         polygone = cv2.approxPolyDP(c, 0.1 * peri, True)
        #         if len(polygone) >= 4 and len(polygone) <= 5:
        #             cnts.append(polygone)
        #         if area > maxArea and len(polygone) == 4:
        #             contour_grille = polygone
        #             maxArea = area
        cv2.imshow("Thresh", thresh)
        # if contour_grille is not None:
        #     cv2.drawContours(image, [contour_grille], 0, (0, 255, 0), 2 )
        if len(cnts) != 0:
            cv2.drawContours(image, cnts, -1, (0, 255, 0), 2 )
        
        
            
        txt = "ADAPTIVE_THRESH_MEAN_C" if method == cv2.ADAPTIVE_THRESH_MEAN_C else "ADAPTIVE_THRESH_GAUSSIAN_C"
        cv2.putText(image, "[m|p]v1: {:2d} [o]method: {}".format(v1, txt), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 
        1)
        
        cv2.putText(image, "image_{}".format(index_anh), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 
        1)
        cv2.imshow("Image",image)
        key =  cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            for cnt in cnts:                        
                x, y, z, t = cnt[0][0], cnt[1][0], cnt[2][0], cnt[3][0]            
                cv2.circle(image, (x[0], x[1]), 3, (51, 51, 153), -1) # Nau
                cv2.circle(image, (y[0], y[1]), 3, (102, 255, 255), -1) # vang
                cv2.circle(image, (z[0], z[1]), 3, (0, 51, 255), -1) # Do
                cv2.circle(image, (t[0], t[1]), 3, (255, 102, 0), -1) # Xanh
                x1 = min(x[0], y[0], z[0], t[0])
                y1 = min(x[1], y[1], z[1], t[1])
                x2 = max(x[0], y[0], z[0], t[0])
                y2 = max(x[1], y[1], z[1], t[1])
                # cv2.imshow(""+str(count), thresh[y1+10:y2-10,x1+10:x2-10].copy())
                cv2.imwrite("D:/data/number/"+str(index)+".png", thresh[y1+5:y2-5,x1+5:x2-5].copy())
                index+=1
            pass
        if key == ord('c'):
            px = nPositionX 
            py = nPositionY 
            crop = thresh[py:py+nGridHeight,px:px+nGridWidth].copy()                
            cv2.imwrite("D:/data/number/"+str(index)+".png", crop) 
            index+=1
        if key == ord('q'):
            break
        if key == ord('p'):
            v1 = min(21, v1 + 2)            
        if key == ord('m'):
            v1 = max(3, v1 - 2)
        if key == ord('n'):
            index_anh = min(17, index_anh + 1)
        if key == ord('b'):
            index_anh = max(0, index_anh - 1)
        # print(v1)
        if key == ord('o'):
            if method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
                method = cv2.ADAPTIVE_THRESH_MEAN_C
            else:
                method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    cv2.destroyAllWindows()
    print("Exit")

if __name__ == "__main__":
    main()