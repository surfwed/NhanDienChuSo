import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    directory = "D:/data/number/"
    dem = 0
    for folder in os.listdir(directory):
        dem = 0
        # print(folder)
        for name in os.listdir(directory + "/" + folder):
            image = cv2.imread(directory + "/" + folder + "/" + name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # print(image.shape)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            # thresh = cv2.dilate(thresh, np.ones((3,3), dtype = np.uint8), iterations = 1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            if (len(contours) != 0):
                number_contour = max(contours, key=cv2.contourArea)
            
            temp = np.zeros_like(gray)
            cv2.drawContours(temp, [number_contour], 0, 255, -1)
            temp = cv2.bitwise_and(thresh, temp)
            # temp = cv2.dilate(temp, np.ones((1,1), dtype = np.uint8), iterations = 1)
            contours, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            if (len(contours) != 0):
                number_contour = max(contours, key=cv2.contourArea)
            temp2 = np.zeros_like(gray)
            cv2.drawContours(temp2, [number_contour], 0, 255, -1)
            temp = cv2.bitwise_and(temp, temp2)

            contours, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            if (len(contours) != 0):
                number_contour = max(contours, key=cv2.contourArea)
            
            x = tuple(number_contour[number_contour[:, :, 0].argmin()][0])
            y = tuple(number_contour[number_contour[:, :, 0].argmax()][0])
            z = tuple(number_contour[number_contour[:, :, 1].argmin()][0])
            t = tuple(number_contour[number_contour[:, :, 1].argmax()][0])            
            x1 = min(x[0], y[0], z[0], t[0])
            y1 = min(x[1], y[1], z[1], t[1])
            x2 = max(x[0], y[0], z[0], t[0])
            y2 = max(x[1], y[1], z[1], t[1])
            wb = 70
            hb = 70
            board = np.zeros((hb,wb))
            py1 = int(hb / 2) - int((y2 - y1) / 2)
            py2 = py1 + (y2 - y1)
            px1 = int(wb / 2) - int((x2 - x1) /2)
            px2 = px1 + (x2 - x1)
            board[py1:py2,px1:px2] = temp[y1:y2,x1:x2].copy()
            add = "D:/data/number_normalize/"+ folder+ "/" + str(dem) + '.png'
            # board = cv2.resize(board, (28,28))
            # //cv2.imshow(name, board)
            cv2.imwrite(add, board)
            dem+=1
            # break
    cv2.waitKey()
    pass
if __name__ == "__main__":
    main()