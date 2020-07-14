import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
              

    


        
def find_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis = 0)
    return centers  


def clasify(x, X, label, K, center):
    m_distance = -1
    index = -1    
    for j, cen in enumerate(center):
        distance = np.sqrt(np.sum((x-cen)*(x-cen)))
        if j == 0:
            m_distance = distance
            index = j
        if distance < m_distance:
            index = j             
            m_distance = distance  
    return index 
def main():
    directory = "D:/data/number_normalize/"
    dem = 0
    X = []
    original_label = []
    for folder in os.listdir(directory):        
        # print(folder)
        for name in os.listdir(directory + "/" + folder):
            image = cv2.imread(directory + "/" + folder + "/" + name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            X.append(gray)
            original_label.append(int(folder))
    X = np.reshape(X, (-1, image.shape[0]*image.shape[1]))
    original_label = np.asarray(original_label)
    index = np.asarray([x for x in range(X.shape[0])])
    index = np.random.choice(index.shape[0], index.shape[0], replace = False)
    # print(index)
    X_train = X[index[:X.shape[0] - 100]]
    lbl_train = original_label[index[:X.shape[0] - 100]]
    X_test = X[index[X.shape[0] - 100:]]
    lbl_test = original_label[index[X.shape[0] - 100:]]

    K = 10
    # print(index)
    # print(X.shape)
    # x = X[1200].copy()
    
    center = find_centers(X_train, lbl_train, K)
    lbl_predict = []
    for x in X_test:
        lbl_predict.append(clasify(x, X, lbl_train, K, center))
    lbl_predict = np.asarray(lbl_predict)
    print(lbl_test)
    print(lbl_predict)
    cm = confusion_matrix(lbl_test, lbl_predict)
    print(cm)
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sn.set(font_scale = 1.4)
    sn.heatmap(df_cm, annot = True)
    plt.show()
    # anh = X_test[0]
    # anh = anh.reshape((image.shape[0],image.shape[1]))
    # cv2.imshow("Image",anh)
    # cv2.waitKey()
    pass
if __name__ == "__main__":
    main()