import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as ds
import math

r = 20000 # no. of train data
data = pd.read_csv("train_data.csv")
Data = data.values[0:r]
Pix = Data[:,1:]
Lab = Data[:,0]

def euc_dist(d1,d2):                 
    dis = 0                          
    
    for i in range(len(d1)):         
         dis += (d1[i]-d2[i])**2
    dis = math.sqrt(dis)             
    return dis

def knnf(train_x,train_y,test,n):    
    train_y = np.array(train_y)
    dist = []
    for i in range(len(train_x)):
        dist.append(euc_dist(train_x[i],test))
        
    args = np.argsort(dist)          
    train_y = train_y[args]
    train_y = train_y[:n]            
    return np.bincount(train_y).argmax()

# Test = pd.read_csv("test_data.csv")
# T = Test.values[:10]
# for i in range(len(T)):
#     label = knnf(Pix,Lab,T[i],5)
#     img = np.reshape(T[i],(28,28))
#     plt.imshow(img)
#     plt.show()
#     print("Number is: ",label)

# for any arbitary input
def classifier(img):
    print("Hello I am here")
    k = img
    k = cv2.cvtColor(k,cv2.COLOR_BGR2GRAY)
    k = cv2.bitwise_not(cv2.threshold(k, 100, 255, cv2.THRESH_BINARY)[1])
    resized = cv2.resize(k,[28,28], interpolation = cv2.INTER_AREA)
    cv2.imwrite("ok1.png",resized)
    img = np.asarray(resized).reshape(-1)
    num = knnf(Pix,Lab,img,5)
    return num
