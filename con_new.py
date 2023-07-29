import cv2  # importing necessary packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

widthImg = 800  # width and height of image to be resized
heightImg = 400

r = 2000  # no. of train data
D = pd.read_csv("train_data.csv")  # reading the train data and getting the necessary datas
D = D.values[:10000]
X = D[:, 1:]
Y = D[:, 0]


def preProcessing(img):  # preprocess the image to find the biggest contour
    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # coloured to gray image
    img_Blur = cv2.GaussianBlur(img_Gray, (5, 5), 1)  # blurring thr image
    img_Canny = cv2.Canny(img_Blur, 200, 200)  # noise reduction , dialation
    imgDial = cv2.dilate(img_Canny, np.ones((5, 5)), iterations=2)
    imgThres = cv2.erode(imgDial, np.ones((5, 5)), iterations=1)
    return imgThres


def getContours(img):
    biggest_contour = np.array([])  # list to store the position of the contour
    maxArea = 0  # used to store of areas of contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get the contours in the image

    for cnt in contours:  # traverse through all contour values
        area = cv2.contourArea(cnt)  # find its area
        if area > 5000:
            # if its area > 5000 and is also the greatest contour ( in terms of area ) replace biggest with the
            # current contour positions
            perimeter = cv2.arcLength(cnt, True)
            cnt_current = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if area > maxArea and len(cnt_current) == 4:
                biggest_contour = cnt_current
                maxArea = area
    cv2.drawContours(imgContour, biggest_contour, -1, (255, 0, 0), 20)

    # return the biggest the contour
    return biggest_contour


def reorder(myPoints):
    # reorder the position of the contour such that the image is cropped correctly
    # since the first point has the least sum and the last point in the set of co-ordinates has the maximum sum
    # and in the second set x co-ordinate - y co-ordinate will be lowest compared to that of third set
    # rearranging according to the condition
    if type(myPoints) != list:
        myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img, biggest_contour):  # crop around the biggest contour ( representing the cheque )
    biggest_contour = reorder(biggest_contour)  # reorder the set of co-ordinates
    pts1 = np.float32(biggest_contour)  # convert the list values to float type
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # co-ordinates of the total image
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # change the perspective
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # warping
    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]  # get the cropped image
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))  # resize to required dimensions

    # return the cropped image
    return imgCropped


def preprocess(img):  # process the image ( to detect the number )
    k = img  # processing as explained previously
    k = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)
    k = cv2.bitwise_not(cv2.threshold(k, 100, 255, cv2.THRESH_BINARY)[1])
    resized = cv2.resize(k, [28, 28], interpolation=cv2.INTER_AREA)
    img = np.asarray(resized).reshape(-1)
    return img


def get_info(img, info):  # function to get the date and money int he cheque

    rf = RandomForestClassifier(n_estimators=125, max_depth=100, min_samples_split=20)
    rf.fit(X, Y)
    # get the shape of the image
    h, w, c = img.shape
    nums = list()

    # if dat is being read
    if info == "date":
        for i in range(8):  # get all the images corresponding each number in the date

            # since all numbers are treated equally divide the date section to 8 column and get the image of all digits
            img_new = getWarp(img, np.array(
                [[2 + i * w / 8 + 2, h], [2 + i * w / 8 + 2, 2], [(i + 1) * w / 8 - 2, 2], [(i + 1) * w / 8 - 2, h]]))

            cv2.imwrite(f"dat{i}.jpg", img_new)  # write the image ( optional )
            img_new = preprocess(img_new)  # process the image and send it into the classifier
            num = rf.predict(img_new.reshape(1, -1))
            nums.append(num)  # add to the number to the date
        date = ""
        for j in range(8):
            if j == 2 or j == 4:  # change the numbers obtained to date using "," when needed
                date = date + "," + str(nums[j])
            else:
                date = date + str(nums[j])
        print("Dated for : ", date)

    elif info == "money":
        for i in range(8):  # get all the images corresponding each number in the amount of money
            if i != 7:
                # since all numbers are treated equally divide the date section to 8 column and get the image of all digits
                img_new = getWarp(img, np.array(
                    [[i * w / 3.4 * 0.4 + 2, h], [i * w / 3.4 * 0.4 + 2, 0], [(i + 1) * w / 3.4 * 0.4 - 4, 0],
                     [(i + 1) * w / 3.4 * 0.4 - 4, h]]))
            else:
                # for the last digit read till the end
                img_new = getWarp(img, np.array(
                    [[i * w / 3.4 * 0.4 + 2, h], [i * w / 3.4 * 0.4 + 2, 0], [w - 2, 0], [w - 2, h]]))
            cv2.imwrite(f"mon{i}.jpg", img_new)  # write the image ( optional )
            # process the image and send it into the classifier
            img_new = preprocess(img_new)
            num = rf.predict(img_new.reshape(1, -1))
            nums.append(num)  # add to the number to cost
        money = ""
        for j in range(7):
            if j == 2 or j == 4:
                money = money + "," + str(nums[j])  # change the numbers obtained to money using "," when needed
            else:
                money = money + str(nums[j])
        print("Money is : ", money)


img = cv2.imread("c11.jpg")  # read the image of the cheque
img = cv2.resize(img, (widthImg, heightImg))  # resize the image
imgContour = img.copy()

imgThres = preProcessing(img)  # preprocess the image

# get the biggest contour in the image ( the points corresponds to the posititon of
# cheque in the screen since it will be the largest object in the screen )
biggest_contour = getContours(imgThres)

# crop the image to the cheque
imgWarped = getWarp(img, biggest_contour)

# if the image of the cheque is scaled to the specified size
# then the posititon of the date and money section is constant
# so the images can be cropped again and sent into the get_infos function to get the information
date_pic = getWarp(imgWarped, np.array([[615, 15], [615, 40], [777, 15], [777, 40]]))
cv2.imwrite("dates.png", date_pic)
get_info(date_pic,"date")

money_pic = getWarp(imgWarped, np.array([[617, 171], [617, 138], [779, 139], [778, 174]]))
cv2.imwrite("mon.jpg", money_pic)
get_info(money_pic, "money")
money_pic = preprocess(money_pic)

