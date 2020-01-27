import cv2 as cv2
import numpy as np
import os
from datetime import datetime

###########################################################
# With help of this class the correct HSV values ca be
# selected from a picture color space
# this values can be further used i.e. for skin detection.
# Track-bars will appear if debug mode is activated!
###########################################################
class clPreProcessing():

    def __init__(self,img, debug=True, h=122,s=22,v=0):
        self.img = img
        self.debug = debug
        self.h = h
        self.s = s
        self.v = v

        if self.debug == True:
            # create trackbars for color change
            cv2.createTrackbar('h', 'img', 0, 255, self.nothing)
            cv2.createTrackbar('s', 'img', 0, 255, self.nothing)
            cv2.createTrackbar('v', 'img', 0, 255, self.nothing)

    def nothing(self,x):
        pass

    def SetColorFilteringThresholds(self, h,s,v):
        '''
        Set the new hue, saturation, value from HSV color space
        :param h: new hue value from HSV space
        :param s: new saturation rom HSV space
        :param v: new value from HSV space
        :return:
        '''

        self.h = h
        self.s = s
        self.v = v

    def processImg(self, img):
        '''

        :param img: input img
        :return: processed img
        '''

        if self.debug == True:
            self.h = cv2.getTrackbarPos('h', 'img')
            self.s = cv2.getTrackbarPos('s', 'img')
            self.v = cv2.getTrackbarPos('v', 'img')

        # smooth the image
        self.img = cv2.medianBlur(img, 7)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # set colr space domain
        lower_skin = np.array([self.h, self.s, self.v])
        upper_skin = np.array([self.h + 80, 255, 255])

        # filter HSV image
        mask = cv2.inRange(self.img, lower_skin, upper_skin)

        # apply mask on the original image
        self.img = cv2.bitwise_and(img, img, mask=mask)

        return self.img

    def processImg1(self, img):
        '''
        Prost processing for image
        :param img: input image
        :return: processed image
        '''

        kernel = np.ones((5, 5), np.uint8)
        # smooth the image
        self.img = cv2.medianBlur(img, 11)

        self.img = cv2.dilate(self.img, kernel, iterations=7)

        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)

        return self.img

###############################################################################
#Provide easy access functions to create / save / load labels and training sets
###############################################################################
class clTraningSetManager():


    def __init__(self):
        pass

    def ReadTrainingDirectory(self,dir):
        '''
        :param: dir: directory, to the training data
        :return: unique ID, directory name as label, directory
        '''

        pathlist = []
        labellist = []
        count = 0
        labelsAndIDs = []

        # get all directories
        for root, dirs, files in os.walk(dir):
            pathlist.append(root)
            labellist.append(dirs)

        if len(labellist) > 0:
            # just first level is needed
            labellist = labellist[0]

            # first level of paths (root) not needed
            pathlist.pop(0)

            #create list
            for label, dir in zip(labellist, pathlist):
                count += 1
                labelsAndIDs.append(str(count) + "," + label + "," + dir)
        else:
            print("Wrong path!")

        return labelsAndIDs

    def SaveLabelsFile(self, dir, file):
        '''
        :param dir: directory, to the training data
        :param file: name of the labels file
        :return: none
        '''

        lai = self.ReadTrainingDirectory(dir)

        # save data into the file
        file = open(file, "w")

        for i in lai:
            file.write(i + "\n")
            pass
        file.close()

    def SaveCalibration(self,labelfile,calval=[]):
        '''
        Add calibration values to label file
        :param labelfile: label file
        :return: none
        '''

        #read all file
        with open(labelfile, "r") as f:
            lines = f.readlines()

        # write lines without calibration values
        with open(labelfile, "w") as f:
            for line in lines:
                if line.find('#cal,') == -1:
                    f.write(line)

        #add to the end the values
        with open(labelfile, "a") as f:
            h = str(calval[0])
            s = str(calval[1])
            v = str(calval[2])
            ws = str(h + ","+ s +"," + v)
            f.write("#cal," + ws)




    def LoadLabelsFile(self, file,calibration=False):
        '''
        :param file: name of the labels file
        :return: array of unique identifier, dir name as label, dir names
        '''

        lf = []
        with open(file) as fp:
            for cnt, line in enumerate(fp):
                if calibration == False:
                    # split data, skip what if needed
                    if line.find('#') == -1:
                        res = [x.strip() for x in line.split(',')]
                        lf.append(res)
                else:
                    if line.find('#cal,') != -1:
                        #get calibration values
                        res = [x.strip() for x in line.split(',')]
                        h = res[1]
                        s = res[2]
                        v = res[3]
                        lf = np.array([h,s,v])

        return lf

    def GetItemsByPosition(self,labellist, id=1):
        '''
        :param labellist: labels file
        :param id, position to read = [0,1,2]
        :return: array
        '''

        return [i[id] for i in labellist]

###############################################################################
# Get contours of a pre-processed image
###############################################################################
class ContourDetector():
    def __init__(self):
        self.gimg = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.items = []
        self.count = 0

    def CotourFilter(self,img, area=1000.0):
        '''
        Used to filter the contours
        :param img: input image for detecting contours
        :param area: minimum aria to detect, skipp smaller ones
        :return: list of detected countours (ID, coordinates)
        '''

        self.items = []
        self.gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(self.gimg, 1, 2)

        id = 1
        for cnt in contours:
            larea = cv2.contourArea(cnt)
            if larea > area:

                x, y, w, h = cv2.boundingRect(cnt)

                center = (int(x+(w/2)), int(y+(h/2)))

                self.items.append(np.array([id, x, y, w, h, center]))

                id +=1
        return self.items

    def DrawDetections(self, img, detections, offset =20, objCenter=True,objRectangle=True, label=[0,0]):
        '''
        Draw detected contours on image
        :param img: input image
        :param detections: list of detected contours (ID, coordinates)
        :param offset: offset to increase detected area
        :param objCenter: draw object center
        :param objRectangle: draw object contour
        :param label: label name
        :return: labeled image
        '''
        # copy image
        limg = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

        # check if there are detections
        if len(detections) > 0:
            for ii,ll in zip(detections,label):
                id, x, y, w, h, center = ii
                if objCenter == True:
                    cv2.circle(limg, center, 2, (0, 255, 0), -1)
                if objRectangle == True:
                    cv2.rectangle(limg, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 1)
                    if ll[0] == id:
                        cv2.putText(limg, str(ll[1]), (x-offset+5, y-offset+15), self.font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        return limg

    def GetRoiForDetections(self, img, detections, offset=20, roi_size=(96, 96)):
        '''
        Get patches from an image
        :param img: input image
        :param detections: list of detected ROIs
        :param offset: detection offset
        :param roi_size: ROI size
        :return: return image patches
        '''

        rois = []
        #check if there are detections
        if len(detections) > 0:
            for ii in detections:
                id, x, y, w, h, center = ii

                try:
                    detRoi = img[y - offset:y + h + offset, x - offset:x + w + offset]
                    detRoi = cv2.normalize(detRoi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    detRoi = cv2.resize(detRoi, roi_size)

                    rois.append(detRoi)
                except:
                    pass
        return rois

    def ShowRoisOnImage(self, img, rois, roi_size=(96, 96)):
        '''
        Helper function, draw  ROIs on blank image
        :param img: imput image
        :param rois: ROIs
        :param roi_size: size
        :return: show roi on image
        '''

        if len(rois) > 0:
            offs = 0
            try:
                for ii in rois:
                    ii = np.uint8((ii + 1) * 255 / 2)
                    img[0:roi_size[0], 0 + offs: 0 + roi_size[0] + offs] = ii
                    offs += roi_size[0]
            except:
                pass


        return img

    def SaveImages(self,img, rois, path=".", initnumber=0, usetime=True,saveframe=False,prefix=""):
        '''
        Used to create training sets, save the detected ROIs, as image patches
        :param img: canera inage
        :param rois: detections
        :param path: location to save the patches
        :param initnumber: append a specific number to the image name
        :param usetime: append to image name the current time
        :return:
        '''
        time = datetime.now().strftime("%H%M%S")

        if len(rois) > 0:
            for ii in rois:
                ii = (ii + 1) * 255 / 2
                if usetime == True:
                    cv2.imwrite(path + "img_" + prefix + str(time) + ".png", ii)
                else:
                    cv2.imwrite(path + "img_" + prefix +str(self.count + initnumber) + ".png", ii)
                self.count +=1
        else:
            cv2.imwrite(path + "img_"+ prefix + str(time) + ".png", img)


class clAutoCalibrate:
    def __init__(self):
        self.img = []
        self.h = 0
        self.s = 0
        self.v = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def ProvideClaibParams(self):
        return (self.h,self.s,self.v)

    def RunCalibration(self, img):

        x = 10; y = 10;
        w = 80 ; h = 80;

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img = cv2.rectangle(img, ( 8, 8), (90, 90), (0, 255, 0), 2)

        detRoi = hsv[y:y + h, x:x + w]

        channels = cv2.split(detRoi)

        self.h = int(np.average(np.array(channels[0])))
        self.s = int(np.average(np.array(channels[1])))
        self.v = int(np.average(np.array(channels[2])))

        img[10:90,10:90] = detRoi
        msg = "hsv=" + str(self.h)+"," + str(self.s)+"," + str(self.v)
        cv2.putText(img,msg, (5,105), self.font, 0.4, (0, 255, 0), 1,cv2.LINE_AA)

        return img