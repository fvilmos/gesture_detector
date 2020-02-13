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

    def __init__(self,img, debug=True, hmin=77,smin=133,vmin=27, hmax=189,smax=170,vmax=153):
        '''
        Set the new hue, saturation, value from a color space
        :param hmin: new value
        :param smin: new value
        :param vmin: new value
        :param hmax: new value
        :param smax: new value
        :param vmax: new value
        :return: processed image
        '''
        self.img = img
        self.debug = debug
        self.hmin = hmin
        self.smin = smin
        self.vmin = vmin
        self.hmax = hmax
        self.smax = smax
        self.vmax = vmax

        if self.debug:
            # create trackbars for color change
            self.hmin = cv2.createTrackbar('hmin', 'img',hmin,255, self.nothing)
            self.smin = cv2.createTrackbar('smin', 'img',smin,255, self.nothing)
            self.vmin = cv2.createTrackbar('vmin', 'img',vmin,255, self.nothing)
            self.hmax = cv2.createTrackbar('hmax', 'img',hmax,255, self.nothing)
            self.smax = cv2.createTrackbar('smax', 'img',smax,255, self.nothing)
            self.vmax = cv2.createTrackbar('vmax', 'img',vmax,255, self.nothing)

    def nothing(self,x):
        pass

    def SetColorFilteringThresholds(self, hmin,smin,vmin,hmax,smax,vmax):
        '''
        Set the new hue, saturation, value from a color space
        :param hmin: new value
        :param smin: new value
        :param vmin: new value
        :param hmax: new value
        :param smax: new value
        :param vmax: new value
        :return: processed image
        '''
        self.hmin = hmin
        self.smin = smin
        self.vmin = vmin
        self.hmax = hmax
        self.smax = smax
        self.vmax = vmax

    def processHSV(self, img,rmask=False,val=[0,100,0,20,255,255,135,31,18,180,255,255]):
        '''

        :param img: input img
        :param rmask return just image mask
        :param val holds the init values for the color pace
        :return: processed img
        '''

        if self.debug == True:
            self.hmin = cv2.getTrackbarPos('hmin', 'img')
            self.smin = cv2.getTrackbarPos('smin', 'img')
            self.vmin = cv2.getTrackbarPos('vmin', 'img')
            self.hmax = cv2.getTrackbarPos('hmax', 'img')
            self.smax = cv2.getTrackbarPos('smax', 'img')
            self.vmax = cv2.getTrackbarPos('vmax', 'img')

        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([val[0],val[1],val[2]])
        upper_skin = np.array([val[3],val[4],val[5]])

        mask1 = cv2.inRange(self.img,lower_skin,upper_skin)

        lower_skin = np.array([val[6],val[7],val[8]])
        upper_skin = np.array([val[9],val[10],val[11]])

        mask2 = cv2.inRange(self.img, lower_skin, upper_skin)

        mask = cv2.bitwise_or(mask1,mask2)

        kernel = np.ones((3, 3), np.uint8)
        kernel1 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel1)

        # apply mask on the original image
        self.img = cv2.bitwise_and(img, img, mask=mask)

        if rmask:
            self.img = mask

        return self.img

    def processFilter(self, img):
        '''
        Prost processing for image
        :param img: input image
        :return: processed image
        '''

        kernel = np.ones((5, 5), np.uint8)
        # smooth the image
        self.img = cv2.medianBlur(img, 7)

        self.img = cv2.dilate(self.img, kernel, iterations=5)

        self.img = cv2.morphologyEx(self.img, cv2.MORPH_ELLIPSE, kernel)

        return self.img

    def processYCrBr(self, img, rmask=False,val=[27,133,28,135,167,145,13,38,20,37,22,12]):
        '''

        :param img: input img
        :param rmask return just image mask
        :param val holds the init values for the color pace
        :return: processed img
        '''

        if self.debug == True:
            self.hmin = cv2.getTrackbarPos('hmin', 'img')
            self.smin = cv2.getTrackbarPos('smin', 'img')
            self.vmin = cv2.getTrackbarPos('vmin', 'img')
            self.hmax = cv2.getTrackbarPos('hmax', 'img')
            self.smax = cv2.getTrackbarPos('smax', 'img')
            self.vmax = cv2.getTrackbarPos('vmax', 'img')

        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        lower_skin = np.array([val[0],val[1],val[2]])
        upper_skin = np.array([val[3],val[4],val[5]])

        mask1 = cv2.inRange(self.img,lower_skin,upper_skin)

        lower_skin = np.array([val[6],val[7],val[8]])
        upper_skin = np.array([val[9],val[10],val[11]])

        mask2 = cv2.inRange(self.img, lower_skin, upper_skin)

        mask = cv2.bitwise_or(mask1,mask2)

        kernel = np.ones((3, 3), np.uint8)
        kernel1 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel1)

        # apply mask on the original image
        self.img = cv2.bitwise_and(self.img, self.img, mask=mask)

        if rmask:
            self.img = mask

        return self.img

    def CombineDetections(self, img):
        '''

        :param img:
        :return: combined HSV and yCrCb skin color detection
        '''

        #get hsv mask
        hsvm = self.processHSV(img,True)

        #get YCrBr mask
        ycrbrm = self.processYCrBr(img, True)

        #cobine masks
        masks = cv2.bitwise_or(hsvm,ycrbrm)

        # smooth the output
        masks = cv2.medianBlur(masks, 23)

        self.img = cv2.bitwise_and(img, img, mask=masks)

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

    def DrawDetections(self, img, detections, offset =20, objCenter=True,objRectangle=True, label=[0,0], drawline=False):
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

        arr = []
        # check if there are detections
        if len(detections) > 0:
            for ii,ll in zip(detections,label):
                id, x, y, w, h, center = ii
                if objCenter:
                    cv2.circle(limg, center, 2, (0, 255, 0), -1)
                if objRectangle:
                    cv2.rectangle(limg, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 1)
                    if ll[0] == id:
                        cv2.putText(limg, str(ll[1]), (x-offset+5, y-offset+15), self.font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                        arr.append(center)

        arr = np.array(arr)
        if len(arr)>2 and drawline == True:
            cv2.drawContours(limg, [arr], 0, (255, 0, 255), 3)

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


class clHogDetector:
    '''
    Makes all the machine lerning part
    '''
    def __init__(self, sampleSize=64, fn=""):
        winSize = (sampleSize, sampleSize)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64

        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                     histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        self.templates = []
        self.HogAccumulator = []
        self.LabelList = []

        self.svm = cv2.ml.SVM_create()
        # carete SVM
        if fn =="":
            pass
        else:
            self.svm = cv2.ml.SVM_load(fn)

        # n-class classification
        self.svm.setType(cv2.ml.SVM_C_SVC)

        # Binary classification (detections belong to one or other class)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)

        # termination criteria
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    def AddLabelAndFiles(self, FileName, label,strlabel):
        '''
        Caate label, ald files list
        :param FileName: name of the image file
        :param label: cirresponding label
        :param strlabel: lternative, string name
        :return: none
        '''
        self.templates.append((int(label), FileName, strlabel))


    def GetHogForAnImage(self, img, sampleSize=64):
        '''
        Compute HOG feature
        :param img: input image
        :param sampleSize: image path size
        :return: hog feature for the images
        '''
        resized = cv2.resize(img, (sampleSize, sampleSize), interpolation=cv2.INTER_AREA)

        ret = self.hog.compute(resized)

        return ret

    def ImgFloatToInt(self, img):
        '''
        Convert float to int values in image
        :param img: input image
        :return: integer image
        '''
        intimg = np.uint8((img + 1) * 255 / 2)

        return intimg

    def GetImagesHOGFeatures(self, sampleSize=64):
        '''
        Get hog feature for all files from training directory
        :param sampleSize: patch size
        :return: none
        '''

        for i in range(0, len(self.templates)):
            # get files from disk
            img = cv2.imread(self.templates[i][1], 0)

            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img = np.uint8((img + 1) * 255 / 2)

            # resize it
            resized = cv2.resize(img, (sampleSize, sampleSize), interpolation=cv2.INTER_AREA)

            ret = self.hog.compute(resized)

            self.HogAccumulator.append([self.templates[i][0], ret, self.templates[i][2]])


    def GetHogAccumulator(self):
        '''
        Return HOG list
        :return:
        '''
        return self.HogAccumulator

    def UpdateLabelNames(self,labels):
        '''
        Create str Label list
        :param labels: list of labels, IDs, and paths
        :return: none
        '''
        for id,l,dir in labels:
            self.LabelList.append((id,l))


    def TrainSVMWithHOG(self, sampleSize=64):
        '''
        rain SVM
        :param sampleSize: image patch size
        :return: none
        '''

        # compute training set hog features and add to accumulator
        self.GetImagesHOGFeatures(sampleSize=sampleSize)

        trainingData = []
        trainingCalss = []

        for i in range(0, len(self.HogAccumulator)):
            pn, hogv, strlabel = self.HogAccumulator[i]
            hogv = np.array(hogv.T, dtype=np.float32)

            trainingData.append(hogv.T)
            trainingCalss.append(pn)

        trainingData = np.array(trainingData, dtype=np.float32)
        trainingCalss = np.array(trainingCalss, dtype=np.int)

        self.svm.train(trainingData, cv2.ml.ROW_SAMPLE, trainingCalss)

    def ReadTrainingFiles(self, dir):
        '''
        Read images paths from training directory
        :param dir: traning directori for a single label
        :return: list of image paths
        '''
        import os
        paths = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                path = (os.path.join(root, file))
                #print (path)
                paths.append(path)
        return paths

    def AddToTrainingSet(self, dir, label,strlabel=""):
        '''
        Add trinaing set
        :param dir: image locations
        :param label: labels
        :param strlabel: alternative name of labes
        :return:
        '''

        f = self.ReadTrainingFiles(dir)
        for a in f:
            self.AddLabelAndFiles(a,label, strlabel)

    def ClassifyRoi(self,imgs=None, sampleSize=64):
        '''
        Classify a single image patch with SVM
        :param imgs: array of patches
        :param sampleSize: patch size
        :return: detection list
        '''

        arrFound = []
        id = 0
        for r in imgs:
            # hystogram normalization
            bw = cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
            bw = self.ImgFloatToInt (bw)
            ival = cv2.equalizeHist(bw)
            ihog = self.GetHogForAnImage(ival, sampleSize=sampleSize)
            ihog = np.array([ihog], dtype=np.float32)

            id +=1
            val = self.svm.predict(ihog)

            val = int(val[1][0][0])
            val = self.GetStrLabelByClassificationID(val)
            arrFound.append((id,val))

        return arrFound

    def GetStrLabelByClassificationID(self,val):
        '''
        Get Str Label name
        :param val: SVM integer class
        :return: str Label
        '''
        ret = val
        for v,l in self.LabelList:
            if v == str(val):
                if l == "":
                    ret = v
                else:
                    ret= l
                break

        return ret


    def SaveTrainingData(self,fn="./data.xml"):
        '''
        Save trained SVM values
        :param fn: file name
        :return: none
        '''
        self.svm.save(fn)
