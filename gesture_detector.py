#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import os
from utils import clPreProcessing
from utils import clTraningSetManager
from utils import ContourDetector
from utils import clAutoCalibrate


class clHogDetector:

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

# main loop
if __name__ == "__main__":

    # process cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, required=True, metavar='labelfile', help='label file, like ./label.txt')
    parser.add_argument('-t', type=str, required=False, metavar='trainedfile',
                        help='file which holds the trained values like, ./data.xml')
    parser.add_argument('-cmd', type=str, metavar='commands', dest='choises',
                        choices=['retrain', 'savedetections', 'creatlabelfile', 'run'], default='run',
                        help='commands available for processing: [retrain, savedetections, creatlabelfile, run], default=run')
    parser.add_argument('-d', type=int, metavar='detectionswindow', default=0, choices=[0,1],
                        help='show/ hide detection window [0,1], default=0')
    parser.add_argument('-c', type=int, metavar='cameraid', default=0, choices=[0,1],
                        help='use different camera ID, default=0')
    parser.add_argument('-s', type=int, metavar='samplesize', default=96, choices=[32,64,96,192],
                        help='select the training sample size, default=96')
    parser.add_argument('-dir', type=str, required=False, metavar='traindirectory', help='directory with ordered pictures per detection classes')
    parser.add_argument('-dd', type=str, required=False, metavar='detectiondir', help='directory to save the detections or image patches for training set creation')
    parser.add_argument('-ani', type=int, required=False, metavar='annotatedimages', default=0, choices=[0,1],
                        help='save annotated images, [0,1] default=0')
    parser.add_argument('-sf', type=int, required=False, metavar='saveframe', default=0, choices=[0,1],
                        help='save original camera frames, [0,1] default=0')
    parser.add_argument('-sp', type=int, required=False, metavar='savepatches', default=0, choices=[0,1],
                        help='save image detections (patches), [0,1] default=0')

    parser.add_argument('-cal', type=int, required=False, metavar='calibration', default=0, choices=[0,1],
                        help='reclalibrate skin color detection, [0,1] default=0')

    args = parser.parse_args()
    cmd = args.choises

    # define image dimensions
    IMG_WIDTH = 320
    IMG_HEIGHT = 240
    CAMID = args.c
    SAMPLESIZE = args.s

    labelfile = args.l
    trainedfile = args.t
    traindir = args.dir
    detectionsdir = args.dd

    # create named window, set position
    cv2.namedWindow('img', 2)
    cv2.moveWindow('img', 0, 0)

    # create cam instance
    cam0 = cv2.VideoCapture(CAMID)

    # resize, to spare CPU load
    cam0.set(3, IMG_WIDTH)
    cam0.set(4, IMG_HEIGHT)

    # create empty images
    img0 = np.zeros((IMG_WIDTH,IMG_HEIGHT,3),dtype=np.byte)
    imgd = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)

    # pre-processing
    #objPP = clPreProcessing(img0, False, 155, 30, 100)
    objPP = clPreProcessing(img0, False, 150, 66, 66)

    # contour detector
    cd = ContourDetector()

    # object for data set handling
    tsm = clTraningSetManager()

    # skin color autocalibration
    ac = clAutoCalibrate()


    if args.cal == 1:
        #load calibration values from a file
        val = tsm.LoadLabelsFile(labelfile,True)
        objPP.SetColorFilteringThresholds(int(val[0]), int(val[1]), int(val[2]))

    if cmd == 'run':
        if labelfile is None:
            print ("Labels file is missing, use -h for available arguments")
            os._exit(0)

        # load labels file
        lf = tsm.LoadLabelsFile(labelfile)

        if trainedfile is None:
            print ("Training file is missing, use -h for available arguments")
            os._exit(0)
        #load trained file
        det = clHogDetector(SAMPLESIZE,trainedfile)
        det.UpdateLabelNames(lf)

    elif cmd =='retrain':

        if labelfile is None:
            print("Labels file is missing, use -h for available arguments")
            os._exit(0)

        if trainedfile is None:
            print ("Trained file is missing, use -h for available arguments")
            os._exit(0)

        det = clHogDetector(SAMPLESIZE)

        # load labels file
        lf = tsm.LoadLabelsFile(labelfile)

        # load training set
        for i in lf:
            uid = i[0]
            label = i[1]
            path = i[2]
            det.AddToTrainingSet(path, int(uid), label)

        # update label names
        det.UpdateLabelNames(lf)

        # train SVM
        det.TrainSVMWithHOG(SAMPLESIZE)

        # save trained file
        det.SaveTrainingData(trainedfile)

    elif cmd == 'creatlabelfile':
        if traindir is None:
            print("Argument with training directory is missing, use -h for available arguments")
            os._exit(0)
        else:
            tsm.SaveLabelsFile(traindir, labelfile)
            print ("Labelfile " + labelfile + " created, exiting.")
            os._exit(0)

    elif cmd == 'savedetections':
        # show helper windows and save detections to a folder
        if detectionsdir is None:
            print("Argument with detections directory is missing, use -h for available arguments")
            os._exit(0)
        else:
            print("Detections will be save in: " + detectionsdir)
            args.d = 1

    while (True):

        _, img0 = cam0.read()

        # test cam instances
        if (cam0):

            img = objPP.processImg(img0)
            img = objPP.processImg1(img)
            aa = cd.CotourFilter(img,500.0)

            rois = cd.GetRoiForDetections(img0,aa,0)

            imgd = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3],dtype=np.uint8)
            imgd = cd.ShowRoisOnImage(imgd,rois)

            val = det.ClassifyRoi(rois, SAMPLESIZE)
            if args.cal is 0:
                img = cd.DrawDetections(img0, aa, 0, True, True, val)
            else:
                img = ac.RunCalibration(img0)

            if detectionsdir is not None:
                if args.sp == 1:
                    cd.SaveImages(img0, rois,detectionsdir)

                if args.ani == 1:
                    cd.SaveImages(img, [], detectionsdir,prefix="ani_")

                if args.sf == 1:
                    cd.SaveImages(img0, [], detectionsdir,prefix="sf_")

            cv2.imshow('img', img)
            #cv2.imshow('imgx',imgx)
            if args.d == 1:
                cv2.imshow('imgd', imgd)


        k = cv2.waitKey(1) & 0xFF

        # quit on keypress
        if k == ord('q'):
          break

        # calibrate
        if k == ord('c'):
            if args.cal is not 0:
                # save calibration values
                val = ac.ProvideClaibParams()
                objPP.SetColorFilteringThresholds(int(val[0]), int(val[1]), int(val[2]))

                tsm.SaveCalibration(labelfile, val)

                #exit
                break

    # release cam
    cam0.release()
    cv2.destroyAllWindows()

