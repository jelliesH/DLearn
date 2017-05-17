#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:25:21 2017

@author: 390771
"""
import os
import pydub
import python_speech_features as psf
import matplotlib.pyplot as plt
import numpy as np
import sys

#path = r'/home/390771/Data/noise/audiolib/errNoise/99_4名义制冷压缩机42Hz机组左面近压缩机有嗡嗡声.WAV'
#sound = pydub.AudioSegment.from_wav(path)
#os.path.exists(path)
##sound = pydub.AudioSegment.from_mp3(path)
#data = sound.get_array_of_samples()
#nordata = 2*(data-min(data))/(max(data)-min(data))-1
#fs = sound.frame_rate
#c = np.array(data.tolist())
#pm = psf.mfcc(c,fs)
#
#path1 = r'/home/390771/Data/noise/audiolib/errNoise/9_GMV-H335WLA多联机电机声.mp3'
#sound1 = pydub.AudioSegment.from_mp3(path)
#data1 = sound1.get_array_of_samples()
#fs1 = sound1.frame_rate
#c1 = np.array(data1.tolist())
#pm1 = psf.mfcc(c1,fs)
#hh = np.vstack((pm1,pm))
class Pretreatment():
    def __init__(self,audioPath,outputPath):
        self.__audioPath = audioPath
        self.__outputPath = outputPath
        self.__audioPathlist = os.listdir(audioPath)

    def getAudioPath(self):
        audioPath = self.__audioPath
        return audioPath

    def getOutputPath(self):
        print self.__outputPath

    def writeToTxt(self,list_name,file_path):
        fp = open(file_path, "w+")
        for item in list_name:
            fp.write(str(item) + "\n")  # list中一项占一行
        fp.close()

    def exactMFCC(self,fileName):
        pathName = os.path.join(self.__audioPath,fileName)
        print 'pathName is ',pathName
        if fileName.find(".mp3") != -1 or fileName.find(".MP3") != -1:
            try:
                sounds = pydub.AudioSegment.from_mp3(pathName)
            except:
                return 'error !'
        elif fileName.find(".wav") != -1 or fileName.find(".WAV") != -1:
            try:
                sounds = pydub.AudioSegment.from_wav(pathName)
            except:
                return 'error !'        
        else :
            return 'error !'
        datas = sounds.get_array_of_samples()
        frame_rate = sounds.frame_rate
        datasArrary = np.array(datas.tolist())
        psMFCC = psf.mfcc(datasArrary,frame_rate)
        temp = psMFCC.tolist()
        file_path = os.path.join(self.__outputPath,fileName)
        if fileName.find(".mp3") != -1 or fileName.find(".MP3") != -1:
            outPutMFCCFile = fileName.replace(".mp3", '.txt')
            outPutMFCCFile = outPutMFCCFile.replace(".MP3", '.txt')
        elif fileName.find(".wav") != -1 or fileName.find(".WAV") != -1:
            outPutMFCCFile = fileName.replace(".wav", '.txt')
            outPutMFCCFile = outPutMFCCFile.replace(".WAV", '.txt')
        else :
            return 'error !'
        outPutMFCCFile_path = os.path.join(self.__outputPath,outPutMFCCFile)
        # print 'The fileName is ', outPutMFCCFile_path
        self.writeToTxt(temp,outPutMFCCFile_path)
        print 'writeToTxt suc !'
        return psMFCC

    def catMFCCFile(self):
        arr = np.array(range(13))
#        psMFCC = self.exactMFCC(self.__audioPathlist[1])
#        print self.__audioPathlist[1]
#        arr = np.vstack((psMFCC, arr))
#        outPutMFCCFileTotalpath = os.path.join(self.__outputPath, 'a.txt')
#        self.writeToTxt(arr,outPutMFCCFileTotalpath)
#        print outPutMFCCFileTotalpath,'Total suc !'
        for audioName in self.__audioPathlist:
            if audioName.find(".mp3") != -1 or audioName.find(".MP3") != -1 or audioName.find(".WAV") != -1 or audioName.find(".wav") != -1:
                psMFCC = self.exactMFCC(audioName)
                if type(psMFCC) == str:
                    continue
                arr = np.vstack((psMFCC,arr))
            else:
                continue
        outPutMFCCFileTotalpath = os.path.join(self.__outputPath, 'finel.txt')
        self.writeToTxt(arr, outPutMFCCFileTotalpath)
        return arr

if __name__ == "__main__":
    
    path = r'/home/390771/Data/noise/audiolib/norNoise'
    outputPath = r'/home/390771/cc/norNoise'
    preNorNoise = Pretreatment(path,outputPath)
    arr = preNorNoise.catMFCCFile()
    
    classLabelOne = np.ones(len(arr))
    LabelOneData = np.c_[arr,classLabelOne]
    
    errNoisePath = r'/home/390771/Data/noise/audiolib/errNoise'
    outErrNoisePutPath = r'/home/390771/cc/errNoise'
    preNorNoise = Pretreatment(errNoisePath,outErrNoisePutPath)
    psErrMFCC = preNorNoise.catMFCCFile()
    
    classLabelTwo = np.zeros(len(arr))
    LabelTwoData = np.c_[arr,classLabelTwo]
    
    train = np.vstack((LabelOneData,LabelTwoData))
    print 'LabelOneData is ',sys.getsizeof(LabelOneData)/1024,'byte.  ','LabelTwoData is ',sys.getsizeof(LabelTwoData)/1024,'byte.  '
    print 'train data is ',sys.getsizeof(train)/1024,'byte . '
    
    binaryData = {}
    binaryData['data'] = np.vstack((arr,psErrMFCC))
    binaryData['label'] = np.vstack((classLabelOne,classLabelTwo))
    binaryData['title'] = u'Noise Nor-Error Databases'
    binaryData['detail'] = u'the true is normal display 1, ortherwise display 0'
    
    import pickle
    output = open('binaryData.pkl','wb')
    pickle.dump(binaryData,output)
    output.close()
    