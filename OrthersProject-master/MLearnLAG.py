#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:56:21 2017

@author: 390771
"""
import tensorflow as tf
import numpy as np
class MLearnLAG():

    def __init__(self,inputTrainPath,inputTrainTarget,inputTestPath,inputTestTarget,modeName,outPutModePath,outPutModeEvelPath):
        self.__inputTrainPath = inputTrainPath
        self.__inputTrainTarget = inputTrainTarget
        self.__inputTestPath = inputTestPath
        self.__inputTestTarget = inputTestTarget
        self.__outPutModePath = outPutModePath
        self.__outPutModeEvelPath = outPutModeEvelPath
        self.__modeName = modeName
        self.__modeNameMethod = modeName['method']
        self.__modeNamehideLayerNum = modeName['hideLayerNum']
        self.__modeNamehideNum = modeName['hideNum']
        
    def proData(self):
        print  'this is proData !'
        
    def train(self):
        #重定向方法
        mode = self.methods_to_ref()
        if type(mode) == str:
            return 'error 0  :input The modeName is error !'
        methodModeName = mode()
        print  'this is trainData !'
        return methodModeName
    
    def fit(self):
        print 'this is fit !'
    def evluateMode(self):
        print 'evluateMode suc !'
        
    def methods_to_ref(self):
        switch = {
                u'nn': self.tensorflowNN
                }
        return switch.get(self.__modeNameMethod,"nothing")    
    
    def tensorflowNN(self):
        learning_rate = 0.001
#        input x,y
        train_x = tf.placeholder("float", shape = self.__inputTrainPath.shape)
        train_y = tf.placeholder("float",shape = self.__inputTrainTarget.shape)
        
        Wrows = self.__inputTrainPath.shape[1]
        W = range((self.__modeNamehideLayerNum+1))
        b = range((self.__modeNamehideLayerNum+1))
        WxB = range((self.__modeNamehideLayerNum+1))
        y = range((self.__modeNamehideLayerNum+1))
        for index in range(self.__modeNamehideLayerNum):
            #隐含层数量
            hideNum = self. __modeNamehideNum[index]
            #set var W and b
            W[index] = tf.Variable(tf.zeros([Wrows,hideNum]))
            b[index] = tf.Variable(tf.zeros([hideNum]))
            WxB[index] = tf.add(tf.matmul(train_x,W[index]),b[index])
            y[index] = tf.nn.relu( WxB[index] )
            #update inputs
#            assign_op = tf.assign(train_x,y[index] )
            train_x = y[index] 
            Wrows =hideNum
            
        W[self.__modeNamehideLayerNum] = tf.Variable(tf.zeros([Wrows,hideNum]))
        b[self.__modeNamehideLayerNum] = tf.Variable(tf.zeros([hideNum]))
        WxB[self.__modeNamehideLayerNum] = tf.add(tf.matmul(train_x,W[self.__modeNamehideLayerNum]),b[self.__modeNamehideLayerNum])
        output = tf.nn.sigmoid( WxB[self.__modeNamehideLayerNum] )
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=train_y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # Initializing the variables
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            sess.run([W,b,cost,optimizer],feed_dict={train_x:self.__inputTrainPath,train_y:self.__inputTrainTarget})
            #  model
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(train_y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({train_x:self.__inputTestPath,train_y:self.__inputTestTarget}))
        
        print 'this is nn !'
#        return 'suc'
        return accuracy        
    
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data=load_iris()
    train_x  = data['data'][:130]
    train_y  = data['target'][:130]
    test_x = data['data'][130:]    
    test_y = data['target'][130:]
    modeName = dict()
    modeName['method'] = 'nn'
    modeName['hideLayerNum'] = 2 
    modeName['hideNum']=np.array([7,5])
    ML = MLearnLAG(train_x,train_y,test_x,test_y,modeName,r'/home/390771/',r'/home/390771/')
    accuracy =   ML.train()
    
    
    
#    import tensorflow as tf
#    sess = tf.InteractiveSession()
#    x = tf.placeholder(tf.float32, shape=[None, 784])
#    y_ = tf.placeholder(tf.float32, shape=[None, 10])
#    W = tf.Variable(tf.zeros([784,10]))
#    b = tf.Variable(tf.zeros([10]))
#    sess.run(tf.global_variables_initializer())
#    y = tf.matmul(x,W) + b
