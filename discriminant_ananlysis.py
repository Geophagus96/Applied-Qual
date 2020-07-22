# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 03:54:50 2020

@author: Yuze Zhou
"""

import numpy as np

class Discriminant_Analysis:
    def __init__(self, classify_order):
#    If the classify_order is assigned a value 1, Linear Discriminat Analysis is adopted; If the 
#    classify order is 2, then quadratic Discriminant Analysis is adopted
        if (classify_order != 1)&(classify_order != 2):
            raise ValueError('Not a valid classification creterion')
        self.classify_order = classify_order
    
#    The function of obtaining the estimated covariance and mean of each class based on the training
#    set, the labels of the training set shall start from 1 with consecutive integers     
    def train(self, X_train, train_label):
#    Training process for LDA
#    Description for class attributes:
#    class_size: total number of groups for LDA
#    data_dimension: the dimension of the data
#    class_cov: the commonly-shared covaraince for all groups
#    class_mean: individual mean value for each group
#    class_proportion: proportions of each group among all observations
        if (self.classify_order == 1):
            self.class_size = train_label.max()
            self.data_dimension = int(X_train.shape[1])
            self.class_cov = np.zeros([self.data_dimension, self.data_dimension])
            self.class_mean = np.zeros([self.class_size, self.data_dimension])
            self.proportion = np.zeros(self.class_size)
            for i in range(0,self.class_size):
                train_subset = X_train[(train_label == (i+1)),]
                subset_size = train_subset.shape[0]
                self.class_mean[i,] = np.mean(train_subset, 0)
                self.class_cov = self.class_cov + (subset_size-1)*np.cov(train_subset.T)
                self.proportion[i] = subset_size
            self.class_cov = self.class_cov/(X_train.shape[0]-self.class_size)
#    Training process for QDA
#    Description for class attributes:
#    class_cov: a 3-d array stroing the individual covariance for each group
#    other attributes remain the same as LDA
        elif (self.classify_order == 2):
            self.class_size = train_label.max()
            self.data_dimension = int(X_train.shape[1])
            self.class_mean = np.zeros([self.class_size, self.data_dimension])
            self.proportion = np.zeros(self.class_size)
            self.class_cov = np.zeros([self.data_dimension,self.data_dimension,self.class_size])
            for i in range(0,self.class_size):
                train_subset = X_train[(train_label == (i+1)),]
                subset_size = train_subset.shape[0]
                self.class_mean[i,] = np.mean(train_subset, 0)
                self.class_cov[:,:,i] = np.cov(train_subset.T)
                self.proportion[i] = subset_size
                
            
#    The function of obtaining the log-likelihood of each observation in the test set for each group
#    the corresponding group they are being assigned to            
    def test(self, X_test):
#    Testing process for LDA
#    Description for class attributes:
#    Likelihood: the log-likelihood of each observations in the test set for each group
        if (self.classify_order == 1):
            likelihood = np.zeros([X_test.shape[0],self.class_size])
            for i in range(0, self.class_size):
                mean_row = np.reshape(np.repeat(self.class_mean[i,],X_test.shape[0]),(self.data_dimension,X_test.shape[0]))
                class_likelihood = -np.sum((np.dot((X_test-mean_row.T),np.linalg.inv(self.class_cov))*(X_test-mean_row.T)),axis=1)+np.log(self.proportion[i])
                likelihood[:,i] = class_likelihood
            self.likelihood = likelihood
#    Testing process for QDA
#    Description for class attributes is the same LDA
        elif (self.classify_order == 2):
            likelihood = np.zeros([X_test.shape[0], self.class_size])
            for i in range(0,self.class_size):
                mean_row = np.reshape(np.repeat(self.class_mean[i,],X_test.shape[0]),(self.data_dimension,X_test.shape[0]))
                class_likelihood = -np.sum(np.dot((X_test-mean_row.T),np.linalg.inv(self.class_cov[:,:,i]))*(X_test-mean_row.T),axis=1)+np.log(self.proportion[i])
                likelihood[:,i] = class_likelihood
            self.likelihood = likelihood
        return (np.argmax(likelihood,axis=1)+1)


                