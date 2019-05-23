#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))

def load_data(csv_filename):
    """ 
    Returns a numpy ndarray in which each row repersents
    a wine and each column represents a measurement. There should be 11
    columns (the "quality" column cannot be used for classificaiton).
    """
    #print("FILENAME:", csv_filename)
    
    np.set_printoptions(suppress=True)
    
    #f = open(csv_filename,'r')
    #for i in range(10):
        #print(f.readline().strip())
    
    #this is numpy of data from file with the last element (quality) in it that we don't want
    data_wi_qual = np.genfromtxt(csv_filename, delimiter=';', skip_header=1) #skip header means skip the first line
   
    
    #this is numpy of data sliced without the last column
    actual_data = data_wi_qual[:, 0:data_wi_qual.shape[1]-1]
    
    #printing out the data, mainly for debugging
   # print(actual_data)
    
    #print("Shape of actual data", actual_data.shape)
    
    return actual_data
 
    
def split_data(dataset, ratio = 0.9):
    """
    Return a (train, test) tuple of numpy ndarrays. 
    The ratio parameter determines how much of the data should be used for 
    training. For example, 0.9 means that the training portion should contain
    90% of the data. You do not have to randomize the rows. Make sure that 
    there is no overlap. 
    """
    #create the index that splits into respective ratios
    split_index = int(ratio * dataset.shape[0])
    
    #print("the index to split is", split_index)
    
    test = dataset[:split_index,:] #first split become test set
    train = dataset[split_index:,:] #last split becomes training data 
    
    #return ratio of test & train
    return (test, train) 


def compute_centroid(data):
    """
    Returns a 1D array (a vector), representing the centroid of the data
    set. 
    """
    
    #don't need to do uncessary slicing because using all these elements
    return sum(data) / len(data) #this is the average of all dimensions 

    
def experiment(ww_train, rw_train, ww_test, rw_test):
    """
    Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy. 
    """
#    print("entering experiment")
#    print("ww_train size is", ww_train.shape)
#    print("rw_train size is", rw_train.shape)
#    print("ww_test size is", ww_test.shape)
#    print("rw_test size is", rw_test.shape)
    
    #compute the centroid of white wine and red wine
    white_centroid = compute_centroid(ww_train)
    red_centroid = compute_centroid(rw_train)
    
    #now test it on the new data
    correct = 0
    wrong = 0
    
    #print("iterating through white flowers")
    #iterating through all the rows with the data in WHITE FILE
    for row in ww_test:
        #white flower distance
        dist_white = euclidean_distance(row, white_centroid) #calcualtes the euclidean_distance with distance formula
        #red flower distance
        dist_red = euclidean_distance(row, red_centroid)
        #if closer to white
        if(dist_white < dist_red):
            #then the prediction = white, so it is correct
           # print("prediction: white, CORRECT")
            correct += 1
        elif(dist_red > dist_white):
           # print("prediction: red, WRONG")
            wrong += 1
    
    #print("iterating through red flowers")
    #iterating through all the rows with the data in RED FILE
    for row in rw_test:
        #white flower distance
        dist_white = euclidean_distance(row, white_centroid) #calcualtes the euclidean_distance with distance formula
        #red flower distance
        dist_red = euclidean_distance(row, red_centroid)
        #if closer to red
        if(dist_red < dist_white):
            #then the prediction = red, so it is correct
           # print("prediction: red, CORRECT")
            #increment correct
            correct += 1
        #closer to white
        elif(dist_red > dist_white):
          #  print("prediction: white, WRONG")
            wrong += 1
    
    total_predictions = ww_test.shape[0] + rw_test.shape[0]
    
    accuracy = correct / total_predictions
    
    print("Total predictions:", total_predictions)
    
    print("Correct predictions:", correct)
    
    print("Accuracy:", accuracy)
    
    return accuracy 
 
    
def cross_validation(ww_data, rw_data, k):
    """
    Perform k-fold crossvalidation on the data and print the accuracy for each
    fold. 
    One common solution to this problem is to split the available data into
    k-partitions of equal size. We perform k training/testing repetitions.
    In each repetition we test on one of the k partitions after training on 
    the remaining k-1 partitions. In this way, each of the partitions will
    be tested on once. We can then average the accuracy we got for each of 
    the k repetitions. This is called k-fold cross validation.
    """
    k_accuracy_list = []
    
    partition_size = ww_data.shape[0]/k
    
    #print("partition_size is", partition_size)

    #iterate through each partition (which is k)
    for i in range(k):
        #print("partition_size is", partition_size)

        #print which partition it is
        #print("Partition index", i, ":")
        
        #print("i is", i)
        #set index to start based on partiion #
        start = int(i * partition_size)
        
        #print("start is", start)
        #because range doesn't include the last one
        
        end = int(start + partition_size + 1)
        
        #print("end is", end)
        #create list of indexes in the data you are going to iterate over
        #indexes = [k for k in range(start, end)]
        
        #indexing it to just be the partition
        kww_data = ww_data[start:end, :]
        #print("kww_data is", kww_data)
        krw_data = rw_data[start:end, :]
        #print("krw_data is", krw_data)

        
        #splits the data into test and train
        k_ww_train, k_ww_test = split_data(kww_data, 0.9)
        k_rw_train, k_rw_test = split_data(krw_data, 0.9)
    
         #append the ratio to list
        k_ratio = experiment(k_ww_train, k_rw_train, k_ww_test, k_rw_test)
        k_accuracy_list.append(k_ratio)
        
        print("Partition", i, "accuracy is :", k_ratio)
        
        #finally calcualte average of the accuracy list and return the ratio
    
    final_accuracy = sum(k_accuracy_list) / len(k_accuracy_list)
    
    print("Accuracy of all k partitions is:", final_accuracy)
    
    return final_accuracy
        
if __name__ == "__main__":
    
    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')
    
    #to test if shapes are the same
    #print("SHAPES:")
    #print(ww_data.shape)
    #print(rw_data.shape)

    # Uncomment the following lines for step 2: 
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    experiment(ww_train, rw_train, ww_test, rw_test)
    
    # Uncomment the following lines for step 3:
    k = 10
    acc = cross_validation(ww_data, rw_data, k)
    print("{}-fold cross-validation accuracy: {}".format(k,acc))
    
