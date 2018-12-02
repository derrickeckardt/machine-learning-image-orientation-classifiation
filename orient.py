#!/usr/bin/python
#
# ./orient.py : Perform Machine Learning Image Orientation, training usage:
#     ./orient.py train train_file.txt model_file.txt [model]

# ./orient.py : Perform Machine Learning Image Orientation, testing usage:
#     ./orient.py test test_file.txt model_file.txt [model]
# 
# 
###############################################################################
# CS B551 Fall 2018, Assignment #4 - Machine Learning
#
# Completed by:
# Derrick Eckardt
# derrick@iu.edu
# 
# Completed on December 9, 2018
#
# For the assignment details, please visit:
#
# https://github.iu.edu/cs-b551-fa2018/derrick-a4/blob/master/a4.pdf
#
################################################################################
################################################################################
# Overall
################################################################################
################################################################################

# Improve, find the optimal k number
# Remove, the iterative k finding for final copy

import sys
from operator import itemgetter
from collections import Counter, deque
import profile
import numpy as np
from scipy.spatial.distance import euclidean as euclid
from math import pow
from numpy import square

# Import command line inputs
traintest, input_file, model_file, model = sys.argv[1:]

def import_images(input_file):
    training_images =[]
    with open(input_file, 'r') as file:
        for line in file:
            splitline = line.split()
            # Had to convert these numbers into integers.
            # List of list, with format of [train_image, orientation, [rgb values]]
            training_images.extend([[splitline[0][6:], int(splitline[1]), [int(value) for value in splitline[2:]]]])
    return training_images

# For all take a list of list with inputs in form of [image, guess_orientation, actual_orientation]
def output(results):
    correct = sum([1 if guess == actual else 0 for image, guess, actual in results])
    total_images = len(results)
    # Print to screen
    print "Utilizing the "+model+" model, the results are as follows:"
    print 'Photos Correct:        '+str(correct)
    print 'Photos Incorrect:      '+str(total_images - correct)
    print 'Total Accuracy:        %.3f%%' % (correct/float(total_images)*100)

    # Print to file
    output_file = open("output.txt","w+")
    for image, guess, actual in results:
        output_file.write(traintest + "/" + str(image) +" "+ str(guess)+"\n")  #add input image number, # add guess
    output_file.close
    print "Individual test cases outputted to 'output.txt'."
        
# Use within nearest()
# Not currently used, as found it slightly faster to embed formula within code
def euclidean(train_features,test_features):
    feature_range = range(len(train_features))
    return sum([(train_features[i]-test_features[i])**2 for i in feature_range])
    # return sum([(train_features[i]-test_features[i])**2 for i in range(len(train_features)))])
    # return sum([(train-test)**2 for train, test in zip(train_features, test_features)])**(0.5)

def nearest_train():
    # Only copy the file over, since kNN is dependent on the test data
    train_output = open(model_file,"w+")
    with open(input_file, 'r') as file:
        for line in file:
            train_output.write(line)
    train_output.close
    print "k-Nearest Neighbors Model outputted to '"+model_file+"'."
    
def nearest_test(train_file, test_file, k):
    print "Loading training images from '"+train_file+"'."; train_images = import_images(train_file)
    print "Loading test images from '"+test_file+"'."; test_images = import_images(test_file)
    print "Classifying %d images. Estimated runtime is %d minutes and %.0f seconds." % (len(test_images) , (len(test_images) / 150), ((len(test_images) % 150) * (60.0/150.0)))
    feature_range, results = range(len(train_images[0][2])), []
    # create dictionary of squares, save computation time later.
    # takes a lot of time to keep finding euclidean distances.
    euclid_dict = {}
    euc_range = range(-256,256)
    # euc_range = range(0,256)
    for eu in euc_range:
        euclid_dict[eu] = eu**2
    # for eu in euc_range:
    #     euclid_dict[eu] = {}
    #     for ue in euc_range:
    #         euclid_dict[eu][ue] = (eu-ue)**2
    # for eu in euc_range:
    #     for ue in euc_range:
    #         euclid_dict[str(eu)+"-"+str(ue)] = (eu-ue)**2
    for test_image, actual_orientation, test in test_images:
        distances = [["",360,float('inf')]]*k
        max_k = distances[-1][2]
        # test_dict = {}
        # for m in range(len(test)):
        #     test_dict[m] = test[m]
        for image, orientation, train in train_images:
            # train_dict ={}
            # for m in range(len(train)):
            #     train_dict[m] = train[m]
            euclidean = 0 # This was actually the fastest way
            for j in feature_range:
                # euclidean += (train[j]-test[j])**2  # Don't need to find square root, since relative, save the operation
                euclidean += euclid_dict[train[j]-test[j]]
                # euclidean += euclid_dict[train_dict[j]-test_dict[j]]
                # euclidean += train[j]-test[j] if train[j] > test[j] else test[j] - train[j]
                # euclidean += abs(sum([train[j]-test[j]]))
                # euclidean += max([train[j]-test[j],test[j] - train[j]])
                # euclidean += sum((train[j]-test[j])
                # euclidean += abs(train[j]-test[j])
                # euclidean += interim** 2 #interim
                # euclidean += (train[j]-test[j]) **2   
                # euclidean += (train[j]-test[j]) *  (train[j]-test[j])
                # euclidean += pow(train[j]-test[j], 2)
                # euclidean += square(train[j]-test[j])
                # euclidean += euclid_dict[str(train[j])+"-"+str(test[j])]
                if euclidean > max_k:
                    break
            # euclidean = np.cumsum([(train[j]-test[j])**2 for j in feature_range])[-1]
            # print euclidean
            # euclidean = euclid(train,test)
            if euclidean < max_k:
                # distances[distances.index(max_k)] = [image, orientation, euclidean]
                # distances.pop()
                # distances.extend([[image, orientation, euclidean]])    
                distances[-1] = [image, orientation, euclidean]
                distances = sorted(distances, key=itemgetter(2))
                max_k = distances[-1][2]
        #         print "interim", distances
        # print "final  ",distances
        vote_guess = Counter([vote[1] for vote in distances]).most_common(1)[0][0]
        results.extend([[test_image, distances, actual_orientation]])  #vote_guess
    # back-up copy
    # for test_image, actual_orientation, test in test_images:
    #     distances = []
    #     for image, orientation, train in train_images:
    #         euclidean = 0 # This was actually the fastest way
    #         for j in feature_range:
    #             euclidean += (train[j]-test[j])**2  # Don't need to find square root, since relative, save the operation
    #         # euclidean = np.cumsum([(train[j]-test[j])**2 for j in feature_range])[-1]
    #         # print euclidean
    #         # euclidean = euclid(train,test)
    #         distances.extend([[image, orientation, euclidean]])
    #     vote_guess = Counter([vote[1] for vote in sorted(distances, key=itemgetter(2))[:k]]).most_common(1)[0][0]
    #     results.extend([[test_image, vote_guess, actual_orientation]])

    return results
    
if traintest == "train":
    # Import train file
    if model == "nearest":
        nearest_train()
    else:
        print "Unsupported Machine Learning Model."

elif traintest == "test":

    if model == "nearest":
        print "Classifying via k-Nearest Neighbors algorithm."
        # for k in [1,3,5,7,9,11,13,15,17,19,21,23,25,30,35,40,45,50,60,70,80,90,100]:
        # knn_file = open("knn-optimal.txt","w+")
        # for k in range(1,2):
            # print "k = ",k
        k=4000
        k_range = range(1,k+1)
        results = nearest_test(model_file, input_file, k)
            # correct = sum([1 if guess == actual else 0 for image, guess, actual in results])
        #     total_images = len(results)
        #     print "k =",k,"  ",correct/total_images
        #     knn_file.write(str(k)+" "+str(correct)+" "+str(total_images))
        # knn_file.close
            # output(results)
        # profile.run("nearest_test(model_file,input_file, 45)")
    else:
        print "Unsupported Machine Learning Model."

    total_images = len(results)
    output_file = open("knn-values.txt","w+")
    max_K_value = 0
    max_K_values = []
    for K in k_range:
        correct = 0
        for image, votes, actual in results:
            vote_guess = Counter([vote[1] for vote in votes[:K]]).most_common(1)[0][0]
            correct += 1 if vote_guess == actual else 0
        K_percent = correct/float(total_images)
        if K_percent >= max_K_value:
            max_K_values.append([K, K_percent])
        output_file.write(str(K) + " " + str(correct) +" "+ str(total_images)+" "+str(round(K_percent,5))+"\n")  #add input image number, # add guess
    output_file.close
    print "Various kNN cases Individual test cases outputted to 'knn-values.txt'."

    # output(results)
    
else:
    print "You entered an incorrect mode.  Only 'train' or 'test' are accepted."
    
#### To Do    
def adaboost():
    pass

def entropy():
    pass

def forest():
    pass

def best():
    pass

def other_ml_technique():
    pass

# For adaboost, forest
def output_model():
    # Train Mode Outputs for ada 
    # Output model_file
    training_file = open(model_file, "w+")
    training_file.write("Something") # Add model information
    training_file.close

