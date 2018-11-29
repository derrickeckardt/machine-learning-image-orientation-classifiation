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

import sys
from operator import itemgetter
from collections import Counter
import profile

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
    print "Utilizing the "+traintest+" model, the results are as follows:"
    print 'Photos Correct:   '+str(correct)
    print 'Photos Incorrect: '+str(total_images - correct)
    print 'Total Accuracy:   %.3f%%' % (correct/float(total_images)*100)

    # Print to file
    output_filename = "output.txt"
    output_file = open(output_filename,"w+")
    for image, guess, actual in results:
        output_file.write(traintest + "/" + str(image) +" "+ str(guess)+"\n")  #add input image number, # add guess
    output_file.close
    print "Individual test cases output to: output.txt"
        
# Use within nearest()
def euclidean(train_features,test_features):
    total_features = len(train_features)
    feature_range = range(total_features)
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
    print "k-Nearest Neighbors Model outputted to: "+model_file
    
def nearest_test(train_file, test_file):
    train_images = import_images(train_file)
    test_images = import_images(test_file)
    k = 5
    results = []
    i = 1
    total_features = len(train_images[0][2])
    feature_range = range(total_features)
    print "Classifying "+str(len(test_images))+ " images."
    for test_image, actual_orientation, test in test_images:
        distances = []
        for image, orientation, train in train_images:
            distances.extend([[image, orientation, sum((train[j]-test[j])**2 for j in feature_range)]])
        vote_guess = Counter([vote[1] for vote in sorted(distances, key=itemgetter(2))[:k]]).most_common(1)[0][0]
        results.extend([[test_image, vote_guess, actual_orientation]])
        # print "Classifying image ", i
        i += 1
    return results
    

if traintest == "train":
    # Import train file
    if model == "nearest":
        nearest_train()
    else:
        print "Unsupported Machine Learning Model"

elif traintest == "test":

    if model == "nearest":
        results = nearest_test(model_file, input_file)
        # profile.run("nearest_test(model_file,input_file)")
    else:
        print "Unsupported Machine Learning Model"

    output(results)
    
else:
    print "You entered an incorrect mode.  Only 'train' or 'test' are accepted."
    
#### To Do    
def adaboost():
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

