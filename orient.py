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
from pprint import pprint
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean as euclid
from math import pow
from numpy import square
import random
import pickle

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
    
def import_for_trees(input_file):
    # create headings
    pixels = [str(i) for i in range(1,9)]
    features =['filename','orientation'] + [color+row+col for row in pixels for col in pixels for color in 'rgb' ]

    # Load into a pandas dataframe, since that will make sorting by features easiest
    training_images = pd.read_csv(input_file, names=features, sep=" ")
    
    # drop filename column
    # training_images.drop(['filename'], axis = 1)

    # features doesn't need to track filename or orientation
    features = features[2:]

    return training_images, features

def dt_filters(training_images):
    
    # Make a copy
    filtered_images = training_images * 1


    # Various filters I have used
    # Convert them to dark or light (0 or 1)
    def halves(x):
        return 0 if x <128 else 1

    # Converts them to quarters, from 0, 1, 2, or 3
    def quarters(x):
        return 0 if x <64 else 1 if x < 128 else 2 if x < 192 else 3

    # apply filters
    for column in features:
        filtered_images[column] = filtered_images[column].apply(quarters)

    return filtered_images

def entropy(images,eval_column):
    items = np.unique(images[eval_column])
    counts = Counter(images[eval_column])
    total_items = float(sum(counts.values()))
    entropy = sum([(-counts[each]/total_items)*np.log2(counts[each]/total_items) for each in items])
    return entropy, list(items)

def plurality_value(images,class_col):
    return Counter(images[class_col]).most_common(1)[0][0]

def create_decision_tree(images, parent_images, features, class_col, current_depth, depth):
    # Based on Figure 18.5 in Course Text
    # If no images, then use the most common value of the parent node
    # Or, if at the deepest level
    if len(images) == 0:
        return plurality_value(parent_images, class_col)
    # If they are all the same classification, return the classification
    elif len(np.unique(images[class_col])) <= 1:
        return np.unique(images[class_col])[0]
    # If no more features left, then return most common value of the examples
    elif len(features) == 0 or current_depth == depth:
        return plurality_value(images, class_col)
    else:
        # O, Decision Tree! O, Decision Tree!    
        entropies = [[feature] + list(entropy(images,feature)) for feature in features]
        # print entropies
        max_feature, max_entropy_value, max_entropy_items = sorted(entropies, key=itemgetter(1), reverse=True)[0]
        decision_tree = {max_feature: {}}

        # take max_feature out of my feature set
        features = [feature for feature in features if feature != max_feature]

        # Iterate over each unique value 
        for item in max_entropy_items:
            
            # Get the data-subset
            item_images = images.where(images[max_feature] == item).dropna()

            # Recursively call this algorithm
            sub_decision_tree = create_decision_tree(item_images, images, features, 'orientation', current_depth+1,depth)

            # Add branch to tree:
            decision_tree[max_feature][item] = sub_decision_tree
            
        return decision_tree
        
def predict_decision_tree(learned_decision_tree, image):
    # print "image", image
    for key in learned_decision_tree.keys():
        
        # if [image.iloc[0][key]] in learned_decision_tree[key][image.iloc[0][key]].values():
        value = learned_decision_tree[key].get(image.iloc[0][key],0.0)
        if isinstance(value, dict):
            return predict_decision_tree(value,image)
        else:
            return value
        # else:
        #     # in the event a test_image has value not in the model
        #     return 0.0
     
def create_forest():
    pass

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
        
# Use within nearest() - Not currently used, as found it slightly faster to use dictionaries
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
    # takes a lot of time to keep finding the squares for the euclidean distances.
    euclid_dict, euc_range = {}, range(-256,256)
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
        results.extend([[test_image, vote_guess, actual_orientation]])  #vote_guess

    return results
    
if traintest == "train":
    # Import train file
    if model == "nearest":
        print "Training via k-Nearest Neighbors algorithm."
        nearest_train()
    elif model =="forest":
        print "Training via Forest algorithm."
        print "Loading images from '"+input_file+"'."
        training_images, features = import_for_trees(input_file)
        print "Running filters to make features friendlier."
        filtered_images = dt_filters(training_images)
        # randomly selecting five features to be used, since there are 192
        # did not want too deep a tree.  Since making a forest, this will fix itself
        # also makes it easier to use for decision stumps
        # sub_features = features*1  # Testing only
        
        # Select subset of features
        sub_features = random.sample(features,20)
        # Make a Decision Tree
        print "Learning Decision Tree"
        decision_tree_old = create_decision_tree(filtered_images, filtered_images, sub_features,'orientation',0,10)
        # pprint(decision_tree)

        # Outputting File
        print "Outputting Model via Pickle"
        pickle.dump(decision_tree_old, open(model_file, "wb" ))
        print "Loading Model via Pickle"
        decision_tree = pickle.load(open(model_file, "rb" ))

        # Testing Decision Tree
        print "Testing Decision Tree"
        testing_images, test_features = import_for_trees("test-data.txt")
        testing_images = dt_filters(testing_images)
        image_range = range(len(testing_images))
        correct = 0
        for i in image_range:
            image = testing_images[i:i+1]
            # print predict_decision_tree(decision_tree, image) 
            correct += 1 if predict_decision_tree(decision_tree, image) == image.iloc[0]['orientation'] else 0
        print correct," ", correct/float(len(testing_images))
        
    else:
        print "Unsupported Machine Learning Model."

elif traintest == "test":

    if model == "nearest":
        print "Classifying via k-Nearest Neighbors algorithm."
        k = 11 # based on results of code to find optimal range for k
        results = nearest_test(model_file, input_file, k)

        # Testing code to find slowest part of code
        # profile.run("nearest_test(model_file,input_file, 11)")
    else:
        print "Unsupported Machine Learning Model."

    # Testing code to find optimal k value
    multiple_k(results)

    # Output results
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


def multiple_k(results):
    total_images = len(results)
    output_file = open("knn-values.txt","w+")
    max_K_value = 0
    max_K_values = []
    k_range = range(1,k+1)
    for K in k_range:
        correct = 0
        for image, votes, actual in results:
            vote_guess = Counter([vote[1] for vote in votes[:K]]).most_common(1)[0][0]
            correct += 1 if vote_guess == actual else 0
        K_percent = correct/float(total_images)
        if K_percent >= max_K_value:
            max_K_values.append([K, K_percent])
            max_K_value = K_percent
        output_file.write(str(K) + " " + str(correct) +" "+ str(total_images)+" "+str(round(K_percent,5))+"\n")  #add input image number, # add guess
    output_file.close
    print "Various kNN cases Individual test cases outputted to 'knn-values.txt'."