#!/usr/bin/python
#
# ./orient.py : Perform Machine Learning Image Orientation, training usage:
#     ./orient.py train train_file.txt model_file.txt [model]

# ./orient.py : Perform Machine Learning Image Orientation, testing usage:
#     ./orient.py test test_file.txt model_file.txt [model]
# 
# 
###############################################################################
# CS B551 Fall 2018, Assignment #3 - Part 2: Optical Character Recognition (OCR)
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

import sys

# Import command line inputs
traintest, input_file, model_file, model = sys.argv[1:]

def nearest():
    pass

def adaboost():
    pass

def forest():
    pass

def best():
    pass

def other_ml_technique():
    pass

if traintest == "train":
    # Import train file
    training_images =[]
    with open(input_file, 'r') as file:
        for line in file:
            splitline = line.split()
            # Had to convert these numbers into integers.
            # List of list, with format of [train_image, orientation, [rgb values]]
            training_images.extend([[splitline[0][6:], int(splitline[1]), [int(value) for value in splitline[2:]]]])
        

    # Train!
    for image in training_images:
        pass

    # Train Mode Outputs
    # Output model_file
    training_file = open(model_file, "w+")
    training_file.write("Something") # Add model information
    training_file.close

elif traintest == "test":
    # Test Mode Outputs
    # Output information to screen
    print 'Photos incorrectly evaluated: '
    print 'Photos Correctly Evaluated: '
    print 'Total Accuracy: '
    
    # Output test results data to file
    output_filename = "output.txt"
    output_file = open(output_filename,"w+")
    output_file.write(traintest + "/" + "#####.jpg" + " " + "XXX")  #add input image number, # add guess
    output_file.close
    print "Individual test cases output to 'output.txt'"

else:
    print "You entered an incorrect mode.  Only 'train' or 'test' are accepted."