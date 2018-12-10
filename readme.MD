# CS B551 - Assignment 4: Machine Learning

Completed by Derrick Eckardt on December 9, 2018.  Please direct any questions to [derrick@iu.edu](mailto:derrick@iu.edu)

The assignment prompt can be found at [Assignment 4 Prompt](https://github.iu.edu/cs-b551-fa2018/derrick-a4/blob/master/a4.pdf)

## Getting Started

As directed in the assignment, to run this program type the following at the command line:

./orient.py : Perform Machine Learning Image Orientation, training usage:

    ./orient.py train train_file.txt model_file.txt [model]

./orient.py : Perform Machine Learning Image Orientation, testing usage:

    ./orient.py test test_file.txt model_file.txt [model]

For example, to run all of the training scenarios:

    ./orient.py train train-data.txt nearest_model.txt nearest
    ./orient.py train train-data.txt adaboost_model.txt adaboost
    ./orient.py train train-data.txt forest_model.txt forest
    ./orient.py train train-data.txt best_model.txt best

And, for all of the testing scenarios:

    ./orient.py test test-data.txt nearest_model.txt nearest
    ./orient.py test test-data.txt adaboost_model.txt adaboost
    ./orient.py test test-data.txt forest_model.txt forest
    ./orient.py test test-data.txt best_model.txt best
    
Which results in the following scenarios:

    output_nearest.txt
    output_adaboost.txt
    output_forest.txt
    output_best.txt
    
All of which can be found in the directory.

## k-Nearest Neighbors (kNN)

### Training

The kNN requires almost no training, since the prediciton is based on the properties of the test image.  For this, I simply copied the information over.

### Testing

## Decision Forest

TBD

## Adaboost

TBD

## Best -- Ensemble Voting

TBD

## Performance Accuraces

2) present neatly-organized tables or graphs showing classication accuracies and running times as a function of the parameters you choose.

## What do I recommend


## Perforamnce Variations

3) Which classiers and which parameters would you recommend to a potential client? 

How does performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data? 

## Sample Image Results

4) Show a few sample images that were classied correctly and incorrectly. Do you see any patterns to the errors?