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

To calculate this, for the 192 features, I calculated the Euclidean distance of each feature in the training set to the test image.

I spent some effort to finding the optimal k.  I was able to modify my code such that I could find out my accuracies for k from 1 to 4000.  Surprisingly, the value of k didn't matter all that much.  It varied from 70 to 72% from k=9 all the way up to 4000.  In effect, there was not that big a deal.  For this test set, I found that the maximum was technically at k=48. I found that any k below 400 got over 71%.  Below is a chart showing the different k values.  it's a rolling average of the next k values, so that the chart would be a little smoother:

![Rolling K Values -- How k varies](https://github.iu.edu/cs-b551-fa2018/derrick-a4/blob/master/part1/Rolling_Percent_vs_k.png "How k Varies")

As you cann see, there are a couple of regions where it seems to go up.  That's around k = 50, and k = 325.  I choose to set my k to 50 for two reasons.  One, on a different traininset, 300-350 might not be as accurate, and it could be a bit of overfitting. It doesn't seem to follow the rest of the curve.  There is an increase, and then a gradual decline.  The peak around 325 seems more like an anomaly.  Secondly, for about the result, k=50 compared to k=325 is faster to compute.

In order to cut my run time for the testing, I had to use the python "profile" library to do so to find out what was using valuable resources.  This helped me see where I could easily save time.  It also helped me figure out what the most expensive operations where.  Surprisingly, it was the calculation of the square for the euclidean distance.  If you look at my code, you will see dozed of different ways that I attempted to calculate it.  The fastest way is what I implemented, where I actually created a dictionary of the squares of -256 through 256.

The main thing that cut my run time down was that I kept a running list of the closest images, and then as I added together the euclidean distances, once it exceeded the maximum value of my closest images, I stopped considering that point.  This allowed me to cut my run time to around 6 minutes.  Hooray!

## Decision Forest

### Training

Compared to kNN, the decision forest was the complete opposite, because it does the hard work on the training side, where it has to grow each of the trees.

To grow the trees, I followed very closely the algorithm presented in Figure 18.5 of the course book on page 715.  The code is very concise.  The biggest challenge is to understand how the recursive nature of the tree works.

The first thing I did was filter my trees so that light for the trees was broken down into quarters, so that instead of dealing with 256 potential class values, I only had four.

One of the big design decision on had on my tree, was how deep would I let my trees grow.  Related, how many features would I make available to my tree to select from. Ultimately, I decided on more shallow trees, as opposed to bigger deeper trees.  The main reason was that each tree increased computing time not just to build, but also later on to find the predictive values.

Since I went with short trees with a depth of only 3, I went with 50 trees.  This yielded pretty good results.

### Testing

Testing, is rather easy now that we did the hardwork of growing an entire forest.  I just simply had to take a vote of each tree for each test image, and then take the most popular vote. This resulted in accuracy of around 67%

## Adaboost

### Training

This was definitely most difficult for me to wrap my head around, as there were many different ways and different weak classifiers to use.  For this, I used a decision stump, utilizing a lot of the code that I had written for the decision forest.  I built my algorithm following the steps that Reza lines out in the video on Adaboost in Module 17.1.

In terms of features, I created some features based on brightness of pixels, as suggested in the assignment prompt, office hours, and numerous piazza posts.  One piazza post, I got a suggestion on Piazza Post @742, that suggested only using the corners.  This allowed me to work with a much smaller feature set, that also potentially had much more value.  THe logic is that the corners are as far away from everyone else, so that they are the most likely to be different than each other, making it possible for there to be a great difference.

I ended up using 500 stumps, which was way more than I wanted to.  After that I still ended up with performance of 53.34%

There are likely many ways that I could improve it.  If I were to do it over, I would definitely find a better way to handle the multiple class nature of the code.

### Testing

TBD

## Best -- Ensemble Voting

### Training

One of the thing sthat interested me the most in this class that we didn't spend a lot of time on, was the thought to use some ensemble methods.  For my best method, I attempted to do a voting of the three methods.

So, since I had already computed it three other ways, I just loaded these results into training file.  **In order to use this model, you must run the other three algorithms first.  It will then use that ouput to compute the ensemble result from the three of them**

### Testing




## Performance Accuraces

Below are the accuracies of the various algorithms:

    Algorithm       Accuracy    Train Time  Test Time
    kNN	            71.26%      1s          6 minutes
    Decision Forest	67.13%      30 minutes  1 minutes
    AdaBoost	    53.34%      16 minutes  8 minutes
    Best - Voting	            10 seconds  10 seconds


## What do I recommend

If you were my consulting client, after recommending a larger budget, I would recommend that you go with the kNN algorithm.  It provided me the highest 


## Perforamnce Variations

3) Which classiers and which parameters would you recommend to a potential client? 

How does performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data? 

## Sample Image Results

4) Show a few sample images that were classied correctly and incorrectly. Do you see any patterns to the errors?