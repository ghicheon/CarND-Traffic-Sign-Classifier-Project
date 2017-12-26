# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image1.png "Traffic sign example"
[img1]: ./caution.jpg     "Traffic sign example"
[img2]: ./bumpy_road.jpg  "Traffic sign example"
[img3]: ./30.jpg          "Traffic sign example"
[img4]: ./100.jpg         "Traffic sign example"
[img5]: ./stop.jpg        "Traffic sign example"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This file is for that. Here is a link to my code.

[code](https://github.com/ghicheon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy library for it.

* The size of training set is ?   
There are 34799 training data.

* The size of the validation set is ?   
There are 4410  validation data.

* The size of test set is ?    
There are 12630 testing examples.

* The shape of a traffic sign image is ?   
All data is 32x32 images.All has 3 channels.

* The number of unique classes/labels in the data set is ?   
There are 43 Traffic sign in data set.

#### 2. Include an exploratory visualization of the dataset.

I'll show one example image in training data set.   

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After I loaded all data set, I divided all pixcel by 255 for normalization.  Therefore, all pixcel values will be between 0 and 1.  It will reduce training time and avoide local minimum. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My model has 7 layers below.

#Layer 1: Convolutional. Input: 32x32x3. Output: 28x28x32 =>14x14x32     
The output of convolution will be 28x28 because the kernel size is 5x5 and padding is VALID. 32 filters will be enough.   
After that, max polling makes the final output 14*14     
max    Pooling. Input = 28x28x32. Output = 14x14x32   

#Layer 2: Convolutional. Input: 14x14x32 Output = 10x10x64 =>5x5x64
This layer has 64 filters. The convolution shape will be having 10x10x64 because kernel size is 5x5 and valid padding is done.   
Final output will be 5x5x64  due to max pooling.   

Flattening is done for fully connected layers.   

#Layer 3: Fully Connected. Input = 5*5*64 Output = 256.   

#Layer 4: drop out
Drop out is added. It reduced overfitting for sure. The keeping probability is 50%.  

#Layer 5: Fully Connected. Input = 256. Output = 128.

#Layer 6: drop out
The probability is 50%.

#Layer 7: Fully Connected. Input = 128. Output = 43.
Final output is 43 probilities. Each of them means the possibilities for each traffic signs.

Yeah, the model is based on LeNet. Compared to LeNet, I used more neurons/filters and added drop out layer.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used Adam optimizer to train the model because it gave me good results(97% test accuracy).
The batch size is 128.
To select the training data,I got 128 random number among the number of traing set. Then trained the model with it. I repeated this pross 30000 times. It means around 234 epoches was executed.

((34799/128)*30000)/34799  = 234.375

Every 100 repeats, I checked test/validation accuracy. If this validation accuracy is better than previous best score, it will be saved as the file "./my_traffic_sign_classifier".





#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of ?   
 0.99985631771      

* validation set accuracy of ?     
 0.971428571429

* test set accuracy of ?    
0.970467141651    


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?    
LeNet Architecture.

* What were some problems with the initial architecture?    
It gave me not bad results. But it seems to suffer from under fitting  due to less neurons.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.    
I used more filters for convolution and more nodes for fully connected layer.

* Which parameters were tuned? How were they adjusted and why?    
After using more neurons, I got better results around 94% test accuracy. But it didn't improve  the performance any more. It seems to suffer from over-fitting!
Finally, I added drop out layers. It overcame the limitation. I got 97% test accuracy.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?    

Well.. I found out there is no optimal solution in all cases. When over fitting occurs, it will be better to add drop out or max pooling. if we encounter under fitting,it might help add more nodes or filters.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][img1] ![alt text][img2] ![alt text][img3] 
![alt text][img4] ![alt text][img5]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

---------------------------------------   
Image         /     Prediction    
---------------------------------------   
road work     / road work(0)     
bumpy road    / Traffic signal(X)     
30 km limit   / 30 km limit(O)     
100 km limit  / Yield (X)     
stop          / stop  (O)    
---------------------------------------     

The number of the correct answers  is 3 of 5. Its accuracy is 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

image no:  [5 softmax probabilities]       / traffic sign number for each probabilities]     
------------------------------------------------------------------------------------------------   
image 1 : [ 0.36683694  0.24223693  0.17578086  0.13434605  0.08079918]  /  [25 31 29 22 30]    
image 2 : [ 0.47163591  0.42746091  0.10126442  0.03440959 -0.03477075]  /  [26 18 20 17 11]    
image 3 : [ 0.66248077  0.15732659  0.14507991  0.03227759  0.00283519]  /  [ 1  5  2  3 15]    
image 4 : [ 0.30347174  0.28631833  0.26845863  0.12229205  0.0194592 ]  /  [13 15 12  1 26]    
image 5 : [ 0.75186783  0.11564886  0.10517246  0.08320452 -0.05589365]  /  [14 17  5  1  3]    

* image 1: the prediction is correct. It's kind of easy becuase the background is so simple and the traffic sign is so clear.    
* image 2: the answer is wrong. I think the reason is that background is so complicated and the sign is so small in the image.     
* image 3: correct. the sign is so clear and so big(compared to the image 2).    
* image 4: wrong. the size of the image is so small and  the background is not simple.    
           I think some preprocessing is needed to eliminate meaningless area.I mean the image needs to occupy lots of area in 32x32 image.    
* image 5: it's correct.so simple background & big sign compared to the size of the image.    



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


