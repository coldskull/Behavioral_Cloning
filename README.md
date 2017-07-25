# **Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/left.jpg "Left Image"
[image3]: ./examples/right.jpg "Right Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with multiple convolutions interleaved with maxpooling layers. The inspiration is based on Nvidia network, but much simpler/less deeper :


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   							| 
| Keras Lambda Layer         		| 320x160x3 RGB Normalized image   							|
| Keras Cropping2D         		| Crop 70 pixels from top and 20 from bottom			| 
| Convolution 3x3    	| depth 6 	|
| RELU					|												|
| Max pooling	      	| 				|
| Convolution 3x3	    | depth 6      									|
| RELU          		|         									|
| Max Pooling			|      									|
| Convolution 5x5	    | depth 6      									|
| RELU          		|         									|
| Flatten		        | 										|
| Keras Dense (FC Layer)   			| output 160										|
| Keras Dense (FC Layer)			| output 60											|
| Keras Dense (FC Layer)   				|	output 1 (steering angle)											|


The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer 

#### 2. Attempts to reduce overfitting in the model

I did not use dropouts as the loss curves did not point to overfitting 

The model was trained and validated on different data sets to ensure that the model was not overfitting (20% of overall set was used for validation). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. Dense layers were handtuned to get lowest loss in minimal epochs

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:
* center lane driving
* recovering from the left and right sides of the road
* udacity base set
* augmented data generated on simulator for curves (especially for the sharp curves after the bridge)
* To increase the frequency distribution of curves vs straight segments, I copied and repeated entries in csv file with high steering angles




### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to come up with a model that could drive the vehicle at max speed (30mph) in most straight segments of the road.

My first step was to use a convolution neural network model similar to the Nvidia model, but smaller in size because of lower complexity of simulated environment vs real life. 

I did not run into issues of overfitting, but ran into issues like 'drifting to left' and going off lane on curves. 
I solved the 'drifting' issue by lowering the steering angle for left/right camera images. For the curves, I generated extra data and augmented base csv file with this data. I also increased data set size by adding grayscale images.

After this, I was able to complete the whole track at low speed -- 10mph

I made changes to drive.py to simulate more realistic drive experience. Basically, I varied the set point speed linearly with the current steering angle predicted by my model (within certain constrained range of steering angle).
This naive approach helped my car accelarate to max speed (30mph) in straight sections of the road and slow down on sharp curves.

A more sophisticated approach would have been to control setpoint based on some historic steering angles information included in calculations. This would allow car to go through a curve of constant curvature with a higher speed.



#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   							| 
| Keras Lambda Layer         		| 320x160x3 RGB Normalized image   							|
| Keras Cropping2D         		| Crop 70 pixels from top and 20 from bottom			| 
| Convolution 3x3    	| depth 6 	|
| RELU					|												|
| Max pooling	      	| 				|
| Convolution 3x3	    | depth 6      									|
| RELU          		|         									|
| Max Pooling			|      									|
| Convolution 5x5	    | depth 6      									|
| RELU          		|         									|
| Flatten		        | 										|
| Keras Dense (FC Layer)   			| output 160										|
| Keras Dense (FC Layer)			| output 60											|
| Keras Dense (FC Layer)   				|	output 1 (steering angle)											|




#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the base training data and added driving data collected on simulator (mainly curves)

![alt text][image1]
![alt text][image2]
![alt text][image3]

To augment the data set:
1) I flipped images with correponding flipping of steering angles
2) Added grayscale image for every original as well as flipped image

Original set had around 19777, and for each image the following images were added:
1) flipped
2) grayscale
3) flipped and grayscale

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by flatening of training loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
