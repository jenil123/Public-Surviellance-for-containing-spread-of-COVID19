# Public-Surviellance-for-containing-spread-of-COVID-19

This project is focused to contain spread of COVID-19.
The project was divided into 2 parts:
1. Social Distancing monitor:
In this module I made a model using YOLOV3 and OpenCV to monitor social distancing both in videos and real-time.
Here I have also shown Bird's eye view of given frame in order to get a better view of the distance between humans in video.

For realtime change videoCapture('filename) to videoCapture(0) 

2. Spit Detection:

a. Here I made a system  that  detect whether a person is spitting or not.

b. For this I am preparing the dataset and will update the work done by me.

c. I first got points for 2 reference videos using pose estimation model. 

d. For test videos I firstly generate points for test videos using pose estimation model. 

e. After that I compare 2 pose using Dynamic Time Warping technique.

The Dataset used here was custom and feel free to get access to the dataset.
