# EE211A-Digital-Image-Processing-Final-Project
Real-time Face Swap System based on Autoencoder
  
  
### 1/25: Test original source codes
The source codes for the original face-swapping project was found [here](https://github.com/joshua-wu/deepfakes_faceswap/).
The codes and data were downloaded and tested on the local machine. Everything is working properly.
  
  
### 1/29: Test training original dataset
![alt text](https://github.com/YufeiHu/EE211A-Digital-Image-Processing-Final-Project/blob/master/trump-cage.png)  
The network was trained for limited epochs based on the original trump-cage dataset. The result is shown in the figure above.
  
  
### 2/3: Test OpenCV and webcam
OpenCV was successfully installed on the local machine.
The webcam was called, the captured image was then fed into our pre-trained model in real-time. Everything is working properly except the output image makes no sense. But it's OK since the network is not trained yet and everything for now is just for testing purposes.
  
  
### 2/7: Test OpenCV and face-detection API
![alt text](https://github.com/YufeiHu/EE211A-Digital-Image-Processing-Final-Project/blob/master/face-shot.png)  
I tested the face-detection API provided in OpenCV on my local machine. The screenshot of running this API is shown in the figure above. It's functioning properly. Now we can start to prepare building our own dataset.
  
  
### 2/14: Set up an Amazon server for training our neural network models
I have successfully set up an Amazon server. This platform is pretty powerful as we can train our model using a NVIDIA V100 Tesla GPU which is THE MOST POWERFUL GPU on the planet earth! But of course, we have pay 3$/hour for the price.
  
  
### 2/23: Build our own yufei-qi dataset
![alt text](https://github.com/YufeiHu/EE211A-Digital-Image-Processing-Final-Project/blob/master/yufei-qi.png)  
Our dataset was successfully built! I (Yufei) have donated 515 faces and my teammate (Qi) has donated 500 faces. I tested training the models based on our own dataset for just a few epochs. The training result is shown in the figure above. Now the effect is not perfect, but it is expected to be much better after a long enough time of training.


### 2/26: Train model based on our own yufei-qi dataset
![alt text](https://github.com/YufeiHu/EE211A-Digital-Image-Processing-Final-Project/blob/master/yufei-qi2.png)  
After training for 8 hours, the result is making more sense now!
