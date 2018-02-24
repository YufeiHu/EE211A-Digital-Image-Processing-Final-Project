# EE211A-Digital-Image-Processing-Final-Project
Real-time Face Swap System based on Autoencoder

### 1/25: Test original source codes
The source codes for the original face-swapping project is found [here](https://github.com/joshua-wu/deepfakes_faceswap/).
The codes and data are downloaded and tested on the local machine. Everything is working properly.

### 1/29: Test training original dataset
![alt text](https://github.com/YufeiHu/EE211A-Digital-Image-Processing-Final-Project/blob/master/trump-cage.png)
The network is trained for limited epochs based on the original trump-cage dataset. The result is shown in the figure above.

### 2/3: Test OpenCV and webcam
OpenCV is successfully installed on the local machine.
The webcam is called, the captured image is then fed into our pre-trained model in real-time. Everything is working properly except the output image makes no sense. But it's OK since the network is not trained yet and everything for now is just for testing purposes.

### 2/23: Build our own yufei-qi dataset

