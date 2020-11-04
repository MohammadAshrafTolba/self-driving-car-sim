# self-driving-car-sim
This is a deep neural network to help the car self steer or as udacity calls it _car behavioral cloning_ ,
the network is fed three images from 3 different angles where each 3 images
are labeled with a steering angle

The model is built based on the [Nvidia model](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), more details of this model
can be found in their published paper.

I used the [udacity simulator](https://github.com/udacity/self-driving-car-sim), for generating data and testing in the autonomous
mode.

Note: This project was inspired by [naokishibuya's project](https://github.com/naokishibuya/car-behavioral-cloning) which is based on
Nvidia's paper too.

See the final results [See video](https://youtu.be/Dfoi3h3SETs)

### Notes
- Currently both training and validation loss are about 0.04-0.05, which results in a pretty good result

- Generating more data for the udacity simulator would definitely help

- Data used [here](https://github.com/rslim087a/track)

### How to run
Open the udacity simulator, choose autonomous mode and type the following command in your terminal:
``python drive.py``

### Framework used
Tensorflow 2 (with Keras interface).
