Gesture Recognition using Hidden Markov Models
==============================================

y

This project presents an approach for recognizing and classifying different arm motion gestures by using IMU sensor readings from gyroscopes and accelerometers to train a set of Hidden Markov Models, which gives the result as log-likelihood for each class and the class with the highest likelihood is chosen as the result.

The data format of the IMU readings collected from consumer mobile device is:

| ts | Ax Ay Az	| Wx Wy Wz |
| ----------- | ----------- | ----------- |
| Time (ms) | 3x Accelerometer (m/s<sup>2</sup>) |	3x Gyroscope (rad/s) |


