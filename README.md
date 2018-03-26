# Image_Classification

For the project,Fashion MNIST data set was chosen, released by Zolando, an e-commerce fashion retail company, which has its major operations in Europe

Many aspiring data scientists have already classified images of this data set, their go to algorithm was Convolution Neural Networks(CNN) algorithm using which around 90% accuracy was acheived in classification of images, but the algorithm is less intutive and complex to implement

Intent of the project was to understand image classification by applying intutive and easy to implement algorithms. Two such algorithms were K-Nearest Neighbors(KNN) and Random Forests

Though KNN was giving good accuracy (85%), the time complexity is high O(N*N)

Random forest model, a strong learner, intutive and simple algorithm to implement was choosen
Two step approach was followed to improve the accuracy of the Random forests model to 87%                                                 --> clustering technique to group data points with similar pixel intensities using K-Means clustering                                     --> Tuning the hyper parameters by varying number of estimators or trees and minimum samples leaf

These are just baby steps to learn image classification

Team mates https://github.com/PraveenDubba https://github.com/ramachandrakarthik
