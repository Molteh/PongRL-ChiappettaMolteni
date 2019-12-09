# Solution
## A2C with Supervised Pre-Processing

In the following section we are going to describe the algorithms that we decided to submit since it gave us the most promising results in the testing phase. It can be described as an A2C model whose inputs are fed by the results extracted by a number of Neural Networks who themselves took the image frames as input. We decide to try this algorithm mainly for two reasons.

First, during the project, we were playing around a bit also with the non-visual version of the environment and noticed that we were able to reach interesting results with an A2C approach. Secondly, we tried to use NN on the image frames and noticed that we were able to extract with extreme precision some game-specifc information such as the position of the ball. The next subsections are going to be organized as follows: one section will be dedicated to describing how we structured and trained the NN used for information extraction, the next section will briefly describe the parameters on the training of the subsequent A2C algorithm. Finally one section will present the overall results of the model.

### Information extraction with NN

of the game. The data we were able to retrieve with a satisfactory degree of precision were the
following:

 - Y-coordinate of the ball
 - Y-coordinate of the player's paddle
 - Y-coordinate of the opponent's paddle

We were not able to extract with a sucient precision margin the X-position of the bull thus the
subsequent A2C model will not use this information.

#### Network structure

The first design decision the was taken regarding the network architecture was to train for each
feature a different network with just one output. The input of the model was a 100000 (100x100)
1D  float vector obtained from the 200*200*3 images returned by the environment. The G and
B layers were removed from the original images, it was then resized by a factor of 2 and finally, the background was set to 0 while the rest (ball, paddles) to 1. There was no need to add any convolutional layer because we do not want to enhance any characteristic of the images but rather preserving the exact position of a precisely shaped object.

We tried 2 different architectures to perform the task. The  first one, which proved to be the
most effective, was formed by two fully connected hidden layers. There were 10000 input features,
256 neurons in the first hidden layer, followed by a Rectifier Linear Unit, and 64 neurons in the
second fully connected hidden layer, again followed by ReLU. Finally, the output layer only had 1 output corresponding to the trained feature.

The alternative architecture that was tested included only one hidden layer with 100 neurons followed by ReLU.

#### Training, Validation and Testing

We applied a supervised learning approach to solve the task of extracting some game-specific information because we were able to generate a training data-set using the environment attributes and functions.


