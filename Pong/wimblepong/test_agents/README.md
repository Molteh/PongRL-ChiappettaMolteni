# Solution
## A2C with Supervised Pre-Processing

In the following section we are going to describe the algorithms that we decided to submit since it gave us the most promising results in the testing phase. It can be described as an A2C model whose inputs are fed by the results extracted by a number of Neural Networks who themselves took the image frames as input. We decide to try this algorithm mainly for two reasons.

First, during the project, we were playing around a bit also with the non-visual version of the environment and noticed that we were able to reach interesting results with an A2C approach. Secondly, we tried to use NN on the image frames and noticed that we were able to extract with extreme precision some game-specifc information such as the position of the ball. The next subsections are going to be organized as follows: one section will be dedicated to describing how we structured and trained the NN used for information extraction, the next section will briefly describe the parameters on the training of the subsequent A2C algorithm. Finally one section will present the overall results of the model.
