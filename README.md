# image_classification_task_nn
Image classification task using neural networks. Completed for a class project. Part 1 is a classical 80s style shallow network, and part 2 is a more modern network. For part 2, I added activation functions, implemented L2 Regularization, changed network depth and width, and used Convolutional Neural Nets to improve performance. 
The dataset consists of 10000 32x32 colored images total, split into 2500 development examples and 7500 training examples. There are 2999 negative examples and 4501 positive examples in the training set. Tried batch normalization but did not include it in my final implementation. 

This project was changed since I took this class but if you're here to copy code pls don't! Our university has great cheat detection, and you will likely get caught. Thanks to my fellow TAs/instructor for making this awesome assignment.

There are a few ways to run this code. Before you do, make sure you have Python3, Pytorch, Numpy, and any other required libraries installed. Thanks so much! To run, navigate to this repository in your terminal and then run the commands in the quotation marks below. Thanks so much! 

For help on running the Machine-Problem(MP), run 

"python main.py -h"

Run Part 1

"python main.py --part 1"

Run Part 2

"python main.py --part 2"

batches = max_iter, 500 batches (max_iter = 500)
