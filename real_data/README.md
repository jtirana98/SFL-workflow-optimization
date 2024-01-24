The excel file have profiled data from the following ML tasks:


*datasets*: 
- MNIST[1], 
- CIFAR-10[2]

*models*: 
- Resene-101[3], 
- VGG-19[4]

Find the corresponding file using the rule: 'model_dataset.xlsx'

In each file there is a different sheet for each machine from our testbed (check Table 1). 
Each row has a computing cost (in ms) per-layer for one of the training operations. 
Specifically, the first column is the training time for the forward-propagation, the second for the back-propagation, and the last for the operation of updating the weights. 
The final sheet, named 'memory' has the memory load (KBytes) per-layer for the generated activations (first column) and the storing cost for the weights (second column)

##References:

[1]: Yann LeCun, L ́eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998

[2]: Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.


[3]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.


[4]: Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
