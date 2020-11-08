Concrete Surface Crack Segmentation
===================================
Data
----
Data has been downloaded from [Mendeley](https://data.mendeley.com/datasets/jwsn7tfbrp/1). It includes 458 images of concrete cracks with their corresponding ground truth masks.
Data augmentation has also been performed by applying horizontal, vertical flips, rotating, scaling, shifting and adjusting the hue and saturation.

Model
-----
The CNN in this study is an encoder-decoder based network. The network architecture is depicted below. The [EfficientNetB7](https://arxiv.org/abs/1905.11946) without its 7th block and the fully connected layer at the top is utilized as an encoder. In the network [dense atrous convolution](https://arxiv.org/abs/1903.02740) module is used in the middle. As an attention mechanism, the [scSE](https://arxiv.org/abs/1803.02579) module is also inserted in some places which are shown in the following figure. 

![model](https://github.com/ae5n/Concrete-Crack-Segmentation/blob/master/imgs/architecture.png)

Training
--------
The model has been trained with initial settings such as 250 number of epochs (with an early stop criterion of 26 epochs) and a learning rate of 1e-4 (reducing by a factor of 0.1 after every 12 epochs). The Adam with Nesterov momentum (NAdam) is used for optimization, and the loss function for training is the combination of dice loss and cross-entropy loss. Dice score, Precision and Recall metrics have been monitored to evaluate the performance of the network.

![loss](https://github.com/ae5n/Concrete-Crack-Segmentation/blob/master/imgs/loss.png)

Testing
--------------------------
Precision, recall, dice score, and AUC for the testing dataset are 0.91, 0.88, 0.89 and 0.998. The following figure shows some of the segmentation results in comparison with the U-Net Model.
![segmentation results](https://github.com/ae5n/Concrete-Crack-Segmentation/blob/master/imgs/results.png)
