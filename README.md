# CIFAR-10 Classification
This is an example of a convolutional neural network for image classification using CIFAR-10 dataset.  
If you do not know about CIFAR-10, please see https://www.cs.toronto.edu/~kriz/cifar.html .  
# Validation
I checked these codes running in Bitfusion Ubuntu 14 Chainer on June 4, 2017.  
To run them, you might need to change from
```python:train.py
chainer.cuda.get_device_from_id().use()
```
to
```python:train.py
chainer.cuda.get_device().use()
```
