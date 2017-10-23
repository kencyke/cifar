# cifar
This is an python implementation of [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html).
## Validation
These codes are verified to run 3986.784109 seconds in Bitfusion Ubuntu 14 Chainer on June 4, 2017.   
If you are using a older version of Chainer , it might be necessary to change from
```python:train.py
chainer.cuda.get_device_from_id().use()
```
to
```python:train.py
chainer.cuda.get_device().use()
```
