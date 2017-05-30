import chainer
import chainer.functions as F
import chainer.links as L

class CNN(chainer.Chain):
	def __init__(self):
		super(CNN, self).__init__(
			conv1 = F.Convolution2D(3, 64, 5, stride=1, pad=2),
			conv2 = F.Convolution2D(64, 64, 5, stride=1, pad=2),
			conv3 = F.Convolution2D(64, 128, 5, stride=1, pad=2),
			l1 = L.Linear(4 * 4 * 128, 1000),
			l2 = L.Linear(1000, 10)
		)
	def __call__(self, x, train):
		y = x
		y = F.max_pooling_2d(F.relu(self.conv1(y)), 3, 2)
		y = F.max_pooling_2d(F.relu(self.conv2(y)), 3, 2)
		y = F.max_pooling_2d(F.relu(self.conv3(y)), 3, 2)
		y = F.dropout(y, 0.5, train=train)
		y = F.relu(self.l1(y))
		y = F.relu(self.l2(y))
		return y
