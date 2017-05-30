import numpy as np
import pickle
import time
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers
from chainer.datasets import get_cifar10
from chainer.optimizer import WeightDecay

import network

def main():
	parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
	args = parser.parse_args()

	train, test = get_cifar10()
	train_x = [x[0][None, :, :, :] for x in train]
	train_t = [x[1] for x in train]
	test_x = [x[0][None, :, :, :] for x in test]
	test_t = [x[1] for x in test]

	train_mirror = [x[0][None, :, :, ::-1] for x in train]
	train_x.extend(train_mirror)
	train_t = train_t * 2

	train_mean = np.vstack(train_x).mean(axis=0)
	train_x -= train_mean[None, :, :, :]
	test_x -= train_mean[None, :, :, :]

	train_count = len(train_x)
	test_count = len(test_x)

	model = network.CNN()

	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(WeightDecay(5e-4))

	maxepoch = 100
	batchsize = 50
	train_loss = []
	train_acc = []
	test_loss = []
	test_acc = []

	start_time = time.clock()

	for epoch in range(maxepoch):
		order = np.random.permutation(train_count)
		sum_train_loss = 0
		sum_train_acc = 0
		for i in range(0, train_count, batchsize):
			mbx = model.xp.vstack([model.xp.asarray(train_x[j]) for j in order[i: i+batchsize]])
			mbt = model.xp.asarray([train_t[j] for j in order[i: i+batchsize]])
			y = model(mbx, train=True)
			loss = F.softmax_cross_entropy(y, mbt)
			acc = F.accuracy(y, mbt)
			model.zerograds()
			loss.backward()
			optimizer.update()
			sum_train_loss += float(loss.data) * len(mbx)
			sum_train_acc += float(acc.data) * len(mbx)
			print('epoch:{0} i_train:{1} loss:{2} accuracy:{3}'.format(epoch, i, float(loss.data), float(acc.data)))
		train_loss.append(sum_train_loss / train_count)
		train_acc.append(sum_train_acc / train_count)

		sum_test_loss = 0
		sum_test_acc = 0
		for i in range(0, test_count, batchsize):
			mbx = model.xp.vstack([model.xp.asarray(x) for j in test_x[i: i+batchsize]])
			mbt = model.xp.asarray(test_t[i: i+batchsize], dtype='int32')
			y = model(mbx, train=False)
			loss = F.softmax_cross_entropy(y, mbt)
			acc = F.accuracy(y, mbt)
			sum_test_loss += float(loss.data) * len(mbx)
			sum__test_acc += float(acc.data) * len(mbx)
			print('epoch:{0} i_test:{1} loss:{2} accuracy:{3}'.format(epoch, i, float(loss.data), float(acc.data)))
		test_loss.append(sum_test_loss / test_count)
		test_acc.append(sum_test_acc / test_count)

	print('save the model')
	serializers.save_npz('model', model)
	print('save the optimizer')
	serializers.save_npz('state', optimizer)

	with open ('log.pkl', 'wb') as f:
		dict = {'train_loss' : train_loss, 'train_acc' : train_acc,
				'test_loss' : test_loss, 'test_acc' : test_acc}
		pickle.Pickler(f).dump(dict)

	end_time = time.clock()
	print('total time:', end_time - start_time)

if __name__ == '__main__':
	main()
