import numpy as np
import pickle
import matplotlib.pyplot as plt

epc = []
maxepoch = 100
box_train_loss = []
box_test_loss = []
box_train_acc = []
box_test_acc = []

with open ('log.pkl', 'rb') as f:
	d = pickle.Unpickler(f).load()

for epoch in range(maxepoch):
	epc.append(epoch)
	box_train_loss.append(d['train_loss'][epoch])
	box_test_loss.append(d['test_loss'][epoch])
	box_train_acc.append(d['train_acc'][epoch])
	box_test_acc.append(d['test_acc'][epoch])

fig, (loss, acc) = plt.subplots(ncols=2, figsize=(10, 4))

loss.plot(epc, box_train_loss, label='train')
loss.plot(epc, box_test_loss, label='test')
loss.set_title('Model loss')
loss.set_xlabel('Epoch')
loss.set_ylabel('Loss')
loss.legend()

acc.plot(epc, box_train_acc, label='train')
acc.plot(epc, box_test_acc, label='test')
acc.set_title('Model accuracy')
acc.set_xlabel('Epoch')
acc.set_ylabel('Accuracy')
acc.legend(loc="lower right")

plt.show()
