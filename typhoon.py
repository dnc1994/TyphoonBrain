# -*- encoding: utf-8 -*-
import math
import numpy as np
from pybrain.structure import LinearLayer, SigmoidLayer, LSTMLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

INPUT	=	8
HIDDEN	=	3
OUTPUT	=	1

data = np.loadtxt('data.txt')
data_x = data[:,:-1]
data_y = data[:,-1:]

net = buildNetwork(INPUT, HIDDEN, OUTPUT, hiddenclass=SigmoidLayer, bias=True, recurrent=True)
net.randomize()

ds_train = SupervisedDataSet(INPUT, OUTPUT)
ds_test = SupervisedDataSet(INPUT, OUTPUT)
for _ in range(5):
	for x, y in zip(data_x, data_y):
		ds_train.addSample(tuple(x), tuple(y))

for x, y in zip(data_x, data_y):
		ds_test.addSample(tuple(x), tuple(y))
		
trainer = BackpropTrainer(net, ds_train, learningrate=0.01, momentum=0.01, verbose=True, weightdecay=0.01)
trainer.trainUntilConvergence()
#trainer.trainEpochs(500)

print net.params

result = net.activateOnDataset(ds_test)

for p, y in zip(result, data_y):
	print p, '\t', y
	
output = ''
cnt = 0
err = 0
for x, y in zip(data, result):
	cnt += 1
	text = ' & '.join([str(cnt)] + [str(t) for t in x] + [str(y[0])])
	output = output + text + ' \\\\ \hline\n'
	err += (x[-1] - y[0]) ** 2 / 2	
	
print err
print math.sqrt(err * 2 / 19)
	
with open('tex.txt', 'w') as f:
	f.write(output)