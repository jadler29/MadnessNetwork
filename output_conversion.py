import numpy as np
import pandas as pd

w1 = np.array([1., 0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0., 1.])
w2 = np.array([0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0.])
w3 = np.array([1., 1., 0., 0., 0., 1., 0., 1.])
w4 = np.array([1., 0., 0., 1.])
w5 = np.array([0., 1.])
w6 = np.array([0.])

def output_conversion(w):
	w1, w2,w3,w4,w5,w6 = [np.round(w[i]) for i in range(1,7)]
	output = np.zeros([64,6])

	for i in range(w1.shape[0]):
		if w1[i] == 1.0:
			output[(2*i),0] = 1.0
			output[(2*i+1),0] = 0
		else:
			output[(2*i),0] = 0
			output[(2*i+1),0] = 1.0

	for i in range(w2.shape[0]):
		subset = output[(4*i):(4*i+4),0]
		contenders = []
		for x in range(len(subset)):
			if subset[x] == 1:
				contenders.append(4*i+x)
			else:
				output[(4*i+x),1] = 0 
		if w2[i] == 1.0:
			output[(contenders[0]), 1] = 1.0
			output[(contenders[1]), 1] = 0
		else:
			output[(contenders[0]), 1] = 0
			output[(contenders[1]), 1] = 1.0
	
	for i in range(w3.shape[0]):
		subset = output[(8*i):(8*i+8),1]
		contenders = []
		for x in range(len(subset)):
			if subset[x] == 1:
				contenders.append(8*i+x)
			else:
				output[(8*i+x),2] = 0 
		if w3[i] == 1.0:
			output[(contenders[0]), 2] = 1.0
			output[(contenders[1]), 2] = 0
		else:
			output[(contenders[0]), 2] = 0
			output[(contenders[1]), 2] = 1.0

	for i in range(w4.shape[0]):
		subset = output[(16*i):(16*i+16),2]
		contenders = []
		for x in range(len(subset)):
			if subset[x] == 1:
				contenders.append(16*i+x)
			else:
				output[(16*i+x),3] = 0 
		if w4[i] == 1.0:
			output[(contenders[0]), 3] = 1.0
			output[(contenders[1]), 3] = 0
		else:
			output[(contenders[0]), 3] = 0
			output[(contenders[1]), 3] = 1.0

	for i in range(w5.shape[0]):
		subset = output[(32*i):(32*i+32),3]
		contenders = []
		for x in range(len(subset)):
			if subset[x] == 1:
				contenders.append(32*i+x)
			else:
				output[(32*i+x),4] = 0 
		if w5[i] == 1.0:
			output[(contenders[0]), 4] = 1.0
			output[(contenders[1]), 4] = 0
		else:
			output[(contenders[0]), 4] = 0
			output[(contenders[1]), 4] = 1.0

	for i in range(w6.shape[0]):
		subset = output[(64*i):(64*i+64),4]
		contenders = []
		for x in range(len(subset)):
			if subset[x] == 1:
				contenders.append(64*i+x)
			else:
				output[(64*i+x),5] = 0 
		if w6[i] == 1.0:
			output[(contenders[0]), 5] = 1.0
			output[(contenders[1]), 5] = 0
		else:
			output[(contenders[0]), 5] = 0
			output[(contenders[1]), 5] = 1.0

	return output

if __name__ == "__main__":	
	print(output_conversion([0,w1, w2, w3, w4, w5, w6]))



