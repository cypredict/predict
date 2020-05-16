import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

def get_class_name():
	matfn = 'cars_meta.mat'
	data=sio.loadmat(matfn)
	print(data)
	class_list = []
	class_names = data['class_names'][0]
	for class1 in class_names:
		class_list.append(class1[0])
		#print(class1)
	label_list = range(1, 197)
	df = pd.DataFrame({'label': label_list, "name":class_list})
	df.to_csv('cars_class.csv', index=False) 

def analyze_train():
	matfn='cars_train_annos.mat'
	data=sio.loadmat(matfn)
	#print(data)
	cars = data['annotations'][0]
	label_list = []
	for car in cars:
		#print(car)
		x1, y1, x2, y2, label = car[0][0][0], car[1][0][0], car[2][0][0], car[3][0][0], car[4][0][0]
		print(label)
		#print(x1,y1,x2,y2,label)
		label_list.append(label)
	print(max(label_list)) #196

#matfn='cars_test_annos.mat'
get_class_name()
analyze_train()
