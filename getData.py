import csv
import tflowtools as TFT
import math
import mnist_basics as mb
import random

results = []
feature_data = []
feature_class = []
dataStructured = []

def getData(ds, caseFraction, noClasses):
	dataStructured = []
	if ds == "mnist":
		features,labels = mb.load_all_flat_cases()
		scale_MNIST(features)
		ohl = make_one_hot(labels, noClasses)
		for i in range(len(features)):
			dataStructured.append([features[i], ohl[i]])
	elif len(ds) > 4 and ds[-4] == '.':
		dataStructured = getTextFileData(ds, caseFraction, noClasses)
	else:
		t = ds.split(';')
		temp = t[1].split(',')
		par = list(map(int, temp))
		dataStructured = getattr(TFT, t[0])(*par)
	if caseFraction != 1: 
			random.shuffle(dataStructured)
			dataStructured = dataStructured[:int(caseFraction*len(dataStructured))]
	return dataStructured

def getTextFileData(x, caseFraction, noClases):
	no_of_lines = 0
	with open('./Data sets/'+ x, newline='') as inputfile:
		no_of_lines = sum( 1 for _ in inputfile)
	iterations = int(no_of_lines*caseFraction)
	with open('./Data sets/'+ x, newline='') as inputfile:
		if x == 'winequality_red.txt':
			for row in csv.reader(inputfile, delimiter=';'):
				if iterations == 0:
					break
				iterations -= 1
				feature_data = row[:-1]
				feature_data = list(map(float, feature_data))
				feature_class_value = int(row[-1])
				feature_class = TFT.int_to_one_hot(feature_class_value, noClases)
				dataStructured.append([feature_data, feature_class])
		else:
			for row in csv.reader(inputfile):
				if iterations == 0:
					break
				iterations -= 1
				feature_data = row[:-1]
				feature_data = list(map(float, feature_data))
				feature_class_value = int(row[-1])
				feature_class = TFT.int_to_one_hot(feature_class_value, noClases)
				dataStructured.append([feature_data, feature_class])
	scale_others(dataStructured)
	return dataStructured


def scale_MNIST(unscaled):
	for i in unscaled:
		for k in range(len(i)):
			i[k]=i[k]/255

def scale_others(unscaled):
	smallest = [10000000 for i in range(len(unscaled[0][0]))]
	largest = [0 for i in range(len(unscaled[0][0]))]
	for i in unscaled:
		for k in range(len(i[0])):
			if i[0][k]>largest[k]: largest[k]=i[0][k]
			if i[0][k]<smallest[k]: smallest[k]=i[0][k]
	for i in unscaled:
		for k in range(len(i[0])):
			i[0][k]=(i[0][k]-smallest[k])/(largest[k]-smallest[k])



def make_one_hot(labels, noClases):
	ohl = []
	for label in labels:
		ohl.append(TFT.int_to_one_hot(label, noClases))
	return ohl



