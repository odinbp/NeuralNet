import csv
import tflowtools as TFT
import math

results = []
feature_data = []
feature_class = []
dataStructured = []

#Remember wine ;

def getData(ds, caseFraction):
	if len(ds) > 4 and ds[-4] == '.':
		return getTextFileData(ds, caseFraction)
	else:
		t = ds.split(';')
		temp = t[1].split(',')
		par = list(map(int, temp))
		return getattr(TFT, t[0])(*par)



def getTextFileData(x, caseFraction):
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
				feature_class = TFT.int_to_one_hot(feature_class_value, len(feature_data))
				dataStructured.append([feature_data, feature_class])
		else:
			for row in csv.reader(inputfile):
				if iterations == 0:
					break
				iterations -= 1
				feature_data = row[:-1]
				feature_data = list(map(float, feature_data))
				feature_class_value = int(row[-1])
				feature_class = TFT.int_to_one_hot(feature_class_value, len(feature_data))
				dataStructured.append([feature_data, feature_class])
	return dataStructured
