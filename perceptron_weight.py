#Para este dataset utilizar el archivo data3.csv

# Perceptron para el dataset de obesidad 
from random import seed
from random import randrange
from csv import reader

# Cargar archivo CSV y leerlo fila por fila
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# convertir string a float las primeras 3 columnas 
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# convertir columna de string a integer la ultima columna
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Dividir dataset en k folds implementando cross-validation para evaluar
# y probar el rendimiento del modelo y encontrar el mejor modelo
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calcular la metrica de precision 
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluar el algoritmo con cross validation 
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Hacer las predicciones con los pesos 
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimar pesos de Perceptron usando descenso de gradiente estocástico
#funcion dd entrenamiento de pesos
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights #regresa los pesos

# Algoritmo de perceptrón con descenso de gradiente estocástico
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions) #regresa las predicciones

# Probar con el dataset de pesos de diabetes
seed(1)
# Solicitar al usuario indicar la ruta donde se encuentra el archivo csv 
filename = input("Introduce la ruta del archivo que deseas cargar: ")
try:
    #filename = 'C:/Users/rocky/OneDrive/Documentos/Machine learning/data3.csv'
    dataset = load_csv(filename)
	#cambiar de string a float
    for i in range(len(dataset[0])-1):
	    str_column_to_float(dataset, i)
    # cambiar de string a integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # evaluar el algoritmo
    n_folds = 3
	#learning rate
    l_rate = 0.1
	#epocas
    n_epoch = 50
	#evaluar el algoritmo y obtener las metricas del modelo
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
	#imprimir los scores y las metricas
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

#si no se encunetra el archivo se regresa este mensaje
except FileNotFoundError:
    print("Archivo no encontrado.")