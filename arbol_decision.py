#para este codigo utilizar el archivo data.csv

#importar librerias
import numpy as np
import pandas as pd

#leemos los datos ingresados por el usuario por csv

filename=input("Ingresa la ruta donde se encuentra el archivo csv: ")
data = pd.read_csv(filename, skiprows=1, header=None)
col_names=list(data.columns)


#inicializamos la clase nodo
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
       
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

#clase para el arbol dando el minimo de split y max depth

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        
        self.root = None

        #condiciones
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    #construir el arbol
    def build_tree(self, dataset, curr_depth=0):
        

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # dividir hasta cumplir las condiciones previamente establecidas 
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
           #revisar la gain
            if best_split["info_gain"]>0:
               
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                #nodo de decision
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value) #regresamos el nodo
    #funcion para obtener el mejor split
    def get_best_split(self, dataset, num_samples, num_features):

        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # regresar la mejor division
        return best_split
    #dividir dataset en derecha e izquierda
    def split(self, dataset, feature_index, threshold):

        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    #gain
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    #entropia
    def entropy(self, y):
        

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    #valor de gini
    def gini_index(self, y):

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def calculate_leaf_value(self, Y):

        Y = list(Y)
        return max(Y, key=Y.count)
    #imprimimos el arbol
    def print_tree(self, tree=None, indent=" "):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    #funcion para entrenar el arbol
    def fit(self, X, Y):

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    #funcion para predecir en el dataset
    def predict(self, X):

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    #hacemos las predicciones
    def make_prediction(self, x, tree):

        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

#Train-Test

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

#Fit al modelo

classifier = DecisionTreeClassifier(min_samples_split=9, max_depth=9)
classifier.fit(X_train,Y_train)
classifier.print_tree()

#test al modelo 

Y_pred = classifier.predict(X_test)

#librerias para las metricas 
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#matriz de confusion
conf_matrix=confusion_matrix(Y_test,Y_pred)
print("matriz de confusion")
print(conf_matrix)

#reporte de clasificacion
reporte=classification_report(Y_test,Y_pred)
print("Reporte de clasificacion")
print(reporte)



#importamos librerias para poder testear el accuracy de nuestro modelo y verlo graficamente
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Definir k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

depths = list(range(1, 10))
mean_accuracies = []

# Probar diferentes profundidades del árbol para ver su desempeño
for depth in depths:
    fold_accuracies = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        Y_train_fold, Y_val_fold = Y[train_index], Y[val_index]

        classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=depth)
        classifier.fit(X_train_fold, Y_train_fold)
        Y_pred_fold = classifier.predict(X_val_fold)

        fold_accuracies.append(accuracy_score(Y_val_fold, Y_pred_fold))

    mean_accuracies.append(np.mean(fold_accuracies))

# Gráfico de rendimiento 
plt.plot(depths, mean_accuracies,'go--')
plt.xlabel('Profundidad del arbol')
plt.ylabel('Accuracy Promedio')
plt.title('Desempeño del arbol respecto a su profundidad')
#plt.show()

import seaborn as sns
# Mostrar la matriz de confusión de manera gráfica
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Matriz de Confusión")
plt.show()