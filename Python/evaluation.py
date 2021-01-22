import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import sklearn.metrics
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
class Evaluation:
    def __init__(self):
        self.true = 0
        self.predictions = 0
        self.true_label = 0
        self.true_point = 0 
        self.predictions_label = 0
        self.predictions_point = 0
        self.cm_true = []
        self.cm_pred = []
        self.distances = 0
        self.indices = 0
        self.max_distance = 100

    def compute_confusion_matrix(self):
        true_fit = (len(self.predictions_point) < len(self.true_point))
        misclassified = self.predictions_label
        num_misclassified = 0
        num_missing = 0

        if (true_fit):
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',  metric='euclidean').fit(self.true_point)
            self.distances, self.indices = nbrs.kneighbors(self.predictions_point)
            misclassified = self.true_point

        indices_to_remove = []
        missing_preds = []

        for i, indx in enumerate(self.indices):
            if true_fit:     
                true = self.true_label[indx[0]]
                pred = self.predictions_label[i]
            else:
                true = self.true_label[i]
                pred = self.predictions_label[indx[0]]

            self.cm_true.append(true)
            self.cm_pred.append(pred)
            
            if true == pred and self.distances[i] < self.max_distance:
                indices_to_remove.append(indx[0])
            else:
                if self.distances[i] < self.max_distance: 
                    num_misclassified += 1
                else: 
                    num_missing += 1
        
        '''
        for index in sorted(indices_to_remove, reverse=True):
            misclassified.pop(index)
        '''

        # missing labels - every true that the closest label to it has a distance of > 100
        if true_fit:
            num_missing += len(self.true_label) -  len(self.indices)

        print("misclassified:", num_misclassified)
        print("missing:", num_missing, "percent:", (num_missing/len(self.true_label))*100, "length of true: ", len(self.true_label))
        
        return num_misclassified, num_missing, len(self.true_label)

    def get_confusion_matrix(self):
        cm = sklearn.metrics.confusion_matrix(self.cm_true, self.cm_pred, labels=np.arange(6)) #, normalize='all')
        return cm

    def plot_confusion_matrix(self, cm, title, path):
        df_cm = pd.DataFrame(cm, columns=np.arange(6), index=np.arange(6))

        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize = (10,7))
        plt.title(title, fontsize=20)
        sn.set(font_scale=1.7)#for label size
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        hm = sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})# font size
        plt.savefig(path)

    def accuracy(self, true, predictions, get_cm=True):

        self.true = true
        self.predictions = predictions
        self.true_label = [i[0] for i in self.true if i] 
        self.true_point = [i[1] for i in self.true if i] 
        self.predictions_label = [i[0] for pred in self.predictions for i in pred]
        self.predictions_point = [i[1] for pred in self.predictions for i in pred]

        # Handle cases where we have more or less predictions than labels
        # Handle how to decide which position to compare against for label
        acc = 0
        avg_distance = 0

        if self.predictions_point:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',  metric='euclidean').fit(self.predictions_point)
            self.distances, self.indices = nbrs.kneighbors(self.true_point)

            for i, indx in enumerate(self.indices):
            
                # If the label is correct and if the distance is smaller than x and if there are not multiple predictions for the same instance
                 if self.distances[i] < self.max_distance:
                    avg_distance += self.distances[i].squeeze()
                    if self.true_label[i] == self.predictions_label[indx[0]] or self.predictions_label[indx[0]] == 6 and self.true_label[i] == 3:
                        acc += 1

            acc /= len(self.true_label)
            avg_distance /= len(self.true_label)
            print("Accuracy: ", acc, "\n")
            print("Average distance: ", int(avg_distance), "\n")
            
        return acc, avg_distance


'''
predictions = [[(0, (1637, 1414))], [(1, (1948, 931)), (1, (1487, 456))]]#, [(2, (1962, 948))], [(3, (730, 1539))], [(4, (1262, 988))], [(5, (1968, 944)), (5, (710, 670)), (5, (1490, 459))]]
true = [(3, (680, 1508)), (5, (716, 656)), (1, (1268, 960)), (4, (1488, 448)), (2, (1960, 928)), (0, (1648, 1420))]

evaluater = Evaluation()
evaluater.accuracy(true, predictions)
'''