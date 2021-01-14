import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

class Evaluation:
    def __init__(self, true, predictions):
        self.true = true
        self.predictions = predictions

    def euclidean_distance(self):
        return 0

    def confusion_matrix(self):
        return 0

    def accuracy(self):
        # Handle cases where we have more or less predictions than labels
        # Handle how to decide which position to compare against for label

        # put predictions as misclassified if there are more than one box very close to the center points.
        max_distance = 100

        true_label = [i[0] for i in self.true if i] 
        true_point = [i[1] for i in self.true if i] 
        predictions_label = [i[0] for pred in self.predictions for i in pred]
        print("Non changed predictions_point:", self.predictions)
        predictions_point = [i[1] for pred in self.predictions for i in pred]
        print("Changed predictions_point:", predictions_point)
        
        acc = 0

        if predictions_point:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',  metric='euclidean').fit(predictions_point)
            distances, indices = nbrs.kneighbors(true_point)

            for i, indx in enumerate(indices):
                print("True: ", true_label[i])
                print("Pred: ", predictions_label[indx[0]])
                # print("Distance: ", distances[i])
                # If the label is correct and if the distance is smaller than x and if there
                # are not multiple predictions for the same instance
                if  true_label[i] == predictions_label[indx[0]] and distances[i] < max_distance:
                    acc += 1
                
                if predictions_label[indx[0]] == 6 and true_label[i] == 3 and distances[i] < max_distance:
                    acc += 1
            
            acc /= len(true_label)
            print("Accuracy: ", acc, "\n")
            # print("Indices: ", indices, ".\n Predictions point: ", predictions_point, ". \nDistances: ", distances)
            
            #print("True:", self.true, ". \n Pred:", self.predictions)

        return acc
        

    


'''
predictions = [[(0, (1637, 1414))], [(1, (1948, 931)), (1, (1487, 456))], [(2, (1962, 948))], [(3, (730, 1539))], [(4, (1262, 988))], [(5, (1968, 944)), (5, (710, 670)), (5, (1490, 459))]]
true = [(3, (680, 1508)), (5, (716, 656)), (1, (1268, 960)), (4, (1488, 448)), (2, (1960, 928)), (0, (1648, 1420))]

evaluater = Evaluation(true, predictions)
evaluater.accuracy()
'''