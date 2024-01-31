import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns

class Model_RegLog:
    def __init__(self):
        """
        Initializes an instance of the Model_RegLog class.
        """
        self.model = None

    def train(self, X, y):
        """
        Trains the logistic regression model on the given input data.

        Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.

        Returns:
        None
        """
        self.model = LogisticRegression(
            multi_class='multinomial', solver='lbfgs', max_iter=10000
        )
        
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the target variable for the given input data.

        Parameters:
        X (array-like): Input data for prediction.

        Returns:
        array-like: Predicted target variable.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predicts the target variable for the given input data.

        Parameters:
        X (array-like): Input data for prediction.

        Returns:
        array-like: Predicted target variable.
        """
        return self.model.predict_proba(X)

    def plot_confusion_matrix(self, y_test, y_pred, name_target):
        """
        Plots the confusion matrix for the predicted labels.

        Parameters:
            y_test (array-like): True labels of the test data.
            y_pred (array-like): Predicted labels of the test data.
            name_target (array-like): Names of the target labels.

        Returns:
            None
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        # Plot non-normalized confusion matrix
        ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt=".0f")
        
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        ax.set_xticklabels(name_target, rotation=45)
        ax.set_yticklabels(name_target, rotation=45)

    def print_evaluation(self, X_test, y_test, y_pred, verbose):
        """
        Prints the evaluation metrics for the model's performance.

        Parameters:
            X_test (array-like): The input features for testing.
            y_test (array-like): The true labels for testing.
            y_pred (array-like): The predicted labels for testing.

        Returns:
            The average of the evaluation metrics.
        """
        score_details = {}
        # Calculer la précision
        accuracy = accuracy_score(y_test, y_pred)

        # Calculer la précision (precision)
        precision = precision_score(y_test, y_pred, average='micro')

        # Calculer le rappel (recall)
        recall = recall_score(y_test, y_pred, average='micro')

        # Prédire les probabilités
        y_pred_prob = self.model.predict_proba(X_test)

        # Calculer l'AUC
        ## Binariser les étiquettes dans un format one-vs-all pour l'utilisation avec roc_auc_score
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test, return_counts=False))
        y_pred_bin = self.model.predict_proba(X_test)

        ## Calculer l'AUC pour chaque classe
        auc = roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr')

        score_details['accuracy'] = accuracy
        score_details['precision'] = precision
        score_details['recall'] = recall
        score_details['AUC'] = auc

        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        if verbose:
            print('accuracy: ', accuracy)
            print('precision: ', precision)
            print('recall: ', recall)
            print('AUC: ', auc)
            print('kappa: ', kappa)
            print('F1_score: ', f1)
        score_details['kappa'] = kappa
        score_details['F1_score'] = f1
        return score_details