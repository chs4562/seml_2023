# SEN P4: Klassifikation_3.py

import matplotlib.pyplot as plt
plt.close('all')    # Alle offenen Diagrammfenster schlieÃen


import pandas as pd
dataframe = pd.read_csv('breast-cancer-wisconsin.data')  # Einlesen der Daten im CSV-File in ein Pandas-DataFrame
# Für nähere Infos bzgl. des Datensatzes, siehe hier: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)

# Extrahieren der Merkmalsmatrix X und des Klassenvektors y: 
features = dataframe.loc[:,'Marginal_Adhesion':'Single_Epithelial_Cell_Size']         # Auswahl der Merkamle aus dem DataFrame (hier zu Demozwecken nur die Merkmale von 'Marginal_Adhesion' bis 'Single_Epithelial_Cell_Size').
X = features.values             # Merkmalsmatrix X
y = dataframe.label.values      # Im DataFrame sind die Labels wie folgt definiert: 2 = "gutartig", 4 = "bösartig"
# Umbenennen der Labels: 
y[y==2] = 0      # gutartig
y[y==4] = 1      # bösartig


# Splitten des Datensatzes in Trainings- und Testdaten:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)   # stratify=y hier zu Demozwecken absichtlich weggelassen, da dann der Einfluss der Schwellwertanpassung deutlicher sichtbar wird. 



# Tranieren unterschiedlicher Classifier und Darstellung der jeweiligen Ergebnisse
# in einer Relevanz-Sensitivitäts-Kurve (mittels der Function plot_precision_recall_curve_function):
# -----------------------------------------------------------------------------
import plot_precision_recall_curve_function as prc
plt.figure()    # Neue Figure öffnen (und alle Diagramme in dieser Figure darstellen)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
prc.plot_precision_recall_curve(X_test, y_test, DT, "Decision Tree", "predict_proba")

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)
prc.plot_precision_recall_curve(X_test, y_test, KNN, "k-Nearest-Neighbor", "predict_proba")

from sklearn.svm import SVC
SVM = SVC(gamma=.5, C=1).fit(X_train, y_train)
prc.plot_precision_recall_curve(X_test, y_test, SVM, "SVM", "decision_function")

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=20, max_depth=4).fit(X_train, y_train)
prc.plot_precision_recall_curve(X_test, y_test, RFC, "Random Forest", "predict_proba")
# -----------------------------------------------------------------------------



# Untersuchung des SVM-Classifiers vor der Schwellwert-Anpassung:
# -----------------------------------------------------------------------------
print("\nResults before threshold-adaption:")
print("-----------------------------------------------------")
print('Training-score: ', SVM.score(X_train, y_train))
print('Test-score: ', SVM.score(X_test, y_test))

# Confusion-Matrix:
from sklearn.metrics import confusion_matrix
y_pred = SVM.predict(X_test)
Conf_Matrix = confusion_matrix(y_test, y_pred)
print('Confusion-matrix:\n', Conf_Matrix)

# True und False Negatives und Positives (aus der Confusion-Matrix):
#TN = Conf_Matrix[0,0]       # True Negative
#FP = Conf_Matrix[0,1]       # False Positive
#FN = Conf_Matrix[1,0]       # False Negative
#TP = Conf_Matrix[1,1]       # True Negative

# Berechnung von Relevanz, Sensitivität und F1-Score (aus der Confusion-Matrix):
#Precision = TP/(TP+FP)      # Relevanz
#Recall = TP/(TP+FN)         # Sensitivität
#F1_Score = 2*Precision*Recall/(Precision+Recall) # F1-Score

# Berechnung von Relevanz, Sensitivität und F1-Score (mittels Python-Functions):
from sklearn.metrics import precision_score, recall_score, f1_score					
Precision = precision_score(y_test, y_pred)
Recall    = recall_score(y_test, y_pred)
F1_Score  = f1_score(y_test, y_pred)

# Ausgabe auf der Konsole:
print('Precision (Relevanz):', Precision)        # Für Krebsdiagnose weniger relevant.
print('Recall (Sensitivitaet):', Recall)         # Für Krebsdiagnose relevante Metrik.
print('F1-Score:', F1_Score)                     # Kann auch für Krebsdiagnose eine relevante Metrik darstellen.

#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))     # Klassifikationsreport auf der Konsole ausgeben

print("-----------------------------------------------------")
# -----------------------------------------------------------------------------



# Untersuchung des SVM-Classifiers mit Schwellwert-Anpassung:
# -----------------------------------------------------------------------------
# Zunächst plotten der Relevanz-Sensitivitäts-Kurve und finden eines geeigneten Schwellwerts
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, SVM.decision_function(X_test))   # Daten für die Relevanz-Sensitivitäts-Kurve berechnen lassen. 

# Darstellung von Precision und Recall als Funktion der Schwellwerte: 
plt.figure()
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Thresholds")
plt.ylabel("Precision and Recall")
plt.legend()
plt.grid()

# Algorithmische Ermittlung des passenden Schwellwerts:
import numpy as np 
idx_thresh_zero = np.argmin(np.abs(thresholds))      # Index bei dem der Schwellwert Null (oder sehr nahe an der Null) ist.
recall_goal = 0.95                                   # Angestrebte Sensitivität
idx_recall_goal = np.argmin(np.abs(recall-recall_goal))  # Index bei dem die SensitivitÃ¤t bzw. Reacll die gewünschte Prozentzahl erreicht. 

# Darstellen der Relevanz-Sensitivitäts-Kurve und Markieren der Punkte für "Schwellwert = 0" und "Recall (Sensitivität) = recall_goal":
plt.figure()
plt.plot(precision[idx_thresh_zero], recall[idx_thresh_zero], 'o', markersize=10, label="Threshold = 0", fillstyle="none", c='k')
plt.plot(precision[idx_recall_goal], recall[idx_recall_goal], 'o', markersize=10, label=("Recall = " + str(recall_goal)), fillstyle="none", c='r')
plt.plot(precision, recall, label="Precision-Recall-Curve")
plt.xlabel("Precision (Relevanz)")
plt.ylabel("Recall (Sensitivitaet)")
plt.legend()
plt.grid(color='k', linestyle='--', linewidth=1)

print("\nResults after threshold-adaption:")
print("-----------------------------------------------------")
chosen_threshold = thresholds[idx_recall_goal]            # Wahl des Schwellwerts für die gewünschte Sensitivität. 
print('Value of chosen threshold:', chosen_threshold)  

y_pred_train = SVM.decision_function(X_train) > chosen_threshold        # Klassifikation der Trainingsdaten (mit dem ermittelten Schwellwert)
y_pred_test = SVM.decision_function(X_test) > chosen_threshold          # Klassifikation der Testdaten (mit dem ermittelten Schwellwert)

# Scores (händisch berechnet):
print('Training-score: ', np.sum(y_pred_train==y_train)/np.size(y_pred_train))
print('Test-score: ', np.sum(y_pred_test==y_test)/np.size(y_pred_test))

Conf_Matrix = confusion_matrix(y_test, y_pred_test)
print('Confusion-matrix:\n', Conf_Matrix)

# Berechnung von Relevanz, Sensitivität und F1-Score (mittels Python-Functions):
Precision = precision_score(y_test, y_pred_test)
Recall    = recall_score(y_test, y_pred_test)
F1_Score  = f1_score(y_test, y_pred_test)

# Ausgabe auf der Konsole:
print('Precision (Relevanz):', Precision)        # Für Krebsdiagnose weniger relevant.
print('Recall (Sensitivitaet):', Recall)         # Für Krebsdiagnose relevante Metrik.
print('F1-Score:', F1_Score)                     # Kann auch für Krebsdiagnose eine relevante Metrik darstellen.

#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred_test))      # Klassifikationsreport auf der Konsole ausgeben

print("-----------------------------------------------------")
# -----------------------------------------------------------------------------


# Erkenntnisse: 
# Die Relevanz-Sensitivitäts-Kurven der untersuchten Classifier sind relativ ähnlich. Hier fiel die Wahl
# für die weitere Untersuchung auf die Kernel-SVM.
# Der Classifier mit den Default-Einstellungen funktioniert relativ gut, mit Scores um die 90 %. Die Sensitivität
# ist mit unter 80 % jedoch relativ gering, was im vorliegenden Anwendungsfall bedeutet, dass viele Patienten mit 
# bösartigem Krebs als "gutartig" diagnostiziert werden. 
# Durch Anpassung des Entscheidungsschwellwerts (aka. "Arbeitspunkteinstellung") wurde die Sensitivität auf ca. 95 % 
# erhöht. Somit kommt es zu deutlich weniger False Negatives. Dies lässt sich auch in der Confusion Matrix erkennen.
# Hierdurch sinkt zwar der Score, für den vorliegenden Anwendungsfall erscheint das Ergebnis nach Anpassung des 
# Schwellwerts jedoch deutlich besser. 

