# plot_precision_recall_curve_function.py

# Funktion zum Visualisieren der Relevanz-Sensitivitäts-Kurve von Classifiern
#
# Eingabeparameter: 
#  - X_test: Datensatz
#  - y_test: Labels (Klassen des Datensatzes)
#  - classifier: Instanz des verwendeten Classifiers
#  - classifier_string: String der den verwendeten Classifier beschreibt (dient der Beschriftung der Grafik)
#  - prob_string: String der angibt, ob der verwendete Classifier die Methode "predict_proba" oder 
#       "decision_function" besitzt (-> führt zu unterschiedlicher Verwendung der precision_recall_curve-Funktion)


def plot_precision_recall_curve(X_test, y_test, classifier, classifier_string, prob_string):        
    
    from sklearn.metrics import precision_recall_curve
    if prob_string == "predict_proba":      # Handelt es sich um einen Classifier mit "predict_proba" Methode?
        precision, recall, thresholds = precision_recall_curve(y_test, classifier.predict_proba(X_test)[:,1])
    else:                                   # Handelt es sich um einen Classifier mit "decision_function" Methode?
        precision, recall, thresholds = precision_recall_curve(y_test, classifier.decision_function(X_test))
        
    # Plotten der Relevanz-Sensitivitäts-Kurve  
    import matplotlib.pyplot as plt
    plt.plot(precision, recall, label=classifier_string)
    plt.xlabel("Precision (Relevanz)")
    plt.ylabel("Recall (Sensitivitaet)")
    plt.legend()
    
    
    
    
    