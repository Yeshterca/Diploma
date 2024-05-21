# IMPORTS
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score



def split_data(labels, betas, weights, no_subjects, splitrat=0.8):
    """
    Split data to training and validation sets.
    :param labels: array of correct labels
    :param betas: 2D array of beta values (selected components, all patients)
    :param no_subjects: number of all subjects
    :param splitrat: split ratio
    :return: betas and labels split randomly to training and validation sets
    """
    # Standardize
    betas_st = StandardScaler().fit_transform(betas)

    # Random shuffle
    r = np.array(range(labels.shape[0]))
    np.random.shuffle(r)
    labels_shuffled = labels[r]
    betas_shuffled = betas_st[r, :]

    # Split the data
    split_idx = int(no_subjects * splitrat)
    betas_train = betas_shuffled[:split_idx, :]
    labels_train = labels_shuffled[:split_idx]
    weights_train = weights[:split_idx]

    betas_val = betas_shuffled[split_idx:, :]
    labels_val = labels_shuffled[split_idx:]

    return betas_train, labels_train, betas_val, labels_val, weights_train


def svm_classifier(betas_train, labels_train, weights, betas_val, kernel):
    """
    Create and train simple SVM.
    :param betas_train: beta values from training set
    :param labels_train: correct labels from training set
    :param betas_val: beta values to predict
    :return: pred = vector, prediction 0=HC, 1=FES
             clf = trained model
    """
    # Create classifier
    if kernel == 'linear':
        clf = svm.SVC(kernel='linear', probability=True)
    else:
        clf = svm.SVC(kernel='rbf', gamma=0.01, probability=True)

    # Training
    clf.fit(betas_train, labels_train, weights)

    # Predict on val. data
    pred = clf.predict(betas_val)
    prob = clf.predict_proba(betas_val)
    coeffs = clf.coef_[0]
    intercept = clf.intercept_[0]

    #np.save('coeffs', coeffs)
    #np.save('intercept', intercept)

    return pred, prob[:, 1]


def measures_classifier(labels_val, pred):
    """
    Calculate several different measures of goodness of fit.
    :param labels_val: correct labels from validation set
    :param pred: predicted labels
    :return: prints
    """

    tn, fp, fn, tp = confusion_matrix(labels_val, pred).ravel()
    acc = metrics.accuracy_score(labels_val, pred, normalize=False)
    sen = metrics.recall_score(labels_val, pred)  # TP / (TP + FN)
    spec = (tn / (tn + fp))
    prec = metrics.precision_score(labels_val, pred)  # TP / (TP + FP)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Sensitivity:", sen)
    print("Specificity:", spec)


    return acc, sen, spec, prec


def cross_validation(labels, betas, weights, no_subjects, kernel, folds=10):
    """
    Perform X-fold cross-validation.
    :param labels: correct labels
    :param betas: 2D matrix of beta values of selected ICs
    :param no_subjects: number of all subjects
    :param folds: number of folds in cross-validation
    :return: accuracy, sensitivity, specificity, precision = average values of measures
    """
    accuracies = np.zeros((folds, 1))
    sensitivities = np.zeros((folds, 1))
    specificities = np.zeros((folds, 1))
    precisions = np.zeros((folds, 1))


    for fold in range(folds):
        betas_train, labels_train, betas_val, labels_val, weights_train = split_data(labels, betas, weights, no_subjects)
        prediction = svm_classifier(betas_train, labels_train, weights_train, betas_val, kernel)
        #prediction = log_reg_fit(betas_train, labels_train, betas_val)
        acc, sen, spec, prec = measures_classifier(labels_val, prediction)

        # Save measures
        accuracies[fold] = acc
        sensitivities[fold] = sen
        specificities[fold] = spec
        precisions[fold] = prec

    accuracy = np.sum(accuracies)/folds
    sensitivity = np.sum(sensitivities)/folds
    specificity = np.sum(specificities)/folds
    precision = np.sum(precisions)/folds
    """
    cnf_matrix = confusion_matrix(labels_val, prediction)
    class_names = ['HC', 'FES']
    fig1 = plt.figure(1)
    fig1, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('True label')
    plt.xlabel('Predictited label')
    """

    return accuracy, sensitivity, specificity, precision


def log_reg_fit(betas_train, labels_train, betas_val, weights):

    # Create classifier
    logreg = LogisticRegression(random_state=2)

    # Training
    logreg.fit(betas_train, labels_train, weights)

    # Predict validation data
    pred = logreg.predict(betas_val)

    prob = logreg.predict_proba(betas_val)

    return pred, prob[:, 1]

def pls(betas, labels):
    pls = PLSRegression(n_components=2)
    betas_new = pls.fit_transform(betas, labels)[0]

    return betas_new

def RBF(X, gamma):
    # Free parameter gamma
    if gamma == None:
        gamma = 1.0 / X.shape[1]

    # RBF kernel Equation
    K = np.exp(-gamma * np.sum((X - X[:, np.newaxis]) ** 2, axis=-1))

    return K

def RBF_gamma_search(betas, labels):

    param_grid = {'gamma': np.logspace(-2, 2, 1000)}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
    grid_search.fit(betas, labels)

    best_gamma = grid_search.best_params_['gamma']
    print("best gamma: ", best_gamma)

    return best_gamma


def get_weights(labels):

    size = len(labels)

    fes = np.where(labels == 1)[0]
    hc = np.where(labels == 0)[0]

    w_fes = (1. / len(fes)) * (size / 2)
    w_hc = (1. / len(hc)) * (size / 2)

    weights = np.empty(size)
    weights[fes] = w_fes
    weights[hc] = w_hc

    return weights


def ica_LOO_cross_validation(betas, labels, weights, kernel):
    """
    Perform X-fold cross-validation.
    :param labels: correct labels
    :param betas: 2D matrix of beta values of selected ICs
    :param no_subjects: number of all subjects
    :param folds: number of folds in cross-validation
    :return: accuracy, sensitivity, specificity, precision = average values of measures
    """
    folds = len(labels)
    predictions = np.zeros((folds,))
    probs = np.zeros((folds,))

    for fold in range(folds):
        betas_train = np.delete(betas, fold, 0)
        labels_train = np.delete(labels, fold)
        weights_train = np.delete(weights, fold)

        betas_val = betas[fold, :]
        betas_val = betas_val.reshape(1, -1)

        prediction, prob = svm_classifier(betas_train, labels_train, weights_train, betas_val, kernel)  #, prob
        #prediction, prob = log_reg_fit(betas_train, labels_train, betas_val, weights_train)
        predictions[fold] = prediction[0]
        probs[fold] = prob

    return predictions, probs


def LOO_cross(betas, labels, weights):

    clf = svm.SVC(kernel='rbf', gamma=0.01, class_weight=weights)
    loo = LeaveOneOut()
    scores = cross_val_score(clf, betas, labels, cv=loo)
    print(scores)

    acc = np.mean(scores)
    print("Mean accuracy:", acc)

    return acc


def get_measures(labels, pred, weights):


    tn, fp, fn, tp = confusion_matrix(labels, pred, sample_weight=weights).ravel()
    #x = (labels == pred) * weights
    #acc = (np.sum(x) / np.sum(weights))  # len(pred) #labels == pred TODO check weighting
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = (tp / (tp + fn))
    spec = (tn / (tn + fp))
    prec = (tp / (tp + fp))

    print("Accuracy:", acc)
    print("Sensitivity:", sen)
    print("Specificity:", spec)
    print("Precision:", prec)

    return acc, sen, spec, prec


def LOO_cross_validation_vw(features, labels, weights, kernel, no_features, transformation, pls, run):
    """
    Perform X-fold cross-validation.
    :param labels: correct labels
    :param betas: 2D matrix of beta values of selected ICs
    :param no_subjects: number of all subjects
    :param folds: number of folds in cross-validation
    :return: accuracy, sensitivity, specificity, precision = average values of measures
    """
    folds = len(labels)
    predictions = np.zeros((folds,))
    probs = np.zeros((folds,))

    for fold in range(folds):
        betas_train = np.delete(features, fold, 1)
        labels_train = np.delete(labels, fold)
        weights_train = np.delete(weights, fold)

        betas_val = features[:, fold]
        betas_val = betas_val.reshape(1, -1)

        if transformation == 'pca':
            pc_train, pca = get_features_pca(betas_train.T, no_features)
            pc_val = pca.transform(betas_val)
            predictions[fold], probs[fold] = svm_classifier(pc_train, labels_train, weights_train, pc_val, kernel)
        elif transformation == 'pls':
            if run == 0:
                pls_train, pls = get_features_pls(betas_train.T, labels_train, no_features)
            else:
                pls_train = pls.transform(betas_train.T)
            pls_val = pls.transform(betas_val)

            predictions[fold], probs[fold] = svm_classifier(pls_train, labels_train, weights_train, pls_val, kernel)

        #prediction = log_reg_fit(pc_train, labels_train, pc_val, weights_train)
        #predictions[fold] = prediction


    return predictions, probs, pls # TODO




