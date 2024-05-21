from vw_run import *
from plots_figures import *


def optim_features(betas_small, labels, weights, segments, no_subjects):

    folds = len(labels)
    predictions = np.zeros((folds,))
    accs_vw = []
    features = []
    for i in range(1, 20):
        print('features:', i)
        no_features = i
        #pls_pixelmeans, pls, pixel_means_all
        pixel_means_norm, pixel_means_all = pixel_means(betas_small, segments, no_subjects)  #, no_features, no_subjects+1)    #vw_features_find# +1 compensates that it is not in LOO
        # pls_val = vw_features_transform(pls, betas_val, segments, pixel_means_all)

        for fold in range(folds):
            means, pixel_means_train, labels_train, weights_train, betas_val = data_split(pixel_means_all.T,
                                                                                                      pixel_means_norm.T,
                                                                                                      labels,
                                                                                                      weights, fold)
            betas_val = betas_val.reshape(1, -1)

            pls_train, pls = get_features_pls(pixel_means_train, labels_train, no_features) #.T
            pls_val = pls.transform(betas_val)

            prediction, probs = svm_classifier(pls_train, labels_train, weights_train, pls_val, kernel='linear')
            predictions[fold] = prediction

        accuracy, sen, spec, prec = get_measures(labels, predictions, weights)
        accs_vw.append(accuracy)
        features.append(i)

    np.save('accuracies_featureopt20.npy', accs_vw)
    np.save('no_features_featureopt20.npy', features)
    plot_features_accuracy(accs_vw, features)



