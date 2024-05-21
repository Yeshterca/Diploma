
def vw_load_prep_data(directory, no_subjects, no_hc, subs):

    betas_large, betas_small = load_subjects_3d(directory, 'beta_0001.nii', no_subjects, subs)
    betas_conc = mean_groups_for_superpixels(betas_large, no_hc)
    betas_conc_disp = mean_groups_for_superpixels(betas_small, no_hc)

    return betas_large, betas_conc, betas_conc_disp, betas_small


def vw_get_features(betas_clustered, betas_all, coords, no_subjects, no_features):

    diff = np.zeros((len(betas_clustered)))
    for i in range(len(betas_clustered)):
        if len(betas_clustered[i]) == 0:
            diff[i] = 0.
            continue
        #if np.any(betas_clustered[i] == 1.):
        #    diff[i] = 0.
        #    continue
        #if betas_clustered[i].size < 3000:  # TODO
        #    diff[i] = 0.
        #    continue
        mean = np.mean(betas_clustered[i], axis=1)
        diff[i] = np.abs(mean[0] - mean[1])

    sorted_idx = np.flip(np.argsort(diff))

    betas_all_norm = (betas_all - betas_all.min(axis=None, keepdims=True)) / \
                     (betas_all.max(axis=None, keepdims=True) -
                      betas_all.min(axis=None, keepdims=True))

    features_VW = np.zeros((no_subjects, no_features))
    for i in range(no_subjects):
        features_VW[i, :], best_sups = get_features(sorted_idx, betas_all_norm[i, :, :, :], coords, no_features)

    return features_VW, best_sups

def vw_classify(features_vw, labels, weights, kernel, no_features, pls, run):

    predictions, probs, pls = LOO_cross_validation_vw(features_vw, labels, weights, kernel, no_features, 'pls', pls, run)

    print('---------- VW-CLASSIFICATION ----------')
    accuracy, sensitivity, specificity, precision = get_measures(labels, predictions, weights)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('VW: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


    # accuracy, sensitivity, specificity, precision = cross_validation(labels, features_vw, weights, no_subjects, 'linear')
    #
    # print("--------------------")
    # print("VOXEL-WISE ANALYSIS")
    # print("Accuracy:", accuracy)
    # print("Sensitivity:", sensitivity)
    # print("Specificity:", specificity)
    # print("Precision:", precision)

    return accuracy, sensitivity, specificity, precision, pls