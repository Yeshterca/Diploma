import matplotlib.pyplot as plt
import numpy as np

# ROIS HISTOGRAM

def plot_rois_distribution(dir):

    rois_out = np.load(dir)
    rois_out = np.squeeze(rois_out)
    #np.save('rois_out', rois_out)

    n, bins, patches = plt.hist(x=rois_out, bins=16, color='darkblue',
                                alpha=0.7, rwidth=0.85, label='subjects')

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    plt.vlines(14, 0, 21, colors='red', linestyles='dashed', label='threshold')  #26.1, 48
    plt.xlabel('Number of ROIs')
    plt.ylabel('Number of subjects')
    plt.title('Histogram of ROIs less than 50% covered in subjects, NUDZ')
    plt.legend()
    plt.show()


def plot_features_accuracy(accuracies, no_features):

    fig, ax = plt.subplots()
    ax.plot(no_features, accuracies)
    ax.set(xlabel='Number of features', ylabel='Weighted accuracy', title='Accuracy of the model in relation to the number of features')
    ax.grid()
    plt.show()


def display_clustering(overlay1, overlay2, overlay3):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(overlay1[:, :, 0])
    ax2.imshow(overlay2[:, :, 0])
    ax3.imshow(overlay3[:, :, 0])

    fig.suptitle('OVERLAYS')
    plt.show()


def display_mean_groups(betas_conc, slice):

    fig = plt.figure()
    axs = fig.subplot_mosaic([["HC 1", "FES 1"],
                              ["HC 2", "FES 2"],
                              ["HC 3", "FES 3"]])
    axs["HC 1"].set_title("HC 1")
    axs["HC 1"].imshow(betas_conc[slice, :, :, 0])
    axs["FES 1"].set_title("FES 1")
    axs["FES 1"].imshow(betas_conc[slice, :, :, 1])

    axs["HC 2"].set_title("HC 2")
    axs["HC 2"].imshow(betas_conc[:, slice, :, 0])
    axs["FES 2"].set_title("FES 2")
    axs["FES 2"].imshow(betas_conc[:, slice, :, 1])

    axs["HC 3"].set_title("HC 3")
    axs["HC 3"].imshow(betas_conc[:, :, slice, 0])
    axs["FES 3"].set_title("FES 3")
    axs["FES 3"].imshow(betas_conc[:, :, slice, 1])

    plt.show()


def display_subjects(betas, number, slice=55):
    plt.imshow(betas[:, slice, :])
    plt.savefig(number)









