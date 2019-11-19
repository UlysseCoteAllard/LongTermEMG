import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet as rawConvNet
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    create_long_term_classification_graph, create_angles_classification_graph
from LongTermClassificationMain.TrainingsAndEvaluations.ForEvaluationSessions.network_standard_evaluation import\
    test_network_convNet


if __name__ == '__main__':
    path_weights = "../weights_full_training"
    print(os.listdir("../../"))
    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']
    angles_arm = dataset_evaluation['angles_and_timestamps']
    #print(angles_arm)
    '''
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         convNet=rawConvNet, use_only_first_training=True, path_weights=path_weights)
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         convNet=rawConvNet, use_only_first_training=False, path_weights=path_weights)
    '''

    # Graphs production
    ground_truths_no_retraining, predictions_no_retraining = np.load(
        "../../results/evaluation_sessions_no_retraining.npy")
    ground_truths_WITH_retraining, predictions_WITH_retraining = np.load(
        "../../results/evaluation_sessions_WITH_retraining.npy")

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')
    create_angles_classification_graph(predictions=predictions_no_retraining, ground_truth=ground_truths_no_retraining,
                                       angles=angles_arm)
    create_angles_classification_graph(predictions=predictions_WITH_retraining,
                                       ground_truth=ground_truths_WITH_retraining, angles=angles_arm)
    '''

    create_long_term_classification_graph(ground_truths_no_retraining=ground_truths_no_retraining,
                                          predictions_no_retraining=predictions_no_retraining,
                                          ground_truths_WITH_retraining=ground_truths_WITH_retraining,
                                          predictions_WITH_retraining=predictions_WITH_retraining,
                                          timestamps=evaluation_datetimes, number_of_seances_to_consider=6)

    fig, axs = create_confusion_matrix(ground_truth=ground_truths_no_retraining, predictions=predictions_no_retraining,
                                       class_names=classes, title="ConvNet standard training", fontsize=font_size)

    # fig.suptitle("ConvNet using AdaDANN training", fontsize=28)
    mng_no_retraining = plt.get_current_fig_manager()
    # mng.window.state('zoomed')  # works fine on Windows!
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.gcf().subplots_adjust(top=0.90)
    plt.show()

    _, _ = create_confusion_matrix(ground_truth=ground_truths_WITH_retraining,
                                   predictions=predictions_WITH_retraining, class_names=classes,
                                   title="ConvNet standard training", fontsize=font_size)

    # fig.suptitle("ConvNet using AdaDANN training", fontsize=28)
    mng_retraining = plt.get_current_fig_manager()
    # mng.window.state('zoomed')  # works fine on Windows!
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.gcf().subplots_adjust(top=0.90)
    plt.show()
    '''