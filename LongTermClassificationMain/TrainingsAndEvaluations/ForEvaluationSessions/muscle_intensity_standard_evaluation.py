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
    create_long_term_classification_graph, create_activation_classification_graph


def test_network_convNet_with_activation(examples_datasets, labels_datasets, convNet,
                                         highest_activation_participants_gestures,  num_kernels, filter_size=(4, 10),
                                         path_weights='../../weights', use_only_first_training=False):
    participants_evaluation_dataloader = load_dataloaders_test_sessions(
        examples_datasets_evaluation=examples_datasets, labels_datasets_evaluation=labels_datasets, batch_size=512)
    predictions = []
    ground_truths = []
    accuracies = []
    activations = []
    for participant_index, (dataset_participant, highest_activation_participant) in enumerate(
            zip(participants_evaluation_dataloader, highest_activation_participants_gestures)):
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        activations_participant = []
        model = convNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()
        for session_index, dataloader_session in enumerate(dataset_participant):
            if use_only_first_training:
                best_weights = torch.load(
                    path_weights + "/participant_%d/best_weights_participant_normal_training_%d.pt" %
                    (participant_index, 0))
            else:
                best_weights = torch.load(
                    path_weights + "/participant_%d/best_weights_participant_normal_training_%d.pt" %
                    (participant_index, int(session_index/2)))  # Because there is 2 evaluation sessions per training
            model.load_state_dict(best_weights)
            predictions_evaluation_session = []
            ground_truth_evaluation_session = []
            activation_evaluation_session = []
            highest_activation_participant_for_this_training_session = highest_activation_participant[
                int(session_index/2)]
            with torch.no_grad():
                model.eval()
                for inputs, labels in dataloader_session:
                    inputs = inputs.cuda()
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    predictions_evaluation_session.extend(predicted.cpu().numpy())
                    ground_truth_evaluation_session.extend(labels.numpy())
                    activations_average = torch.mean(torch.abs(inputs), dim=(1, 2, 3), dtype=torch.double)
                    # the examples which are from the neutral gesture are set to 0
                    activations_average[labels == 0] = 0.
                    if len(labels) > 1:
                        highest_activation_with_associated_labels = torch.from_numpy(np.array(
                            highest_activation_participant_for_this_training_session, dtype=np.double)[labels]).cuda()
                    else:
                        highest_activation_with_associated_labels = torch.from_numpy(np.array([np.array(
                            highest_activation_participant_for_this_training_session, dtype=np.double)[labels]])).cuda()
                    activations_average = activations_average/highest_activation_with_associated_labels
                    activation_evaluation_session.extend(activations_average.cpu().numpy())
            print("Participant: ", participant_index, " Accuracy: ",
                  np.mean(np.array(predictions_evaluation_session) == np.array(ground_truth_evaluation_session)))
            predictions_participant.append(predictions_evaluation_session)
            ground_truth_participant.append(ground_truth_evaluation_session)
            accuracies_participant.append(np.mean(np.array(predictions_evaluation_session) ==
                                                  np.array(ground_truth_evaluation_session)))
            activations_participant.append(activation_evaluation_session)

        predictions.append(predictions_participant)
        ground_truths.append(ground_truth_participant)
        accuracies.append(np.array(accuracies_participant))
        activations.append(activations_participant)
        print("ACCURACY PARTICIPANT: ", accuracies_participant)

    print(np.array(accuracies).flatten())
    accuracies_to_display = []
    for accuracies_from_participant in np.array(accuracies).flatten():
        accuracies_to_display.extend(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))

    if use_only_first_training:
        file_to_open = "../../results/evaluation_sessions_no_retraining_and_with_activations" + str(filter_size[1]) + \
                       ".txt"
        np.save("../../results/evaluation_sessions_no_retraining_and_with_activations", (ground_truths, predictions,
                                                                                         activations))
    else:
        file_to_open = "../../results/evaluation_sessions_WITH_retraining_and_activations" + str(filter_size[1]) + \
                       ".txt"
        np.save("../../results/evaluation_sessions_WITH_retraining_and_activations", (ground_truths, predictions,
                                                                                      activations))
    with open(file_to_open, "a") as \
            myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truths) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))



if __name__ == '__main__':
    print(os.listdir("../../"))
    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']


    with open("../../Processed_datasets/LongTermDataset_training_session.pickle", 'rb') as f:
        dataset_training = pickle.load(file=f)
    highest_activations = dataset_training["highest_activations"]
    print(highest_activations)
    path_weights = "../weights_full_training"
    test_network_convNet_with_activation(examples_datasets=examples_datasets_evaluation,
                                         labels_datasets=labels_datasets_evaluation, convNet=rawConvNet,
                                         highest_activation_participants_gestures=highest_activations,
                                         use_only_first_training=True, path_weights=path_weights,
                                         num_kernels=[16, 32, 64])
    test_network_convNet_with_activation(examples_datasets=examples_datasets_evaluation,
                                         labels_datasets=labels_datasets_evaluation, convNet=rawConvNet,
                                         highest_activation_participants_gestures=highest_activations,
                                         use_only_first_training=False, path_weights=path_weights,
                                         num_kernels=[16, 32, 64])


    # Graphs production
    ground_truths_no_retraining, predictions_no_retraining, activations_no_retraining = np.load(
        "../../results/evaluation_sessions_no_retraining_and_with_activations.npy")
    ground_truths_WITH_retraining, predictions_WITH_retraining, activations_WITH_retraining = np.load(
        "../../results/evaluation_sessions_WITH_retraining_and_activations.npy")

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')

    create_activation_classification_graph(predictions=predictions_no_retraining,
                                           ground_truth=ground_truths_no_retraining,
                                           activations_examples=activations_no_retraining, number_of_bins=9)
    create_activation_classification_graph(predictions=predictions_WITH_retraining,
                                           ground_truth=ground_truths_WITH_retraining,
                                           activations_examples=activations_WITH_retraining, number_of_bins=9)

    create_long_term_classification_graph(ground_truths_no_retraining=ground_truths_no_retraining,
                                          predictions_no_retraining=predictions_no_retraining,
                                          ground_truths_WITH_retraining=ground_truths_WITH_retraining,
                                          predictions_WITH_retraining=predictions_WITH_retraining,
                                          timestamps=evaluation_datetimes)

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
