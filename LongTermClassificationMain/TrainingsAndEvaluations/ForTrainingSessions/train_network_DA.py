import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet as rawConvNet
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import train_DA_convNet
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_in_dataloader import load_dataloaders_training_sessions


def test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels,
                              path_weights_normal='../weights', path_weights_DA='../weights_DANN',
                              filter_size=(4, 10), algo_name="DANN"):
    participant_train, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train,
                                                                                 labels_datasets_train, batch_size=512)

    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_test):
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        model = rawConvNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()
        print(np.shape(dataset_test))
        for session_index, training_session_test_data in enumerate(dataset_test):
            if session_index == 0:
                best_state = torch.load(
                    path_weights_normal + "/participant_%d/best_weights_participant_normal_training_%d.pt" %
                    (participant_index, 0))
            else:
                best_state = torch.load(
                    path_weights_DA + "/participant_%d/best_weights_participant_normal_training_%d.pt" %
                    (participant_index, session_index))  # There is 2 evaluation sessions per training
            best_weights = best_state['state_dict']
            model.load_state_dict(best_weights)

            predictions_training_session = []
            ground_truth_training_sesssion = []
            with torch.no_grad():
                model.eval()
                for inputs, labels in training_session_test_data:
                    inputs = inputs.cuda()
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    predictions_training_session.extend(predicted.cpu().numpy())
                    ground_truth_training_sesssion.extend(labels.numpy())
            print("Participant ID: ", participant_index, " Session ID: ", session_index, " Accuracy: ",
                  np.mean(np.array(predictions_training_session) == np.array(ground_truth_training_sesssion)))
            predictions_participant.append(predictions_training_session)
            ground_truth_participant.append(ground_truth_training_sesssion)
            accuracies_participant.append(np.mean(np.array(predictions_training_session) ==
                                                  np.array(ground_truth_training_sesssion)))
        accuracies.append(np.array(accuracies_participant))
        predictions.append(predictions_participant)
        ground_truths.append(ground_truth_participant)
        print("ACCURACY PARTICIPANT: ", accuracies_participant)
    print(np.array(accuracies).flatten())
    accuracies_to_display = []
    for accuracies_from_participant in np.array(accuracies).flatten():
        accuracies_to_display.extend(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))

    file_to_open = "../../results/test_accuracy_on_training_sessions_" + algo_name + "_" + str(
        filter_size[1]) + ".txt"
    np.save("../../results/predictions_training_session_" + algo_name, (ground_truths, predictions))
    with open(file_to_open, "a") as myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truths) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))


if __name__ == "__main__":
    print(os.listdir("../../"))
    with open("../../Processed_datasets/LongTermDataset_training_session.pickle", 'rb') as f:
        dataset_training = pickle.load(file=f)

    training_datetimes = dataset_training['training_datetimes']
    examples_datasets_train = dataset_training['examples_training']
    labels_datasets_train = dataset_training['labels_training']

    # dilated
    filter_size = (4, 10)
    num_kernels = [16, 32, 64]

    # Training and testing start
    path_weights_fine_tuning = "../weights_TWO_CYCLES_normal_training_fine_tuning"
    # algo_name = "MTDA"
    # algo_name = "MTVADA"
    algo_name = "DANN_TWO_Cycles"
    # algo_name = "VADA"
    # algo_name = "Dirt_T"
    # algo_name = "BatchNorm"

    train_DA_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                     algo_name=algo_name, path_weights_fine_tuning=path_weights_fine_tuning)
    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal=path_weights_fine_tuning)
    '''
    algo_name = "VADA_TWO_Cycles"
    train_DA_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                     algo_name=algo_name, path_weights_fine_tuning=path_weights_fine_tuning)
    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal=path_weights_fine_tuning)

    algo_name = "Dirt_T_TWO_Cycles"
    train_DA_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                     algo_name=algo_name, path_weights_to_load_from_for_dirtT='../weights_VADA_TWO_Cycles',
                     path_weights_fine_tuning=path_weights_fine_tuning)
    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal=path_weights_fine_tuning)
    '''

    '''
    algo_name = "Dirt_T_MTDA"
    train_DA_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                     algo_name=algo_name, path_weights_to_load_from_for_dirtT='../weights_MTDA')
    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../weights_' + algo_name, algo_name=algo_name)
    '''
    # Training and testing stop

    # Graphs production
    ground_truths_no_retraining, predictions_no_retraining = np.load(
        "../../results/predictions_training_session_no_retraining.npy")
    print(ground_truths_no_retraining)

    ground_truths_DANN, predictions_DANN = np.load(
        "../../results/predictions_training_session_" + "DANN" + ".npy")
    ground_truths_VADA, predictions_VADA = np.load(
        "../../results/predictions_training_session_" + "VADA" + ".npy")
    ground_truths_DirtT, predictions_DirtT = np.load(
        "../../results/predictions_training_session_" + "Dirt_T" + ".npy")
    ground_truths_MTDA, predictions_MTDA = np.load(
        "../../results/predictions_training_session_" + "MTDA" + ".npy")

    ground_truths_array = [ground_truths_no_retraining, ground_truths_DANN, ground_truths_VADA, ground_truths_DirtT,
                           ground_truths_MTDA]
    predictions_array = [predictions_no_retraining, predictions_DANN, predictions_VADA, predictions_DirtT,
                         predictions_MTDA]
    text_legend_array = ["No Re-Calibration", "DANN ", "VADA ", "Dirt-T ", "MTDA "]

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')

    long_term_classification_graph(ground_truths_in_array=ground_truths_array,
                                   predictions_in_array=predictions_array,
                                   text_for_legend_in_array=text_legend_array,
                                   timestamps=training_datetimes, number_of_seances_to_consider=5,
                                   remove_transition_evaluation=False)

    fig, axs = create_confusion_matrix(ground_truth=ground_truths_no_retraining, predictions=predictions_no_retraining,
                                       class_names=classes, title="ConvNet standard training", fontsize=font_size)

    # fig.suptitle("ConvNet using AdaDANN training", fontsize=28)
    mng_no_retraining = plt.get_current_fig_manager()
    # mng.window.state('zoomed')  # works fine on Windows!
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.gcf().subplots_adjust(top=0.90)
    plt.show()

    _, _ = create_confusion_matrix(ground_truth=ground_truths_DANN,
                                   predictions=predictions_DANN, class_names=classes,
                                   title="ConvNet standard training", fontsize=font_size)

    # fig.suptitle("ConvNet using AdaDANN training", fontsize=28)
    mng_retraining = plt.get_current_fig_manager()
    # mng.window.state('zoomed')  # works fine on Windows!
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.gcf().subplots_adjust(top=0.90)
    plt.show()

