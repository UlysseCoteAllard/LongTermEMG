import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.spectrogram_ConvNet import SpectrogramConvNet
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import train_DA_spectrograms
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_spectrogram_in_dataloader import load_dataloaders_training_sessions


def test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels,
                              path_weights_normal='../weights', path_weights_DA='../weights_DANN',
                              filter_size=(4, 10), algo_name="DANN", cycle_to_test=None,
                              gestures_to_remove=None, number_of_classes=11):
    participant_train, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train,
                                                                                 labels_datasets_train, batch_size=512,
                                                                                 cycle_for_test=cycle_to_test,
                                                                                 gestures_to_remove=gestures_to_remove)
    model_outputs = []
    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_test):
        model_outputs_participant = []
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        model = SpectrogramConvNet(number_of_class=number_of_classes, num_kernels=num_kernels,
                                   kernel_size=filter_size).cuda()
        print(np.shape(dataset_test))
        for session_index, training_session_test_data in enumerate(dataset_test):
            if session_index == 0:
                best_state = torch.load(
                    path_weights_normal + "/participant_%d/best_state_%d.pt" %
                    (participant_index, 0))
            else:
                best_state = torch.load(
                    path_weights_DA + "/participant_%d/best_state_%d.pt" %
                    (participant_index, session_index))  # There is 2 evaluation sessions per training
            best_weights = best_state['state_dict']
            model.load_state_dict(best_weights)

            model_outputs_session = []
            predictions_training_session = []
            ground_truth_training_sesssion = []
            with torch.no_grad():
                model.eval()
                for inputs, labels in training_session_test_data:
                    inputs = inputs.cuda()
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    model_outputs_session.extend(torch.softmax(output, dim=1).cpu().numpy())
                    predictions_training_session.extend(predicted.cpu().numpy())
                    ground_truth_training_sesssion.extend(labels.numpy())
            print("Participant ID: ", participant_index, " Session ID: ", session_index, " Accuracy: ",
                  np.mean(np.array(predictions_training_session) == np.array(ground_truth_training_sesssion)))
            predictions_participant.append(predictions_training_session)
            model_outputs_participant.append(model_outputs_session)
            ground_truth_participant.append(ground_truth_training_sesssion)
            accuracies_participant.append(np.mean(np.array(predictions_training_session) ==
                                                  np.array(ground_truth_training_sesssion)))
        accuracies.append(np.array(accuracies_participant))
        predictions.append(predictions_participant)
        model_outputs.append(model_outputs_participant)
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
    np.save("../../results/predictions_training_session_" + algo_name, (ground_truths, predictions, model_outputs))
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
    with open("../../Processed_datasets/Spectrograms_training_session.pickle", 'rb') as f:
        dataset_training = pickle.load(file=f)

    training_datetimes = dataset_training['training_datetimes']
    examples_datasets_train = dataset_training['examples_training']
    labels_datasets_train = dataset_training['labels_training']

    # dilated
    filter_size = [[4, 7], [3, 7], [3, 7], [3, 6]]
    num_kernels = [16, 32, 64, 128]
    #gestures_to_remove = [5, 6, 9, 10]
    gestures_to_remove = None
    number_of_class = 11
    number_of_cycle_for_first_training = 4
    number_of_cycles_rest_of_training = 4

    # Training and testing start
    path_weights_fine_tuning = "../Weights/weights_THREE_CYCLES_SPECTROGRAMS_ELEVEN_Gestures"
    algo_name = "DANN_THREE_CYCLES_11Gestures_Spectrogram"

    train_DA_spectrograms(examples_datasets_train, labels_datasets_train, filter_size=filter_size,
                          num_kernels=num_kernels, algo_name=algo_name,
                          path_weights_fine_tuning=path_weights_fine_tuning,
                          gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class,
                          number_of_cycle_for_first_training=number_of_cycle_for_first_training,
                          number_of_cycles_rest_of_training=number_of_cycles_rest_of_training,
                          batch_size=128)

    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../Weights/weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal=path_weights_fine_tuning,
                              gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class, cycle_to_test=3)

    algo_name = "VADA_THREE_CYCLES__11Gestures_Spectrogram"

    train_DA_spectrograms(examples_datasets_train, labels_datasets_train, filter_size=filter_size,
                          num_kernels=num_kernels,
                          algo_name=algo_name, path_weights_fine_tuning=path_weights_fine_tuning,
                          gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class,
                          number_of_cycle_for_first_training=number_of_cycle_for_first_training,
                          number_of_cycles_rest_of_training=number_of_cycles_rest_of_training)

    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../Weights/weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal=path_weights_fine_tuning,
                              gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class, cycle_to_test=3)

    algo_name = "Dirt_T_THREE_CYCLES_11Gestures_Spectrogram"
    train_DA_spectrograms(examples_datasets_train, labels_datasets_train, filter_size=filter_size,
                          num_kernels=num_kernels,
                          algo_name=algo_name, path_weights_to_load_from_for_dirtT='../Weights/weights_VADA_THREE_CYCLES__11Gestures_Spectrogram',
                          path_weights_fine_tuning=path_weights_fine_tuning,
                          gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class,
                          number_of_cycle_for_first_training=number_of_cycle_for_first_training,
                          number_of_cycles_rest_of_training=number_of_cycles_rest_of_training)
    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../Weights/weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal=path_weights_fine_tuning,
                              gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class, cycle_to_test=3)
    # Training and testing stop
