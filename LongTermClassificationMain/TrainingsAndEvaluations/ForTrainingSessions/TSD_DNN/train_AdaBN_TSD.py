import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.TSD_neural_network import TSD_Network
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import train_AdaBN_spectrograms
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_spectrogram_in_dataloader import load_dataloaders_training_sessions


def test_network_AdaBN_algorithm(examples_datasets_train, labels_datasets_train, num_neurons,
                                 path_weights_normal='../weights', path_weights_DA='../Weights/weights_AdaBN',
                                 algo_name="AdaBN", cycle_to_test=None,
                                 gestures_to_remove=None, number_of_classes=11,
                                 feature_vector_input_length=385):
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
        model = TSD_Network(number_of_class=number_of_classes, feature_vector_input_length=feature_vector_input_length,
                            num_neurons=num_neurons).cuda()
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

    file_to_open = "results_tsd/test_accuracy_on_training_sessions_" + algo_name + "_" + str(
        num_neurons[1]) + ".txt"
    np.save("results_tsd/predictions_training_session_" + algo_name, (ground_truths, predictions, model_outputs))
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
    with open("../../../Processed_datasets/TSD_features_set_training_session.pickle", 'rb') as f:
        dataset_training = pickle.load(file=f)

    training_datetimes = dataset_training['training_datetimes']
    examples_datasets_train = dataset_training['examples_training']
    labels_datasets_train = dataset_training['labels_training']

    # dilated
    num_neurons = [200, 200, 200]
    feature_vector_input_length = 385
    gestures_to_remove = [5, 6, 9, 10]
    gestures_to_remove = None
    number_of_class = 11
    number_of_cycle_for_first_training = 4
    number_of_cycles_rest_of_training = 4
    learning_rate = 0.002515

    # Training and testing start
    path_weight_to_save_to = "Weights_TSD/weights_"
    path_weights_fine_tuning = "Weights_TSD/weights_THREE_cycles_TSD_ELEVEN_GESTURES"
    algo_name = "AdaBN_THREE_CYCLES_11Gestures_TSD"

    train_AdaBN_spectrograms(examples_datasets_train, labels_datasets_train, filter_size=None,
                             num_kernels=num_neurons, algo_name=algo_name,
                             path_weights_fine_tuning=path_weights_fine_tuning,
                             gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class,
                             number_of_cycle_for_first_training=number_of_cycle_for_first_training,
                             number_of_cycles_rest_of_training=number_of_cycles_rest_of_training,
                             batch_size=128, spectrogram_model=False,
                             feature_vector_input_length=feature_vector_input_length,
                             path_weights_to_save_to=path_weight_to_save_to)
    '''
    test_network_AdaBN_algorithm(examples_datasets_train, labels_datasets_train, num_neurons=num_neurons,
                                 path_weights_DA='Weights_TSD/weights_' + algo_name, algo_name=algo_name,
                                 path_weights_normal=path_weights_fine_tuning,
                                 gestures_to_remove=gestures_to_remove, number_of_classes=number_of_class,
                                 cycle_to_test=3,
                                 feature_vector_input_length=feature_vector_input_length)
    '''