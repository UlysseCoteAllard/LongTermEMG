import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.TSD_neural_network import TSD_Network
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import train_Spectrogram_fine_tuning
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_spectrogram_in_dataloader import load_dataloaders_training_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph, long_term_pointplot


def test_TSD_DNN_on_training_sessions(examples_datasets_train, labels_datasets_train, num_neurons,
                                      feature_vector_input_length=385,
                                      path_weights='../weights', algo_name="Normal_Training",
                                      use_only_first_training=False, cycle_for_test=None,
                                      gestures_to_remove=None, number_of_classes=11):
    _, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                                                 batch_size=512, cycle_for_test=cycle_for_test,
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
        for session_index, training_session_test_data in enumerate(dataset_test):
            if use_only_first_training:
                best_state = torch.load(
                    path_weights + "/participant_%d/best_state_%d.pt" %
                    (participant_index, 0))
            else:
                best_state = torch.load(
                    path_weights + "/participant_%d/best_state_%d.pt" %
                    (participant_index, session_index))
            best_weights = best_state['state_dict']
            model.load_state_dict(best_weights)

            predictions_training_session = []
            ground_truth_training_sesssion = []
            model_outputs_session = []
            with torch.no_grad():
                model.eval()
                for inputs, labels in training_session_test_data:
                    inputs = inputs.cuda()
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    model_outputs_session.extend(torch.softmax(output, dim=1).cpu().numpy())
                    predictions_training_session.extend(predicted.cpu().numpy())
                    ground_truth_training_sesssion.extend(labels.numpy())
            print("Participant: ", participant_index, " Accuracy: ",
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

    if use_only_first_training:
        file_to_open = "results_tsd/test_accuracy_on_training_sessions_" + algo_name + "_no_retraining_" + str(
            filter_size[1]) + ".txt"
        np.save("results_tsd/predictions_training_session_" + algo_name + "_no_retraining", (ground_truths,
                                                                                               predictions,
                                                                                               model_outputs))
    else:
        file_to_open = "results_tsd/test_accuracy_on_training_sessions_" + algo_name + "_WITH_RETRAINING_" + str(
            filter_size[1]) + ".txt"
        np.save("results_tsd/predictions_training_session_" + algo_name + "_WITH_RETRAINING", (ground_truths,
                                                                                                 predictions,
                                                                                                 model_outputs))
    with open(file_to_open, "a") as \
            myfile:
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

    algo_name = "11Gestures_standard_ConvNet_THREE_Cycles_TSD"
    path_to_save_to = "Weights_TSD/weights_THREE_cycles_TSD_ELEVEN_GESTURES"

    filter_size = [200, 200, 200]
    feature_vector_input_length = 385
    gestures_to_remove = [5, 6, 9, 10]
    gestures_to_remove = None
    number_of_classes = 11
    learning_rate = 0.002515
    '''
    train_Spectrogram_fine_tuning(examples_datasets_train, labels_datasets_train, filter_size=None,
                                  num_kernels=filter_size, number_of_cycle_for_first_training=4,
                                  number_of_cycles_rest_of_training=4, path_weight_to_save_to=path_to_save_to,
                                  gestures_to_remove=gestures_to_remove, number_of_classes=number_of_classes,
                                  batch_size=128, spectrogram_model=False,
                                  feature_vector_input_length=feature_vector_input_length,
                                  learning_rate=learning_rate)

    test_TSD_DNN_on_training_sessions(examples_datasets_train, labels_datasets_train,
                                      num_neurons=filter_size, use_only_first_training=True,
                                      path_weights=path_to_save_to,
                                      feature_vector_input_length=feature_vector_input_length,
                                      algo_name=algo_name, gestures_to_remove=gestures_to_remove,
                                      number_of_classes=number_of_classes, cycle_for_test=3)

    test_TSD_DNN_on_training_sessions(examples_datasets_train, labels_datasets_train,
                                      num_neurons=filter_size, use_only_first_training=False,
                                      path_weights=path_to_save_to,
                                      feature_vector_input_length=feature_vector_input_length,
                                      algo_name=algo_name, gestures_to_remove=gestures_to_remove,
                                      number_of_classes=number_of_classes, cycle_for_test=3)
    '''
    ground_truths_SCADANN, predictions_SCADANN, _ = np.load(
        "results_tsd/predictions_training_session_SCADANN_TWO_CYCLES_11Gestures_TSD_no_retraining.npy",
        allow_pickle=True)

    ground_truths_multiple_vote, predictions_multiple_vote, _ = np.load(
        "results_tsd/predictions_training_session_MultipleVote_TWO_CYCLES_11Gestures_no_retraining.npy",
        allow_pickle=True)

    ground_truths_DANN, predictions_DANN, _ = np.load(
        "results_tsd/predictions_training_session_DANN_TWO_CYCLES_11Gestures_TSD.npy",
        allow_pickle=True)

    ground_truths_VADA, predictions_VADA, _ = np.load(
        "results_tsd/predictions_training_session_VADA_TWO_CYCLES_11Gestures_TSD.npy",
        allow_pickle=True)

    ground_truths_dirt_t, predictions_dirt_t, _ = np.load(
        "results_tsd/predictions_training_session_Dirt_T_TWO_CYCLES_11Gestures_TSD.npy",
        allow_pickle=True)

    ground_truths_AdaBN, predictions_AdaBN, _ = np.load(
        "results_tsd/predictions_training_session_AdaBN_TWO_CYCLES_11Gestures_TSD.npy",
        allow_pickle=True)

    ground_truths_WITH_retraining, predictions_WITH_retraining, _ = np.load(
        "results_tsd/predictions_training_session_11Gestures_standard_ConvNet_TWO_Cycles_TSD_WITH_RETRAINING.npy",
        allow_pickle=True)

    ground_truths_no_retraining, predictions_no_retraining, _ = np.load(
        "results_tsd/predictions_training_session_11Gestures_standard_ConvNet_TWO_Cycles_TSD_no_retraining.npy",
        allow_pickle=True)

    print(ground_truths_no_retraining)



    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
    '''
    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Power Grip",
               "Open Hand"]
    '''
    font_size = 24
    sns.set(style='dark')

    ground_truths_array = [ground_truths_WITH_retraining, ground_truths_no_retraining, ground_truths_SCADANN]
    predictions_array = [predictions_WITH_retraining, predictions_no_retraining, predictions_SCADANN]
    text_legend_array = ["Recalibration", "No Calibration", "SCADANN"]
    '''
    ground_truths_array = [ground_truths_WITH_retraining, ground_truths_SCADANN, ground_truths_multiple_vote,
                           ground_truths_DANN, ground_truths_VADA, ground_truths_dirt_t, ground_truths_AdaBN,
                           ground_truths_no_retraining]
    predictions_array = [predictions_WITH_retraining, predictions_SCADANN, predictions_multiple_vote,
                         predictions_DANN, predictions_VADA, predictions_dirt_t, predictions_AdaBN,
                         predictions_no_retraining]
    text_legend_array = ["Re-Calibration", "SCADANN", "MultipleVote", "DANN", "VADA", "DIRT-T", "AdaBN",
                         "No Calibration"]
    '''
    long_term_pointplot(ground_truths_in_array=ground_truths_array,
                        predictions_in_array=predictions_array,
                        text_for_legend_in_array=text_legend_array,
                        timestamps=training_datetimes, number_of_seances_to_consider=2,
                        remove_transition_evaluation=False)

    long_term_classification_graph(ground_truths_in_array=ground_truths_array,
                                   predictions_in_array=predictions_array,
                                   text_for_legend_in_array=text_legend_array,
                                   timestamps=training_datetimes, number_of_seances_to_consider=2,
                                   remove_transition_evaluation=False)

    fig, axs = create_confusion_matrix(ground_truth=ground_truths_SCADANN, predictions=predictions_SCADANN,
                                       class_names=classes, title="ConvNet standard training", fontsize=font_size)
    plt.show()
