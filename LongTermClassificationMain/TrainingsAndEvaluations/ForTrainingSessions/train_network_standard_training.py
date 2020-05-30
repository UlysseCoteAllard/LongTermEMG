import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet as rawConvNet
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import train_raw_convNet, \
    train_raw_TCN_fine_tuning
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph, long_term_pointplot
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_in_dataloader import load_dataloaders_training_sessions


def test_network_raw_convNet_on_training_sessions(examples_datasets_train, labels_datasets_train, num_kernel,
                                                  path_weights='../weights', algo_name="Normal_Training",
                                                  filter_size=(4, 10), type_of_calibration="None",
                                                  cycle_for_test=None):
    _, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                                                 batch_size=512, cycle_for_test=cycle_for_test)
    model_outputs = []
    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_test):
        model_outputs_participant = []
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        model = rawConvNet(number_of_class=11, num_kernels=num_kernel, kernel_size=filter_size).cuda()
        for session_index, training_session_test_data in enumerate(dataset_test):
            if type_of_calibration == "None":
                best_state = torch.load(
                    path_weights + "/participant_%d/best_state_%d.pt" %
                    (participant_index, 0))
            elif type_of_calibration == "Delayed":
                session_index_to_use = np.max((0, session_index - 1))
                best_state = torch.load(
                    path_weights + "/participant_%d/best_state_%d.pt" %
                    (participant_index, session_index_to_use))
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

    if type_of_calibration == "None":
        file_to_open = "../../results/test_accuracy_on_training_sessions_" + algo_name + "_no_retraining_" + str(
            filter_size[1]) + ".txt"
        np.save("../../results/predictions_training_session_" + algo_name + "_no_retraining", (ground_truths,
                                                                                               predictions,
                                                                                               model_outputs))
    elif type_of_calibration == "Delayed":
        file_to_open = "../../results/test_accuracy_on_training_sessions_" + algo_name + "_Delayed_" + str(
            filter_size[1]) + ".txt"
        np.save("../../results/predictions_training_session_" + algo_name + "_Delayed", (ground_truths,
                                                                                         predictions,
                                                                                         model_outputs))
    else:
        file_to_open = "../../results/test_accuracy_on_training_sessions_" + algo_name + "_WITH_RETRAINING_" + str(
            filter_size[1]) + ".txt"
        np.save("../../results/predictions_training_session_" + algo_name + "_WITH_RETRAINING", (ground_truths,
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


def reduce_gestures(labels, gestures_to_consider_the_same):
    new_labels = []
    for participant in labels:
        participants_labels = []
        for session in participant:
            session_labels = []
            for cycles in session:
                corrected_ground_truth = []
                for label in cycles:
                    if label in gestures_to_consider_the_same:
                        corrected_ground_truth.append(6.)
                    else:
                        corrected_ground_truth.append(label)
                session_labels.append(corrected_ground_truth)
            participants_labels.append(session_labels)
        new_labels.append(participants_labels)
    return new_labels


def get_average_activation_for_training_examples(all_examples, all_labels, highest_activation_participants_gestures):
    from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
        load_dataloaders_training_sessions
    participants_train, _, _ = load_dataloaders_training_sessions(
        all_examples, all_labels, batch_size=512,
        number_of_cycle_for_first_training=3, get_validation_set=False,
        number_of_cycles_rest_of_training=3,
        ignore_first=True, drop_last=True)
    activations_participant = []
    for participant_index, (dataset_participant, highest_activation_participant) in enumerate(
            zip(participants_train, highest_activation_participants_gestures)):
        activation_evaluation_session = []
        for session_index, dataloader_session in enumerate(dataset_participant):
            highest_activation_participant_for_this_training_session = highest_activation_participant[session_index]
            for inputs, labels in dataloader_session:
                activations_average = torch.mean(torch.abs(inputs), dim=(1, 2, 3), dtype=torch.double)
                # the examples which are from the neutral gesture are set to 0
                activations_average[labels == 0] = 0
                if len(labels) > 1:
                    highest_activation_with_associated_labels = torch.from_numpy(np.array(
                        highest_activation_participant_for_this_training_session, dtype=np.double)[labels])
                else:
                    highest_activation_with_associated_labels = torch.from_numpy(np.array([np.array(
                        highest_activation_participant_for_this_training_session, dtype=np.double)[labels]]))
                activations_average = activations_average / highest_activation_with_associated_labels
                activation_evaluation_session.extend(activations_average.cpu().numpy())
                print(activations_average.cpu().numpy())
                print(labels)
        activations_participant.extend(activation_evaluation_session)
    index_without_neutral = np.squeeze(np.nonzero(activations_participant))
    activations_participant = np.array(activations_participant)[index_without_neutral]
    print(activations_participant)
    print(np.nanmean(activations_participant), " STD: ", np.nanstd(activations_participant))


if __name__ == "__main__":
    print(os.listdir("../../"))
    with open("../../Processed_datasets/LongTermDataset_training_session.pickle", 'rb') as f:
        dataset_training = pickle.load(file=f)

    training_datetimes = dataset_training['training_datetimes']
    examples_datasets_train = dataset_training['examples_training']
    labels_datasets_train = dataset_training['labels_training']
    highest_activations = dataset_training["highest_activations"]

    # get_average_activation_for_training_examples(examples_datasets_train, labels_datasets_train, highest_activations)

    # dilated
    filter_size = (4, 10)

    # Training and testing start
    algo_name = "2_cycle_fine_tuning"
    path_to_save_to = "../weights_TWO_CYCLES_normal_training_fine_tuning"

    train_raw_TCN_fine_tuning(examples_datasets_train, labels_datasets_train, filter_size=filter_size,
                              num_kernels=[16, 32, 64], number_of_cycle_for_first_training=3,
                              number_of_cycles_rest_of_training=3, path_weight_to_save_to=path_to_save_to)

    algo_name = "2_cycle_fine_tuning"
    path_to_save_to = "../weights_TWO_CYCLES_normal_training_fine_tuning"
    test_network_raw_convNet_on_training_sessions(examples_datasets_train, labels_datasets_train,
                                                  filter_size=filter_size, type_of_calibration="None",
                                                  num_kernel=[16, 32, 64], path_weights=path_to_save_to,
                                                  algo_name=algo_name, cycle_for_test=3)
    test_network_raw_convNet_on_training_sessions(examples_datasets_train, labels_datasets_train,
                                                  filter_size=filter_size, type_of_calibration="ReCalibration",
                                                  num_kernel=[16, 32, 64], path_weights=path_to_save_to,
                                                  algo_name=algo_name, cycle_for_test=3)
    test_network_raw_convNet_on_training_sessions(examples_datasets_train, labels_datasets_train,
                                                  filter_size=filter_size, type_of_calibration="Delayed",
                                                  num_kernel=[16, 32, 64], path_weights=path_to_save_to,
                                                  algo_name=algo_name, cycle_for_test=3)

    # Training and testing stop

    # Graphs production
    font_size = 24
    sns.set(style='dark')

    ground_truths_TADANN, predictions_TADANN = np.load(
        "../../results/predictions_training_session_TADANN_Two_CYLES.npy")

    ground_truths_WITH_retraining, predictions_WITH_retraining, _ = np.load(
        "../../results/predictions_training_session_2_cycle_fine_tuning_WITH_RETRAINING.npy")

    ground_truths_no_retraining, predictions_no_retraining, _ = np.load(
        "../../results/predictions_training_session_2_cycle_fine_tuning_no_retraining.npy")

    ground_truths_delayed, predictions_delayed, _ = np.load(
        "../../results/predictions_training_session_2_cycle_fine_tuning_Delayed.npy")
    print(ground_truths_no_retraining)

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')

    ground_truths_array = [ground_truths_TADANN, ground_truths_WITH_retraining, ground_truths_delayed,
                           ground_truths_no_retraining]
    predictions_array = [predictions_TADANN, predictions_WITH_retraining, predictions_delayed,
                         predictions_no_retraining]
    text_legend_array = ["TADANN", "Re-Calibration", "Delayed Calibration", "No Calibration"]
    long_term_pointplot(ground_truths_in_array=ground_truths_array,
                        predictions_in_array=predictions_array,
                        text_for_legend_in_array=text_legend_array,
                        timestamps=training_datetimes, number_of_seances_to_consider=3,
                        remove_transition_evaluation=False, only_every_second_session=False)

    long_term_classification_graph(ground_truths_in_array=ground_truths_array,
                                   predictions_in_array=predictions_array,
                                   text_for_legend_in_array=text_legend_array,
                                   timestamps=training_datetimes, number_of_seances_to_consider=4,
                                   remove_transition_evaluation=False)

    fig, axs = create_confusion_matrix(ground_truth=ground_truths_no_retraining, predictions=predictions_no_retraining,
                                       class_names=classes, title="ConvNet standard training", fontsize=font_size)
