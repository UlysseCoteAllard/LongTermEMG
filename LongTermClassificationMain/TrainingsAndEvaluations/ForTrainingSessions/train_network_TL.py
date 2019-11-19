import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet as rawConvNet
from LongTermClassificationMain.Models.raw_TCN_Transfer_Learning import TargetNetwork
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import train_TL_convNet, train_raw_convNet
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_in_dataloader import load_dataloaders_training_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    create_long_term_classification_graph


def test_network_TL_algorithm(examples_datasets_train, labels_datasets_train, num_kernels,
                              path_weights_normal='../weights', path_weights_TL='../weights_TL',
                              filter_size=(4, 10), algo_name="TransferLearning"):
    _, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train,
                                                                 labels_datasets_train, batch_size=512)

    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_test):
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        print(np.shape(dataset_test))
        for session_index, training_session_test_data in enumerate(dataset_test):
            if session_index == 0:
                model = rawConvNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()
                best_weights = torch.load(
                    path_weights_normal + "/participant_%d/best_weights_participant_normal_training_%d.pt" %
                    (participant_index, 0))
                model.load_state_dict(best_weights)
            else:
                weights_pre_training = torch.load(path_weights_TL +
                                                  "/participant_%d/best_weights_participant_pre_training_%d.pt" %
                                                  (participant_index, session_index - 1))
                model = TargetNetwork(weight_pre_trained_convNet=weights_pre_training, num_kernels=num_kernels,
                                      kernel_size=filter_size).cuda()
                best_weights = torch.load(path_weights_TL +
                                          "/participant_%d/best_weights_participant_normal_training_%d.pt" % (
                                              participant_index, session_index))
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

    np.save("../../results/predictions_training_session_" + algo_name, (ground_truths, predictions))
    file_to_open = "../../results/test_accuracy_on_training_sessions_" + algo_name + "_" + str(
        filter_size[1]) + ".txt"
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

    train_TL_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                     number_of_cycles_rest_of_training=4, number_of_cycle_for_first_training=4,
                     path_weight_to_save_to="../weights_TL_full_training")

    train_raw_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                      number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                      path_weight_to_save_to="../weights_full_training")
    #test_network_TL_algorithm(examples_datasets_train, labels_datasets_train, filter_size=filter_size,
    #                         num_kernels=[16, 32, 64])

    # Graphs production
    ground_truths_no_retraining, predictions_no_retraining = np.load(
        "../../results/predictions_training_session_TransferLearning.npy")
    print(ground_truths_no_retraining)
    ground_truths_WITH_retraining, predictions_WITH_retraining = np.load(
        "../../results/predictions_training_session_WITH_RETRAINING.npy")

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')

    create_long_term_classification_graph(ground_truths_no_retraining=ground_truths_no_retraining,
                                          predictions_no_retraining=predictions_no_retraining,
                                          ground_truths_WITH_retraining=ground_truths_WITH_retraining,
                                          predictions_WITH_retraining=predictions_WITH_retraining,
                                          timestamps=training_datetimes, number_of_seances_to_consider=3)

