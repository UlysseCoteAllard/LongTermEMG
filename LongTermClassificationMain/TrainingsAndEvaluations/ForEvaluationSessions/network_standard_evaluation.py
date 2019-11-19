import os
import torch
import pickle
import numpy as np
import seaborn as sns

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    create_long_term_classification_graph


def test_network_convNet(examples_datasets, labels_datasets, convNet, num_kernels, filter_size=(4, 10),
                         path_weights='../../weights', use_only_first_training=False):
    participants_evaluation_dataloader = load_dataloaders_test_sessions(
        examples_datasets_evaluation=examples_datasets, labels_datasets_evaluation=labels_datasets, batch_size=512)
    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_participant in enumerate(participants_evaluation_dataloader):
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
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
            with torch.no_grad():
                model.eval()
                for inputs, labels in dataloader_session:
                    inputs = inputs.cuda()
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    predictions_evaluation_session.extend(predicted.cpu().numpy())
                    ground_truth_evaluation_session.extend(labels.numpy())
            print("Participant: ", participant_index, " Accuracy: ",
                  np.mean(np.array(predictions_evaluation_session) == np.array(ground_truth_evaluation_session)))
            predictions_participant.append(predictions_evaluation_session)
            ground_truth_participant.append(ground_truth_evaluation_session)
            accuracies_participant.append(np.mean(np.array(predictions_evaluation_session) ==
                                                  np.array(ground_truth_evaluation_session)))
        predictions.append(predictions_participant)
        ground_truths.append(ground_truth_participant)
        accuracies.append(np.array(accuracies_participant))
        print("ACCURACY PARTICIPANT: ", accuracies_participant)

    print(np.array(accuracies).flatten())
    accuracies_to_display = []
    for accuracies_from_participant in np.array(accuracies).flatten():
        accuracies_to_display.extend(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))

    if use_only_first_training:
        file_to_open = "../../results/evaluation_sessions_no_retraining_" + str(filter_size[1]) + ".txt"
        np.save("../../results/evaluation_sessions_no_retraining", (ground_truths, predictions))
    else:
        file_to_open = "../../results/evaluation_sessions_WITH_retraining_" + str(filter_size[1]) + ".txt"
        np.save("../../results/evaluation_sessions_WITH_retraining", (ground_truths, predictions))
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
    path_weights = "../weights_full_training"
    print(os.listdir("../../"))

    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']

    '''
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         num_kernels=[16, 32, 64], convNet=TemporalConvNet, use_only_first_training=True,
                         path_weights=path_weights)
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         num_kernels=[16, 32, 64], convNet=TemporalConvNet, use_only_first_training=False,
                         path_weights=path_weights)
    '''
    ground_truths_no_retraining, predictions_no_retraining = np.load(
        "../../results/evaluation_sessions_no_retraining.npy")
    print(ground_truths_no_retraining)
    ground_truths_WITH_retraining, predictions_WITH_retraining = np.load(
        "../../results/evaluation_sessions_WITH_retraining.npy")

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')

    create_long_term_classification_graph(ground_truths_no_retraining=ground_truths_no_retraining,
                                          predictions_no_retraining=predictions_no_retraining,
                                          ground_truths_WITH_retraining=ground_truths_WITH_retraining,
                                          predictions_WITH_retraining=predictions_WITH_retraining,
                                          timestamps=evaluation_datetimes,
                                          number_of_seances_to_consider=4)


