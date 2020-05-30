import os
import torch
import pickle
import numpy as np
import seaborn as sns

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph, long_term_pointplot


def evaluate_session(dataloader_session, model, participant_index,
                     path_weights, session_index, type_of_calibration):
    if type_of_calibration == "None":
        best_state = torch.load(
            path_weights + "/participant_%d/best_state_%d.pt" %
            (participant_index, 0))
    elif type_of_calibration == "Delayed":
        session_index_to_use = np.max((0, int(session_index / 2) - 1))
        best_state = torch.load(
            path_weights + "/participant_%d/best_state_%d.pt" %
            (participant_index, session_index_to_use))
    else:
        best_state = torch.load(
            path_weights + "/participant_%d/best_state_%d.pt" %
            (participant_index, int(session_index / 2)))
    best_weights = best_state['state_dict']
    model.load_state_dict(best_weights)
    predictions_evaluation_session = []
    model_outputs_session = []
    ground_truth_evaluation_session = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloader_session:
            inputs = inputs.cuda()
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            predictions_evaluation_session.extend(predicted.cpu().numpy())
            model_outputs_session.extend(torch.softmax(output, dim=1).cpu().numpy())
            ground_truth_evaluation_session.extend(labels.numpy())
    print("Participant: ", participant_index, " Accuracy: ",
          np.mean(np.array(predictions_evaluation_session) == np.array(ground_truth_evaluation_session)))
    accuracy_session = np.mean(np.array(predictions_evaluation_session) == np.array(ground_truth_evaluation_session))
    return ground_truth_evaluation_session, predictions_evaluation_session, model_outputs_session, accuracy_session


def test_network_convNet(examples_datasets, labels_datasets, convNet, num_kernels, filter_size=(4, 10),
                         path_weights='../../weights', type_of_calibration="None",
                         only_second_evaluation_sessions=False, algo_name="normal"):
    participants_evaluation_dataloader = load_dataloaders_test_sessions(
        examples_datasets_evaluation=examples_datasets, labels_datasets_evaluation=labels_datasets, batch_size=512)
    model_outputs = []
    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_participant in enumerate(participants_evaluation_dataloader):
        model_outputs_participant = []
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        model = convNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()
        for session_index, dataloader_session in enumerate(dataset_participant):
            if only_second_evaluation_sessions and session_index % 2 != 0:
                ground_truth_evaluation_session, predictions_evaluation_session, model_outputs_session, \
                accuracy_session = evaluate_session(dataloader_session, model, participant_index, path_weights,
                                                    session_index, type_of_calibration)

                predictions_participant.append(predictions_evaluation_session)
                ground_truth_participant.append(ground_truth_evaluation_session)
                accuracies_participant.append(accuracy_session)

            elif only_second_evaluation_sessions is False:
                ground_truth_evaluation_session, predictions_evaluation_session, model_outputs_session, \
                accuracy_session = evaluate_session(dataloader_session, model, participant_index, path_weights,
                                                    session_index, type_of_calibration)

                predictions_participant.append(predictions_evaluation_session)
                model_outputs_participant.append(model_outputs_session)
                ground_truth_participant.append(ground_truth_evaluation_session)
                accuracies_participant.append(np.mean(np.array(predictions_evaluation_session) ==
                                                      np.array(ground_truth_evaluation_session)))
        predictions.append(predictions_participant)
        model_outputs.append(model_outputs_participant)
        ground_truths.append(ground_truth_participant)
        accuracies.append(np.array(accuracies_participant))
        print("ACCURACY PARTICIPANT: ", accuracies_participant)

    print(np.array(accuracies).flatten())
    accuracies_to_display = []
    for accuracies_from_participant in np.array(accuracies).flatten():
        accuracies_to_display.extend(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))

    if only_second_evaluation_sessions:
        evaluations_use = "_only_SECOND"
    else:
        evaluations_use = ""
    if type_of_calibration == "None":
        file_to_open = "../../results/test_accuracy_on_EVALUATION_sessions_" + algo_name + "_no_retraining_" + evaluations_use + str(
            filter_size[1]) + ".txt"
        np.save("../../results/predictions_EVALUATION_session_" + algo_name + "_no_retraining" + evaluations_use,
                (ground_truths,
                 predictions,
                 model_outputs))
    elif type_of_calibration == "Delayed":
        file_to_open = "../../results/test_accuracy_on_EVALUATION_sessions_" + algo_name + "_Delayed_" + evaluations_use + str(
            filter_size[1]) + ".txt"
        np.save("../../results/predictions_EVALUATION_session_" + algo_name + "_Delayed" + evaluations_use,
                (ground_truths,
                 predictions,
                 model_outputs))
    else:
        file_to_open = "../../results/test_accuracy_on_EVALUATION_sessions_" + algo_name + "_WITH_RETRAINING_" + evaluations_use + str(
            filter_size[1]) + ".txt"
        np.save("../../results/predictions_EVALUATION_session_" + algo_name + "_WITH_RETRAINING" + evaluations_use,
                (ground_truths,
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


if __name__ == '__main__':
    print(os.listdir("../../"))

    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']

    algo_name = "two_cycles_normal_network_evaluation"
    path_weights = "../weights_TWO_CYCLES_normal_training_fine_tuning"
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         num_kernels=[16, 32, 64], convNet=TemporalConvNet, type_of_calibration="None",
                         path_weights=path_weights, only_second_evaluation_sessions=False, algo_name=algo_name)
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         num_kernels=[16, 32, 64], convNet=TemporalConvNet, type_of_calibration="ReCalibration",
                         path_weights=path_weights, only_second_evaluation_sessions=False, algo_name=algo_name)
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         num_kernels=[16, 32, 64], convNet=TemporalConvNet, type_of_calibration="Delayed",
                         path_weights=path_weights, only_second_evaluation_sessions=False, algo_name=algo_name)

    ground_truths_no_retraining, predictions_no_retraining, _ = np.load(
        "../../results/two_cycles_normal_network_evaluation_evaluation_sessions_no_retraining.npy")
    print(ground_truths_no_retraining)
    ground_truths_WITH_retraining, predictions_WITH_retraining, _ = np.load(
        "../../results/two_cycles_normal_network_evaluation_evaluation_sessions_WITH_retraining.npy")

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')

    ground_truths_array = [ground_truths_WITH_retraining, ground_truths_no_retraining]
    predictions_array = [predictions_WITH_retraining, predictions_no_retraining]
    text_legend_array = ["Re-Calibration ", "No Calibration "]
    long_term_pointplot(ground_truths_in_array=ground_truths_array,
                        predictions_in_array=predictions_array,
                        text_for_legend_in_array=text_legend_array,
                        timestamps=evaluation_datetimes, number_of_seances_to_consider=6,
                        remove_transition_evaluation=False, only_every_second_session=False)

    long_term_pointplot(ground_truths_in_array=ground_truths_array,
                        predictions_in_array=predictions_array,
                        text_for_legend_in_array=text_legend_array,
                        timestamps=evaluation_datetimes, number_of_seances_to_consider=6,
                        remove_transition_evaluation=True, only_every_second_session=False)

    long_term_classification_graph(ground_truths_in_array=ground_truths_array,
                                   predictions_in_array=predictions_array,
                                   text_for_legend_in_array=text_legend_array,
                                   timestamps=evaluation_datetimes, number_of_seances_to_consider=8,
                                   remove_transition_evaluation=False, only_every_second_session=False)
