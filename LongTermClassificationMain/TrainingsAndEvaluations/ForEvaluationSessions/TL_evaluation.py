import os
import torch
import pickle
import numpy as np
import seaborn as sns

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet as rawConvNet
from LongTermClassificationMain.Models.raw_TCN_Transfer_Learning import TargetNetwork
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.ForEvaluationSessions.get_evaluation_score_and_graph import \
    get_scores
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph, long_term_pointplot, accuracy_vs_score_lmplot

np.set_printoptions(precision=3, suppress=True)


def test_network_TL_evaluation_algorithm(examples_datasets_evaluation, labels_datasets_evaluation, num_kernels,
                                         path_weights_normal='../weights', path_weights_TL='../weights_TL',
                                         path_weights_source_network='../weights_TL_Two_Cycles_Recalibration',
                                         filter_size=(4, 10), algo_name="TransferLearning"):
    participants_test = load_dataloaders_test_sessions(
        examples_datasets_evaluation=examples_datasets_evaluation,
        labels_datasets_evaluation=labels_datasets_evaluation, batch_size=512)
    class_predictions = []
    model_outputs = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_test):
        class_predictions_participant = []
        model_outputs_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        print(np.shape(dataset_test))
        for session_index, training_session_test_data in enumerate(dataset_test):
            # if session_index % 2 != 0:
            if session_index < 2:
                model = rawConvNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()
                best_state = torch.load(
                    path_weights_normal + "/participant_%d/best_state_%d.pt" %
                    (participant_index, 0))
            else:
                # Two evaluations session per training sessions
                state_pre_training = torch.load(path_weights_source_network +
                                                "/participant_%d/best_state_participant_pre_training_%d.pt" %
                                                (participant_index, int(session_index / 2)))
                weights_pre_training = state_pre_training['state_dict']
                model = TargetNetwork(weight_pre_trained_convNet=weights_pre_training, num_kernels=num_kernels,
                                      kernel_size=filter_size).cuda()
                best_state = torch.load(
                    path_weights_TL + "/participant_%d/best_state_%d.pt" %
                    (participant_index, int(session_index / 2)))
            model.load_state_dict(best_state['state_dict'])

            predictions_training_session = []
            model_outputs_session = []
            ground_truth_training_sesssion = []
            with torch.no_grad():
                model.eval()
                for inputs, labels in training_session_test_data:
                    inputs = inputs.cuda()
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    predictions_training_session.extend(predicted.cpu().numpy())
                    # print(torch.softmax(output, dim=1).cpu().numpy(), "  ", predicted)
                    model_outputs_session.extend(torch.softmax(output, dim=1).cpu().numpy())
                    ground_truth_training_sesssion.extend(labels.numpy())
            print("Participant ID: ", participant_index, " Session ID: ", session_index, " Accuracy: ",
                  np.mean(np.array(predictions_training_session) == np.array(ground_truth_training_sesssion)))
            class_predictions_participant.append(predictions_training_session)
            model_outputs_participant.append(model_outputs_session)
            ground_truth_participant.append(ground_truth_training_sesssion)
            accuracies_participant.append(np.mean(np.array(predictions_training_session) ==
                                                  np.array(ground_truth_training_sesssion)))
        accuracies.append(np.array(accuracies_participant))
        class_predictions.append(class_predictions_participant)
        model_outputs.append(model_outputs_participant)
        ground_truths.append(ground_truth_participant)
        print("ACCURACY PARTICIPANT: ", accuracies_participant)
    print(np.array(accuracies).flatten())
    accuracies_to_display = []
    for accuracies_from_participant in np.array(accuracies).flatten():
        accuracies_to_display.extend(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))

    np.save("../../results/predictions_EVALUATION_session_" + algo_name, (ground_truths, class_predictions,
                                                                          model_outputs))
    file_to_open = "../../results/test_accuracy_on_EVALUATION_sessions_" + algo_name + "_" + str(
        filter_size[1]) + ".txt"
    with open(file_to_open, "a") as myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(class_predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truths) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))


if __name__ == '__main__':
    path_weights_normal = "../weights_TWO_CYCLES_normal_training_fine_tuning"
    path_weight_TL = "../weights_TL_Two_Cycles_Recalibration"
    path_weights_source_network = '../weights_TL_Two_Cycles_Recalibration'
    algo_name = "TADANN_EVAL"
    print(os.listdir("../../"))

    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']

    filter_size = (4, 10)
    num_kernels = [16, 32, 64]

    test_network_TL_evaluation_algorithm(examples_datasets_evaluation, labels_datasets_evaluation,
                                         filter_size=filter_size, num_kernels=num_kernels,
                                         path_weights_normal=path_weights_normal, path_weights_TL=path_weight_TL,
                                         path_weights_source_network=path_weights_source_network,
                                         algo_name=algo_name)

    ground_truths_TADANN, predictions_TADANN, _ = np.load(
        "../../results/predictions_EVALUATION_session_TADANN_EVAL.npy")
    print(ground_truths_TADANN)
    ground_truths_WITH_retraining, predictions_WITH_retraining, _ = np.load(
        "../../results/predictions_EVALUATION_session_two_cycles_normal_network_evaluation_WITH_RETRAINING.npy")
    ground_truths_no_retraining, predictions_no_retraining, _ = np.load(
        "../../results/predictions_EVALUATION_session_two_cycles_normal_network_evaluation_no_retraining.npy")
    ground_truths_delayed, predictions_delayed, _ = np.load(
        "../../results/predictions_EVALUATION_session_two_cycles_normal_network_evaluation_Delayed.npy")
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
                        timestamps=evaluation_datetimes, number_of_seances_to_consider=7,
                        remove_transition_evaluation=False, only_every_second_session=False)

    long_term_classification_graph(ground_truths_in_array=ground_truths_array,
                                   predictions_in_array=predictions_array,
                                   text_for_legend_in_array=text_legend_array,
                                   timestamps=evaluation_datetimes, number_of_seances_to_consider=7,
                                   remove_transition_evaluation=False, only_every_second_session=False)

    scores = get_scores("../../../datasets/longterm_dataset_3DC")
    accuracy_vs_score_lmplot(ground_truths_in_array=ground_truths_TADANN, predictions_in_array=predictions_TADANN,
                             scores_in_array=scores, number_of_seances_to_consider=7)
