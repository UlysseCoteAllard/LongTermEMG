import os
import torch
import pickle
import numpy as np
import seaborn as sns

from LongTermClassificationMain.Models.TSD_neural_network import TSD_Network
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import create_confusion_matrix, \
    long_term_classification_graph, long_term_pointplot


def evaluate_session(dataloader_session, model, participant_index,
                     path_weights, session_index, use_only_first_training, path_weights_first_state):
    if use_only_first_training:
        best_state = torch.load(
            path_weights + "/participant_%d/best_state_0.pt" % participant_index)
    else:
        if session_index < 2:
            best_state = torch.load(
                path_weights_first_state + "/participant_%d/best_state_%d.pt" %
                (participant_index, int(session_index / 2)))  # There is 2 evaluation sessions per training
        else:
            best_state = torch.load(
                path_weights + "/participant_%d/best_state_%d.pt" %
                (participant_index, int(session_index / 2)))  # There is 2 evaluation sessions per training
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


def test_network_convNet(examples_datasets, labels_datasets, convNet, num_neurons, path_weights_first_state,
                         path_weights='../../weights', use_only_first_training=False,
                         only_second_evaluation_sessions=True, algo_name="normal"):
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
        model = convNet(number_of_class=number_of_classes, feature_vector_input_length=feature_vector_input_length,
                        num_neurons=num_neurons).cuda()
        for session_index, dataloader_session in enumerate(dataset_participant):
            if only_second_evaluation_sessions and session_index % 2 != 0:
                ground_truth_evaluation_session, predictions_evaluation_session, model_outputs_session, \
                accuracy_session = evaluate_session(dataloader_session, model, participant_index, path_weights,
                                                    session_index, use_only_first_training,
                                                    path_weights_first_state=path_weights_first_state)

                predictions_participant.append(predictions_evaluation_session)
                ground_truth_participant.append(ground_truth_evaluation_session)
                accuracies_participant.append(accuracy_session)

            elif only_second_evaluation_sessions is False:
                ground_truth_evaluation_session, predictions_evaluation_session, model_outputs_session, \
                accuracy_session = evaluate_session(dataloader_session, model, participant_index, path_weights,
                                                    session_index, use_only_first_training,
                                                    path_weights_first_state=path_weights_first_state)

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
    if use_only_first_training:
        file_to_open = "Results_tsd_eval/" + algo_name + evaluations_use + "_no_retraining_" \
                       + str(num_neurons[1]) + ".txt"
        np.save("Results_tsd_eval/" + algo_name + evaluations_use + "_no_retraining", (
            ground_truths, predictions, model_outputs))
    else:
        file_to_open = "Results_tsd_eval/" + algo_name + evaluations_use + \
                       "_WITH_retraining_" + str(num_neurons[1]) + ".txt"
        np.save("Results_tsd_eval/" + algo_name + evaluations_use + "_WITH_retraining", (
            ground_truths, predictions, model_outputs))
    with open(file_to_open, "a") as myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truths) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))


if __name__ == '__main__':

    algo_name = "THREE_CYCLES_11Gestures_SCADANN"
    path_weights = "Weights_TSD_eval/weights_THREE_CYCLES_11Gestures_SCADANN"
    path_weights_first_state = "Weights_TSD_eval/weights_THREE_cycles_TSD_ELEVEN_GESTURES"
    print(os.listdir("Weights_TSD_eval/weights_THREE_cycles_TSD_ELEVEN_GESTURES"))

    with open("../../../Processed_datasets/TSD_features_set_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    num_neurons = [200, 200, 200]
    feature_vector_input_length = 385
    gestures_to_remove = None
    number_of_classes = 11

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']
    '''
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         num_neurons=num_neurons, convNet=TSD_Network, use_only_first_training=True,
                         path_weights=path_weights, path_weights_first_state=path_weights_first_state,
                         only_second_evaluation_sessions=True, algo_name=algo_name)
    
    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         num_neurons=num_neurons, convNet=TSD_Network, use_only_first_training=False,
                         path_weights=path_weights, path_weights_first_state=path_weights_first_state,
                         only_second_evaluation_sessions=True, algo_name=algo_name)
    '''
    ground_truths_SCADANN, predictions_SCADANN = np.load(
        "Results_tsd_eval/evaluation_sessions_SCADANN_Spectrogram_FirstEvaluation_no_retraining.npy",
        allow_pickle=True)

    ground_truths_multiple_vote, predictions_multiple_vote = np.load(
        "Results_tsd_eval/evaluation_sessions_MV_TSD_FirstEvaluation_no_retraining.npy", allow_pickle=True)

    ground_truths_DANN, predictions_DANN = np.load(
        "Results_tsd_eval/evaluation_sessions_DANN_Spectrogram_FirstEvaluation_no_retraining.npy",
        allow_pickle=True)

    ground_truths_VADA, predictions_VADA = np.load(
        "Results_tsd_eval/evaluation_sessions_VADA_TSD_FirstEvaluation_no_retraining.npy",
        allow_pickle=True)

    ground_truths_dirt_t, predictions_dirt_t = np.load(
        "Results_tsd_eval/evaluation_sessions_DirtT_TSD_FirstEvaluation_no_retraining.npy",
        allow_pickle=True)

    ground_truths_AdaBN, predictions_AdaBN = np.load(
        "Results_tsd_eval/evaluation_sessions_Three_cycles_AdaBN_EVAL_SESSIONS_no_retraining.npy",
        allow_pickle=True)

    ground_truths_no_retraining, predictions_no_retraining, _ = np.load(
        "Results_tsd_eval/Three_cycles_TSD_DNN_EVALUATION_SESSIONS_only_SECOND_no_retraining.npy",
        allow_pickle=True)

    ground_truths_WITH_retraining, predictions_WITH_retraining, _ = np.load(
        "Results_tsd_eval/Three_cycles_TSD_DNN_EVALUATION_SESSIONS_only_SECOND_WITH_retraining.npy",
        allow_pickle=True)

    print(ground_truths_no_retraining)
    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
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
    ground_truths_SCADANN_recalibrated, predictions_SCADANN_recalibrated = np.load(
        "Results_tsd_eval/evaluation_sessions_SCADANN_Spectrogram_FirstEvaluation_WITH_retraining.npy", allow_pickle=True)
    ground_truths_array = [ground_truths_WITH_retraining, ground_truths_SCADANN_recalibrated,
                           ground_truths_no_retraining, ground_truths_SCADANN]
    predictions_array = [predictions_WITH_retraining, predictions_SCADANN_recalibrated,
                         predictions_no_retraining, predictions_SCADANN]
    text_legend_array = ["Recalibration", "Recalibration SCADANN", "No Calibration", "SCADANN"]

    font_size = 24
    sns.set(style='dark')

    long_term_pointplot(ground_truths_in_array=ground_truths_array,
                        predictions_in_array=predictions_array,
                        text_for_legend_in_array=text_legend_array,
                        timestamps=evaluation_datetimes, number_of_seances_to_consider=2,
                        remove_transition_evaluation=False, only_every_second_session=True)

    long_term_classification_graph(ground_truths_in_array=ground_truths_array,
                                   predictions_in_array=predictions_array,
                                   text_for_legend_in_array=text_legend_array,
                                   timestamps=evaluation_datetimes, number_of_seances_to_consider=2,
                                   remove_transition_evaluation=False, only_every_second_session=True)
