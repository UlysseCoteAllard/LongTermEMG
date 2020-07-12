import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.TSD_neural_network import TSD_Network
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import \
    train_DA_spectrograms_evaluation
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions


def test_network_convNet_only_second_evaluation_session_for_each_training_session(examples_datasets, labels_datasets,
                                                                                  dnn, num_neurons,
                                                                                  path_weights="../weights_evaluation_",
                                                                                  use_only_first_training=False,
                                                                                  algo_name="DANN",
                                                                                  feature_vector_input_length=385):
    participants_evaluation_dataloader = load_dataloaders_test_sessions(
        examples_datasets_evaluation=examples_datasets, labels_datasets_evaluation=labels_datasets, batch_size=512)
    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_participant in enumerate(participants_evaluation_dataloader):
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        model = dnn(number_of_class=number_of_classes, num_neurons=num_neurons,
                    feature_vector_input_length=feature_vector_input_length).cuda()
        for session_index, dataloader_session in enumerate(dataset_participant):
            #  The first evaluation session is used to train the DA algo, load these weights.
            # The second evaluation session is used to test (the right weights will already have been loaded)
            if session_index % 2 == 0:
                if use_only_first_training:
                    best_state = torch.load(
                        path_weights + algo_name + "/participant_%d/best_state_NO_recalibration%d.pt" %
                        (participant_index, session_index))
                else:
                    best_state = torch.load(
                        path_weights + algo_name + "/participant_%d/best_state_WITH_recalibration%d.pt" %
                        (participant_index, session_index))
                best_weights = best_state['state_dict']
                model.load_state_dict(best_weights)
            else:
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
        file_to_open = "results_tsd_eval/evaluation_sessions_" + algo_name + "_no_retraining_" + str(
            num_neurons[1]) + ".txt"
        np.save("results_tsd_eval/evaluation_sessions_" + algo_name + "_no_retraining", (ground_truths, predictions))
    else:
        file_to_open = "results_tsd_eval/evaluation_sessions_" + algo_name + "_WITH_retraining_" + str(
            num_neurons[1]) + ".txt"
        np.save("results_tsd_eval/evaluation_sessions_" + algo_name + "_WITH_retraining", (ground_truths, predictions))
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

    with open("../../../Processed_datasets/TSD_features_set_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']

    num_neurons = [200, 200, 200]
    feature_vector_input_length = 385
    gestures_to_remove = None
    number_of_classes = 11
    learning_rate = 0.002515
    number_of_cycle_for_first_training = 4
    number_of_cycles_rest_of_training = 4

    # Training and testing start
    path_weights_fine_tuning = "Weights_TSD_eval/weights_THREE_cycles_TSD_ELEVEN_GESTURES"
    path_weights_DA = "Weights_TSD_eval/weights_"
    algo_name = "DANN_TSD_FirstEvaluation"


    # Using weights learned by normal classifier
    train_DA_spectrograms_evaluation(examples_datasets_evaluations=examples_datasets_evaluation,
                                     labels_datasets_evaluation=labels_datasets_evaluation,
                                     examples_datasets_train=examples_datasets_train,
                                     labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                     filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                     algo_name=algo_name, path_weights_DA=path_weights_DA,
                                     batch_size=512, patience_increment=10, use_recalibration_data=False,
                                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                     spectrogram_model=False, feature_vector_input_length=feature_vector_input_length,
                                     learning_rate=learning_rate)
    train_DA_spectrograms_evaluation(examples_datasets_evaluations=examples_datasets_evaluation,
                                     labels_datasets_evaluation=labels_datasets_evaluation,
                                     examples_datasets_train=examples_datasets_train,
                                     labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                     filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                     algo_name=algo_name, path_weights_DA=path_weights_DA,
                                     batch_size=512, patience_increment=10, use_recalibration_data=True,
                                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                     spectrogram_model=False, feature_vector_input_length=feature_vector_input_length,
                                     learning_rate=learning_rate)

    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_DA,
        use_only_first_training=True, algo_name=algo_name, feature_vector_input_length=feature_vector_input_length)
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_DA,
        use_only_first_training=False, algo_name=algo_name, feature_vector_input_length=feature_vector_input_length)


    # Training and testing start
    algo_name = "VADA_TSD_FirstEvaluation"

    # Using weights learned by normal classifier
    train_DA_spectrograms_evaluation(examples_datasets_evaluations=examples_datasets_evaluation,
                                     labels_datasets_evaluation=labels_datasets_evaluation,
                                     examples_datasets_train=examples_datasets_train,
                                     labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                     filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                     algo_name=algo_name, path_weights_DA=path_weights_DA,
                                     batch_size=512, patience_increment=10, use_recalibration_data=False,
                                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                     spectrogram_model=False, feature_vector_input_length=feature_vector_input_length,
                                     learning_rate=learning_rate)
    train_DA_spectrograms_evaluation(examples_datasets_evaluations=examples_datasets_evaluation,
                                     labels_datasets_evaluation=labels_datasets_evaluation,
                                     examples_datasets_train=examples_datasets_train,
                                     labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                     filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                     algo_name=algo_name, path_weights_DA=path_weights_DA,
                                     batch_size=512, patience_increment=10, use_recalibration_data=True,
                                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                     spectrogram_model=False, feature_vector_input_length=feature_vector_input_length,
                                     learning_rate=learning_rate)
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_DA,
        use_only_first_training=True, algo_name=algo_name, feature_vector_input_length=feature_vector_input_length)
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_DA,
        use_only_first_training=False, algo_name=algo_name, feature_vector_input_length=feature_vector_input_length)

    # Training and testing start
    path_weights_fine_tuning = "Weights_TSD_eval/weights_VADA_TSD_FirstEvaluation"
    algo_name = "DirtT_TSD_FirstEvaluation"

    # Using weights learned by normal classifier
    train_DA_spectrograms_evaluation(examples_datasets_evaluations=examples_datasets_evaluation,
                                     labels_datasets_evaluation=labels_datasets_evaluation,
                                     examples_datasets_train=examples_datasets_train,
                                     labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                     filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                     algo_name=algo_name, path_weights_DA=path_weights_DA,
                                     batch_size=512, patience_increment=10, use_recalibration_data=False,
                                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                     spectrogram_model=False, feature_vector_input_length=feature_vector_input_length,
                                     learning_rate=learning_rate)
    train_DA_spectrograms_evaluation(examples_datasets_evaluations=examples_datasets_evaluation,
                                     labels_datasets_evaluation=labels_datasets_evaluation,
                                     examples_datasets_train=examples_datasets_train,
                                     labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                     filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                     algo_name=algo_name, path_weights_DA=path_weights_DA,
                                     batch_size=512, patience_increment=10, use_recalibration_data=True,
                                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                     spectrogram_model=False, feature_vector_input_length=feature_vector_input_length,
                                     learning_rate=learning_rate)

    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_DA,
        use_only_first_training=True, algo_name=algo_name)
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_DA,
        use_only_first_training=False, algo_name=algo_name)
