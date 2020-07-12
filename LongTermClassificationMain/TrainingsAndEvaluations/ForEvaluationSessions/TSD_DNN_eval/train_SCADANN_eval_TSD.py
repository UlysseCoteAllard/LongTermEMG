import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from LongTermClassificationMain.Models.TSD_neural_network import TSD_Network
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_spectrogram_in_dataloader import \
    load_dataloaders_training_sessions as load_dataloaders_training_sessions_spectrogram
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import load_checkpoint
from LongTermClassificationMain.TrainingsAndEvaluations.self_learning.self_learning_utils import \
    generate_dataloaders_evaluation_for_SCADANN
from LongTermClassificationMain.Models.model_training_self_learning import SCADANN_BN_training


def test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets, labels_datasets,
        dnn, num_neurons,
        path_weights="../Weights/weights_DANN_Spectrogram_FirstEvaluation",
        use_only_first_training=False,
        algo_name="SCADANN", feature_vector_input_length=385):
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
            # The second evaluation session is used to test (the right weights will already have been loaded
            if session_index % 2 == 0:
                if use_only_first_training or session_index == 0:
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


def run_SCADANN_evaluation_sessions(examples_datasets_evaluations, labels_datasets_evaluation,
                                    examples_datasets_train, labels_datasets_train, algo_name,
                                    num_kernels, filter_size, path_weights_to_load_from, path_weights_SCADANN,
                                    batch_size=512, patience_increment=10, use_recalibration_data=False,
                                    number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                    feature_vector_input_length=385, learning_rate=0.001316):
    # Get the data to use as the SOURCE from the training sessions
    participants_train, _, _ = load_dataloaders_training_sessions_spectrogram(
        examples_datasets_train, labels_datasets_train, batch_size=batch_size,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training, get_validation_set=False,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, gestures_to_remove=None,
        ignore_first=True, shuffle=False, drop_last=False)

    # Get the data to use as the TARGET from the evaluation sessions
    participants_evaluation_dataloader = load_dataloaders_test_sessions(
        examples_datasets_evaluation=examples_datasets_evaluations,
        labels_datasets_evaluation=labels_datasets_evaluation, batch_size=batch_size, shuffle=False, drop_last=False)

    for participant_i in range(len(participants_evaluation_dataloader)):
        print("SHAPE SESSIONS: ", np.shape(participants_evaluation_dataloader[participant_i]))
        for session_j in range(0, len(participants_evaluation_dataloader[participant_i])):
            # There is two evaluation session for every training session. We train on the first one
            if session_j % 2 == 0:
                # Classifier and discriminator
                model = TSD_Network(number_of_class=number_of_classes, num_neurons=num_kernels,
                                    feature_vector_input_length=feature_vector_input_length).cuda()
                # loss functions
                crossEntropyLoss = nn.CrossEntropyLoss().cuda()
                # optimizer
                precision = 1e-8
                optimizer_classifier = optim.Adam(model.parameters(), lr=learning_rate,
                                                  betas=(0.5, 0.999))
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_classifier, mode='min', factor=.2,
                                                                 patience=5, verbose=True, eps=precision)

                if use_recalibration_data:
                    model, optimizer_classifier, _, start_epoch = load_checkpoint(
                        model=model, optimizer=optimizer_classifier, scheduler=None,
                        filename=path_weights_to_load_from +
                                 "/participant_%d/best_state_WITH_recalibration%d.pt" %
                                 (participant_i, session_j))
                    models_array = []
                    for j in range(0, int(session_j / 2) + 1):
                        model_temp = TSD_Network(number_of_class=number_of_classes, num_neurons=num_kernels,
                                                 feature_vector_input_length=feature_vector_input_length).cuda()
                        model_temp, _, _, _ = load_checkpoint(
                            model=model_temp, optimizer=None, scheduler=None,
                            filename=path_weights_to_load_from + "/participant_%d/best_state_WITH_recalibration%d.pt" %
                                     (participant_i, int(j * 2)))
                        models_array.append(model_temp)
                else:
                    model, optimizer_classifier, _, start_epoch = load_checkpoint(
                        model=model, optimizer=optimizer_classifier, scheduler=None,
                        filename=path_weights_to_load_from +
                                 "/participant_%d/best_state_NO_recalibration%d.pt" %
                                 (participant_i, session_j))

                    models_array = []
                    for j in range(0, int(session_j / 2) + 1):
                        model_temp = TSD_Network(number_of_class=number_of_classes, num_neurons=num_kernels,
                                                 feature_vector_input_length=feature_vector_input_length).cuda()
                        model_temp, _, _, _ = load_checkpoint(
                            model=model_temp, optimizer=None, scheduler=None,
                            filename=path_weights_to_load_from + "/participant_%d/best_state_NO_recalibration%d.pt" % (
                                participant_i, int(j * 2)))
                        models_array.append(model_temp)

                corresponding_training_session_index = 0 if use_recalibration_data is False else int(session_j / 2)
                train_dataloader_replay, validationloader_replay, train_dataloader_pseudo, validationloader_pseudo = \
                    generate_dataloaders_evaluation_for_SCADANN(
                        dataloader_session_training=participants_train[participant_i][
                            corresponding_training_session_index],
                        dataloader_sessions_evaluation=participants_evaluation_dataloader[participant_i],
                        models=models_array,
                        current_session=session_j, validation_set_ratio=0.2,
                        batch_size=512, use_recalibration_data=use_recalibration_data)

                best_state = SCADANN_BN_training(replay_dataset_train=train_dataloader_replay,
                                                 target_validation_dataset=validationloader_pseudo,
                                                 target_dataset=train_dataloader_pseudo, model=model,
                                                 crossEntropyLoss=crossEntropyLoss,
                                                 optimizer_classifier=optimizer_classifier,
                                                 scheduler=scheduler, patience_increment=patience_increment,
                                                 max_epochs=500,
                                                 domain_loss_weight=1e-1)
                if use_recalibration_data:
                    if not os.path.exists(path_weights_SCADANN + algo_name + "/participant_%d" % participant_i):
                        os.makedirs(path_weights_SCADANN + algo_name + "/participant_%d" % participant_i)
                    torch.save(best_state, f=path_weights_SCADANN + algo_name +
                                             "/participant_%d/best_state_WITH_recalibration%d.pt" %
                                             (participant_i, session_j))
                else:
                    if not os.path.exists(path_weights_SCADANN + algo_name + "/participant_%d" % participant_i):
                        os.makedirs(path_weights_SCADANN + algo_name + "/participant_%d" % participant_i)
                    print(os.listdir(path_weights_SCADANN + algo_name))
                    torch.save(best_state, f=path_weights_SCADANN + algo_name +
                                             "/participant_%d/best_state_NO_recalibration%d.pt" % (
                                                 participant_i, session_j))


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
    path_weights_fine_tuning = "Weights_TSD_eval/weights_DANN_TSD_FirstEvaluation"
    path_weights_SCADANN = "Weights_TSD_eval/weights_"
    algo_name = "SCADANN_Spectrogram_FirstEvaluation"
    '''
    # Using weights learned by normal classifier
    run_SCADANN_evaluation_sessions(examples_datasets_evaluations=examples_datasets_evaluation,
                                    labels_datasets_evaluation=labels_datasets_evaluation,
                                    examples_datasets_train=examples_datasets_train,
                                    labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                    filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                    algo_name=algo_name, path_weights_SCADANN=path_weights_SCADANN,
                                    batch_size=512, patience_increment=10, use_recalibration_data=False,
                                    number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                    feature_vector_input_length=feature_vector_input_length,
                                    learning_rate=learning_rate)

    run_SCADANN_evaluation_sessions(examples_datasets_evaluations=examples_datasets_evaluation,
                                    labels_datasets_evaluation=labels_datasets_evaluation,
                                    examples_datasets_train=examples_datasets_train,
                                    labels_datasets_train=labels_datasets_train, num_kernels=num_neurons,
                                    filter_size=None, path_weights_to_load_from=path_weights_fine_tuning,
                                    algo_name=algo_name, path_weights_SCADANN=path_weights_SCADANN,
                                    batch_size=512, patience_increment=10, use_recalibration_data=True,
                                    number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                    feature_vector_input_length=feature_vector_input_length,
                                    learning_rate=learning_rate)
    '''
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_SCADANN,
        use_only_first_training=True, algo_name=algo_name)

    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
        dnn=TSD_Network,
        num_neurons=num_neurons, path_weights=path_weights_SCADANN,
        use_only_first_training=False, algo_name=algo_name)
