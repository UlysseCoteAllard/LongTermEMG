import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from LongTermClassificationMain.Models.TSD_neural_network import TSD_Network
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import load_checkpoint
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_spectrogram_in_dataloader import \
    load_dataloaders_training_sessions
from LongTermClassificationMain.TrainingsAndEvaluations.self_learning.self_learning_utils import \
    generate_dataloaders_for_SCADANN
from LongTermClassificationMain.Models.model_training_self_learning import SCADANN_BN_training


def test_network_SLADANN(examples_datasets_train, labels_datasets_train, num_neurons, feature_vector_input_length,
                         path_weights_ASR='Weights_TSD/',
                         path_weights_normal="../2_cycle_TSD_fine_tuning",
                         algo_name="SCADANN_2_CYCLES", cycle_test=None, gestures_to_remove=None,
                         number_of_classes=11):
    _, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                                                 batch_size=512, cycle_for_test=cycle_test,
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
        model = TSD_Network(number_of_class=number_of_classes, num_neurons=num_neurons,
                            feature_vector_input_length=feature_vector_input_length).cuda()
        for session_index, training_session_test_data in enumerate(dataset_test):

            if session_index == 0:
                best_state = torch.load(
                    path_weights_normal + "/participant_%d/best_state_%d.pt" %
                    (participant_index, 0))
            else:
                best_state = torch.load(
                    path_weights_ASR + "/participant_%d/best_state_%d.pt" %
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

    file_to_open = "results_tsd/test_accuracy_on_training_sessions_" + algo_name + "_no_retraining_" + str(
        num_neurons[1]) + ".txt"
    np.save("results_tsd/predictions_training_session_" + algo_name + "_no_retraining", (ground_truths,
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


def run_SCADANN_training_sessions(examples_datasets, labels_datasets, num_neurons, feature_vector_input_length,
                                  path_weights_to_save_to="../weights_SCADANN_One_cycle",
                                  path_weights_Adversarial_training="../weights_REDUCED_DANN_TSD_TWO_Cycles",
                                  path_weights_Normal_training="../Weights/weights_TSD_ELVEN_Gestures",
                                  number_of_cycle_for_first_training=1, number_of_cycles_rest_of_training=1,
                                  gestures_to_remove=None, number_of_classes=11, percentage_same_gesture_stable=0.65,
                                  learning_rate=0.001316):
    participants_train, _, _ = load_dataloaders_training_sessions(
        examples_datasets, labels_datasets, batch_size=512,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, drop_last=False, get_validation_set=False,
        shuffle=False, ignore_first=True, gestures_to_remove=gestures_to_remove)

    for participant_i in range(len(participants_train)):
        for session_j in range(1, len(participants_train[participant_i])):
            model = TSD_Network(number_of_class=number_of_classes, num_neurons=num_neurons,
                                feature_vector_input_length=feature_vector_input_length).cuda()

            # Define Loss functions
            cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

            # Define Optimizer
            learning_rate = 0.001316
            print(model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8

            model, optimizer, _, start_epoch = load_checkpoint(
                model=model, optimizer=optimizer, scheduler=None,
                filename=path_weights_Adversarial_training + "/participant_%d/best_state_%d.pt" % (participant_i,
                                                                                                   session_j))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)
            models_array = []
            for j in range(0, session_j + 1):
                model_temp = TSD_Network(number_of_class=number_of_classes, num_neurons=num_neurons,
                                         feature_vector_input_length=feature_vector_input_length).cuda()
                if j == 0:
                    model_temp, _, _, _ = load_checkpoint(
                        model=model_temp, optimizer=None, scheduler=None,
                        filename=path_weights_Normal_training + "/participant_%d/best_state_%d.pt" % (participant_i,
                                                                                                      j))
                else:
                    model_temp, _, _, _ = load_checkpoint(
                        model=model_temp, optimizer=None, scheduler=None,
                        filename=path_weights_Adversarial_training + "/participant_%d/best_state_%d.pt" % (
                            participant_i,
                            j))
                models_array.append(model_temp)
            print(np.shape(models_array))

            train_dataloader_replay, validationloader_replay, train_dataloader_pseudo, validationloader_pseudo = \
                generate_dataloaders_for_SCADANN(dataloader_sessions=participants_train[participant_i],
                                                 models=models_array,
                                                 current_session=session_j, validation_set_ratio=0.2, batch_size=64,
                                                 percentage_same_gesture_stable=percentage_same_gesture_stable)
            best_state = SCADANN_BN_training(replay_dataset_train=train_dataloader_replay,
                                             target_validation_dataset=validationloader_pseudo,
                                             target_dataset=train_dataloader_pseudo, model=model,
                                             crossEntropyLoss=cross_entropy_loss_classes,
                                             optimizer_classifier=optimizer,
                                             scheduler=scheduler, patience_increment=10, max_epochs=500,
                                             domain_loss_weight=1e-1)
            if not os.path.exists(path_weights_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weights_to_save_to + "/participant_%d" % participant_i)
            print(os.listdir(path_weights_to_save_to))
            torch.save(best_state, f=path_weights_to_save_to +
                                     "/participant_%d/best_state_%d.pt" % (participant_i, session_j))


if __name__ == '__main__':
    print(os.listdir("../../"))
    batch_size = 256

    with open("../../../Processed_datasets/TSD_features_set_training_session.pickle", 'rb') as f:
        dataset_training = pickle.load(file=f)
    examples_datasets_train = dataset_training['examples_training']
    labels_datasets_train = dataset_training['labels_training']

    num_neurons = [200, 200, 200]
    learning_rate = 0.002515
    feature_vector_input_length = 385
    gestures_to_remove = [5, 6, 9, 10]
    gestures_to_remove = None
    number_of_classes = 11
    percentage_same_gesture_stable = 0.65  # 0.65 for 11 gestures, 0.85 for 7 gestures
    path_weight_to_save_to = "Weights_TSD/weights_THREE_CYCLES_11Gestures_SCADANN"
    path_weights_start_with = "Weights_TSD/weights_DANN_THREE_CYCLES_11Gestures_TSD"
    path_weights_Normal_training = "Weights_TSD/weights_THREE_CYCLES_TSD_ELEVEN_Gestures"
    algo_name = "SCADANN_THREE_CYCLES_11Gestures_TSD"

    run_SCADANN_training_sessions(examples_datasets=examples_datasets_train, labels_datasets=labels_datasets_train,
                                  num_neurons=num_neurons, feature_vector_input_length=feature_vector_input_length,
                                  path_weights_to_save_to=path_weight_to_save_to,
                                  path_weights_Adversarial_training=path_weights_start_with,
                                  path_weights_Normal_training=path_weights_Normal_training,
                                  number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4,
                                  gestures_to_remove=gestures_to_remove, number_of_classes=number_of_classes,
                                  learning_rate=learning_rate)
    '''
    path_weights_normal_training = "Weights_TSD/weights_TWO_cycles_TSD_SEVEN_GESTURES"
    test_network_SLADANN(examples_datasets_train=examples_datasets_train, labels_datasets_train=labels_datasets_train,
                         num_neurons=num_neurons, feature_vector_input_length=feature_vector_input_length,
                         path_weights_ASR=path_weight_to_save_to, path_weights_normal=path_weights_normal_training,
                         algo_name=algo_name, cycle_test=3, gestures_to_remove=gestures_to_remove,
                         number_of_classes=number_of_classes)
    '''