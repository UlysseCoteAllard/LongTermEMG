import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet as rawConvNet
from LongTermClassificationMain.TrainingsAndEvaluations.training_loops_preparations import train_DA_convNet_evaluation
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.load_dataset_in_dataloader import \
    load_dataloaders_test_sessions


def test_network_convNet_only_second_evaluation_session_for_each_training_session(examples_datasets, labels_datasets,
                                                                                  convNet, num_kernels,
                                                                                  filter_size=(4, 10),
                                                                                  path_weights="../weights_evaluation_",
                                                                                  use_only_first_training=False,
                                                                                  algo_name="DANN"):
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
            #  The first evaluation session is used to train the DA algo, load these weights.
            # The second evaluation session is used to test (the right weights will already have been loaded
            if session_index % 2 == 0:
                if use_only_first_training:
                    best_weights = torch.load(
                        path_weights + algo_name + "/participant_%d/best_weights_No_Recalibration_participant_evaluation_session_%d.pt" %
                        (participant_index, session_index))
                else:
                    best_weights = torch.load(
                        path_weights + algo_name + "/participant_%d/best_weights_WITH_Recalibration_participant_evaluation_session_%d.pt" %
                        (participant_index, session_index))
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
        file_to_open = "../../results/evaluation_sessions_" + algo_name + "_no_retraining_" + str(filter_size[1]) + ".txt"
        np.save("../../results/evaluation_sessions_" + algo_name + "_no_retraining", (ground_truths, predictions))
    else:
        file_to_open = "../../results/evaluation_sessions_" + algo_name + "_WITH_retraining_" + str(filter_size[1]) + ".txt"
        np.save("../../results/evaluation_sessions_" + algo_name + "_WITH_retraining", (ground_truths, predictions))
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

    with open("../../Processed_datasets/LongTermDataset_training_session.pickle", 'rb') as f:
        dataset_training = pickle.load(file=f)

    training_datetimes = dataset_training['training_datetimes']
    examples_datasets_train = dataset_training['examples_training']
    labels_datasets_train = dataset_training['labels_training']

    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']

    # dilated
    filter_size = (4, 10)
    num_kernels = [16, 32, 64]

    '''
    algo_name = "DANN_FOUR_Cycles"
    path_weights_fine_tuning = "../weights_FOUR_CYCLES_fine_tuning_normal_training"
    train_DA_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                     algo_name=algo_name, path_weights_fine_tuning=path_weights_fine_tuning,
                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4)
    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal="../weights_FOUR_CYCLES_fine_tuning_normal_training")

    algo_name = "DANN_ONE_Cycles"
    path_weights_fine_tuning = "../weights_single_cycle_normal_training_fine_tuning"
    train_DA_convNet(examples_datasets_train, labels_datasets_train, filter_size=filter_size, num_kernels=num_kernels,
                     algo_name=algo_name, path_weights_fine_tuning=path_weights_fine_tuning,
                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=1)
    test_network_DA_algorithm(examples_datasets_train, labels_datasets_train, num_kernels=num_kernels,
                              filter_size=filter_size, path_weights_DA='../weights_' + algo_name, algo_name=algo_name,
                              path_weights_normal="../weights_single_cycle_normal_training_fine_tuning")
    '''
    algo_name = "DANN_FOUR_Cycles_Evaluation"
    path_weights_to_load_from = "../weights_FOUR_CYCLES_normal_training_fine_tuning"
    path_weights_to_save_to = "../weights_"

    # Using weights learned by normal classifier
    train_DA_convNet_evaluation(examples_datasets_evaluation=examples_datasets_evaluation,
                                labels_datasets_evaluation=labels_datasets_evaluation,
                                examples_datasets_train=examples_datasets_train,
                                labels_datasets_train=labels_datasets_train, num_kernels=num_kernels,
                                filter_size=filter_size, path_weights_to_load=path_weights_to_load_from,
                                algo_name=algo_name, path_weights_to_save_to=path_weights_to_save_to,
                                batch_size=512, patience_increment=10, use_recalibration_data=False,
                                number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4)
    train_DA_convNet_evaluation(examples_datasets_evaluation=examples_datasets_evaluation,
                                labels_datasets_evaluation=labels_datasets_evaluation,
                                examples_datasets_train=examples_datasets_train,
                                labels_datasets_train=labels_datasets_train, num_kernels=num_kernels,
                                filter_size=filter_size, path_weights_to_load=path_weights_to_load_from,
                                algo_name=algo_name, path_weights_to_save_to=path_weights_to_save_to,
                                batch_size=512, patience_increment=10, use_recalibration_data=True,
                                number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4)
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation, convNet=rawConvNet,
        num_kernels=num_kernels, filter_size=filter_size, path_weights=path_weights_to_save_to,
        use_only_first_training=True, algo_name=algo_name)
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation, convNet=rawConvNet,
        num_kernels=num_kernels, filter_size=filter_size, path_weights=path_weights_to_save_to,
        use_only_first_training=False, algo_name=algo_name)


    algo_name = "DANN_ONE_Cycle_Evaluation"
    path_weights_to_load_from = "../weights_ONE_CYCLE_normal_training_fine_tuning"
    path_weights_to_save_to = "../weights_"

    # Using weights learned by normal classifier
    train_DA_convNet_evaluation(examples_datasets_evaluation=examples_datasets_evaluation,
                                labels_datasets_evaluation=labels_datasets_evaluation,
                                examples_datasets_train=examples_datasets_train,
                                labels_datasets_train=labels_datasets_train, num_kernels=num_kernels,
                                filter_size=filter_size, path_weights_to_load=path_weights_to_load_from,
                                algo_name=algo_name, path_weights_to_save_to=path_weights_to_save_to,
                                batch_size=512, patience_increment=10, use_recalibration_data=False,
                                number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=1)
    train_DA_convNet_evaluation(examples_datasets_evaluation=examples_datasets_evaluation,
                                labels_datasets_evaluation=labels_datasets_evaluation,
                                examples_datasets_train=examples_datasets_train,
                                labels_datasets_train=labels_datasets_train, num_kernels=num_kernels,
                                filter_size=filter_size, path_weights_to_load=path_weights_to_load_from,
                                algo_name=algo_name, path_weights_to_save_to=path_weights_to_save_to,
                                batch_size=512, patience_increment=10, use_recalibration_data=True,
                                number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=1)

    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation, convNet=rawConvNet,
        num_kernels=num_kernels, filter_size=filter_size, path_weights=path_weights_to_save_to,
        use_only_first_training=True, algo_name=algo_name)
    test_network_convNet_only_second_evaluation_session_for_each_training_session(
        examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation, convNet=rawConvNet,
        num_kernels=num_kernels, filter_size=filter_size, path_weights=path_weights_to_save_to,
        use_only_first_training=False, algo_name=algo_name)
