import time
import copy
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset

import TransferLearning.PrepareAndLoadData.load_NinaPro_dataset as load_NinaPro_dataset
from TransferLearning.Models import target_network_raw_emg_enhanced, models_training


def confusion_matrix(pred, Y, number_class=7):
    confusion_matrice = []
    for x in range(0, number_class):
        vector = []
        for y in range(0, number_class):
            vector.append(0)
        confusion_matrice.append(vector)
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1
    return np.array(confusion_matrice)


def scramble(examples, labels, second_labels=[]):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    if len(second_labels) == len(labels):
        new_second_labels = []
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
            new_second_labels.append(second_labels[i])
        return new_examples, new_labels, new_second_labels
    else:
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
        return new_examples, new_labels


def run_pre_training_for_NinaPro_cycle(examples, labels):
    for index_not_to_use in range(len(labels)):
        list_train_dataloader = []
        list_val_dataloader = []
        human_number = 0

        X_training, Y_gesture, Y_human = [], [], []
        X_validation, Y_gesture_validation, Y_human_validation = [], [], []
        for participant_index in range(len(labels)):  # Go over all the participants
            if participant_index != index_not_to_use:
                examples_personne_training = []
                labels_gesture_personne_training = []
                labels_human_personne_training = []

                examples_personne_valid = []
                labels_gesture_personne_valid = []
                labels_human_personne_valid = []

                last_label = labels[participant_index][0][0]
                nmbr_time_with_label = -1
                for j in range(
                        len(labels[participant_index])):  # Collect training data up to the number of cycles allowed
                    if last_label == labels[participant_index][j][0]:
                        nmbr_time_with_label += 1
                    else:
                        nmbr_time_with_label = 0
                    if nmbr_time_with_label < 3:
                        for k in range(len(examples[participant_index][j])):
                            examples_personne_training.append(examples[participant_index][j][k])
                        labels_gesture_personne_training.extend(labels[participant_index][j])
                        if participant_index > index_not_to_use:  # We skipped a label, which we need to take into acount.
                            labels_human_personne_training.extend(
                                (participant_index - 1) * np.ones(len(labels[participant_index][j])))
                        else:
                            labels_human_personne_training.extend(
                                participant_index * np.ones(len(labels[participant_index][j])))
                    else:
                        for k in range(len(examples[participant_index][j])):
                            examples_personne_valid.append(examples[participant_index][j][k])
                        labels_gesture_personne_valid.extend(labels[participant_index][j])
                        if participant_index > index_not_to_use:  # We skipped a label, which we need to take into acount.
                            labels_human_personne_valid.extend(
                                (participant_index - 1) * np.ones(len(labels[participant_index][j])))
                        else:
                            labels_human_personne_valid.extend(
                                participant_index * np.ones(len(labels[participant_index][j])))

                    last_label = labels[participant_index][j][0]

                examples_personne_scrambled, labels_gesture_personne_scrambled, labels_human_personne_scrambled = scramble(
                    examples_personne_training, labels_gesture_personne_training, labels_human_personne_training)

                examples_personne_scrambled_valid, labels_gesture_personne_scrambled_valid, labels_human_personne_scrambled_valid = scramble(
                    examples_personne_valid, labels_gesture_personne_valid, labels_human_personne_valid)

                X_training.append(examples_personne_scrambled)
                Y_gesture.append(labels_gesture_personne_scrambled)
                Y_human.append(labels_human_personne_scrambled)

                X_validation.append(examples_personne_scrambled_valid)
                Y_gesture_validation.append(labels_gesture_personne_scrambled_valid)
                Y_human_validation.append(labels_human_personne_scrambled_valid)

                print(np.shape(examples_personne_training))
                examples_personne_scrambled, labels_gesture_personne_scrambled, labels_human_personne_scrambled = scramble(
                    examples_personne_training, labels_gesture_personne_training, labels_human_personne_training)

                examples_personne_scrambled_valid, labels_gesture_personne_scrambled_valid, \
                labels_human_personne_scrambled_valid = scramble(
                    examples_personne_valid, labels_gesture_personne_valid, labels_human_personne_valid)

                train = TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled, dtype=np.float32)),
                                      torch.from_numpy(np.array(labels_gesture_personne_scrambled, dtype=np.int64)))
                validation = TensorDataset(
                    torch.from_numpy(np.array(examples_personne_scrambled_valid, dtype=np.float32)),
                    torch.from_numpy(np.array(labels_gesture_personne_scrambled_valid, dtype=np.int64)))

                trainLoader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
                validationLoader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)

                list_train_dataloader.append(trainLoader)
                list_val_dataloader.append(validationLoader)

                human_number += 1
                print("Shape training : ", np.shape(examples_personne_scrambled))
                print("Shape valid : ", np.shape(examples_personne_scrambled_valid))

        model = target_network_raw_emg_enhanced.SourceNetwork(number_of_class=18, dropout_rate=.35).cuda()

        criterion_class = nn.CrossEntropyLoss(reduction='sum')
        criterion_domain = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=0.002335721469090121)
        precision = 1e-8
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15,
                                                         verbose=True, eps=precision)

        best_weights = models_training.pre_train_model(model, criterion_class=criterion_class,
                                                       criterion_domain=criterion_domain, optimizer=optimizer,
                                                       scheduler=scheduler,
                                                       dataloaders={"train": list_train_dataloader,
                                                                    "val": list_val_dataloader}, precision=precision)

        # Save the best weights found to file
        torch.save(best_weights, "weights/best_pre_train_weights_target_raw_for_"+str(index_not_to_use) +
                   ".pt")

        print("Done Pre-Training for a participant")


def generate_source_and_target_dataloader(examples_training, labels_training, participant_to_get, number_of_cycles):
    for participant_index in range(len(labels_training)):  # Go over all the participants
        X_fine_tune_target, Y_fine_tune_target = [], []
        if participant_to_get == participant_index:
            last_label = labels_training[participant_index][0][0]
            nmbr_time_with_label = -1
            X_fine_tune_train, Y_fine_tune_train = [], []
            for j in range(
                    len(labels_training[
                            participant_index])):  # Collect training data up to the number of cycles allowed
                if last_label == labels_training[participant_index][j][0]:
                    nmbr_time_with_label += 1
                else:
                    nmbr_time_with_label = 0
                if nmbr_time_with_label < number_of_cycles:
                    for k in range(len(examples_training[participant_index][j])):
                        X_fine_tune_train.append(examples_training[participant_index][j][k])
                    Y_fine_tune_train.extend(labels_training[participant_index][j])

                last_label = labels_training[participant_index][j][0]

            X_fine_tune, Y_fine_tune = scramble(X_fine_tune_train, Y_fine_tune_train)
            valid_examples = X_fine_tune[0:int(len(X_fine_tune) * 0.1)]
            labels_valid = Y_fine_tune[0:int(len(Y_fine_tune) * 0.1)]

            X_fine_tune = X_fine_tune[int(len(X_fine_tune) * 0.1):]
            Y_fine_tune = Y_fine_tune[int(len(Y_fine_tune) * 0.1):]

            print(np.shape(X_fine_tune))
            train = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                                  torch.from_numpy(np.array(Y_fine_tune, dtype=np.int64)))

            validation = TensorDataset(torch.from_numpy(np.array(valid_examples, dtype=np.float32)),
                                       torch.from_numpy(np.array(labels_valid, dtype=np.int64)))

            trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
            validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)
        else:
            last_label = labels_training[participant_index][0][0]
            nmbr_time_with_label = -1
            for j in range(
                    len(labels_training[
                            participant_index])):  # Collect training data up to the number of cycles allowed
                if last_label == labels_training[participant_index][j][0]:
                    nmbr_time_with_label += 1
                else:
                    nmbr_time_with_label = 0
                if nmbr_time_with_label < 4:
                    for k in range(len(examples_training[participant_index][j])):
                        X_fine_tune_target.append(examples_training[participant_index][j][k])
                    Y_fine_tune_target.extend(np.ones(len(labels_training[participant_index][j])))

                last_label = labels_training[participant_index][j][0]

            print(np.shape(X_fine_tune_target))
            train_target = TensorDataset(torch.from_numpy(np.array(X_fine_tune_target, dtype=np.float32)),
                                  torch.from_numpy(np.array(Y_fine_tune_target, dtype=np.int64)))


    trainloader_target = torch.utils.data.DataLoader(train_target, batch_size=256, shuffle=True, drop_last=True)
    return trainloader, validationloader, trainloader_target


def calculate_fitness(examples_training, labels_training, examples_test, labels_test, number_of_cycles, learning_rate):
    accuracy_test0 = []
    for participant_index in range(len(labels_training)):  # Go over all the participants
        last_label = labels_training[participant_index][0][0]
        nmbr_time_with_label = -1
        X_fine_tune_train, Y_fine_tune_train = [], []
        for j in range(
                len(labels_training[participant_index])):  # Collect training data up to the number of cycles allowed
            if last_label == labels_training[participant_index][j][0]:
                nmbr_time_with_label += 1
            else:
                nmbr_time_with_label = 0
            if nmbr_time_with_label < number_of_cycles:
                for k in range(len(examples_training[participant_index][j])):
                    X_fine_tune_train.append(examples_training[participant_index][j][k])
                Y_fine_tune_train.extend(labels_training[participant_index][j])

            last_label = labels_training[participant_index][j][0]

        X_test_0, Y_test_0 = [], []
        for j in range(len(labels_test[participant_index])):
            for k in range(len(examples_test[participant_index][j])):
                X_test_0.append(examples_test[participant_index][j][k])
            Y_test_0.extend(labels_test[participant_index][j])

        X_fine_tune, Y_fine_tune = scramble(X_fine_tune_train, Y_fine_tune_train)
        valid_examples = X_fine_tune[0:int(len(X_fine_tune) * 0.1)]
        labels_valid = Y_fine_tune[0:int(len(Y_fine_tune) * 0.1)]

        X_fine_tune = X_fine_tune[int(len(X_fine_tune) * 0.1):]
        Y_fine_tune = Y_fine_tune[int(len(Y_fine_tune) * 0.1):]

        print(np.shape(X_fine_tune))
        train = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                              torch.from_numpy(np.array(Y_fine_tune, dtype=np.int64)))

        validation = TensorDataset(torch.from_numpy(np.array(valid_examples, dtype=np.float32)),
                                   torch.from_numpy(np.array(labels_valid, dtype=np.int64)))

        trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
        validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)

        pre_trained_weights = torch.load("weights/best_pre_train_weights_target_raw_for_" +
                                         str(participant_index) + ".pt")
        model = target_network_raw_emg_enhanced.TargetNetwork(number_of_class=18,
                                                            weights_pre_trained_convnet=pre_trained_weights,
                                                            dropout=.5).cuda()

        criterion = nn.CrossEntropyLoss(size_average=False)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.95, weight_decay=.001)

        precision = 1e-6
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)
        # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=np.inf)

        model = models_training.train_model(model, criterion, optimizer, scheduler, dataloaders={"train": trainloader,
                                                                             "val": validationloader},
                          precision=precision)

        model.eval()
        X_test_0, Y_test_0 = scramble(X_test_0, Y_test_0)

        test_0 = TensorDataset(torch.from_numpy(np.array(X_test_0, dtype=np.float32)),
                               torch.from_numpy(np.array(Y_test_0, dtype=np.int64)))

        test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=256, shuffle=False)

        total = 0
        correct_prediction_test_0 = 0
        for k, data_test_0 in enumerate(test_0_loader, 0):
            # get the inputs
            inputs_test_0, ground_truth_test_0 = data_test_0
            inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0.cuda()), Variable(ground_truth_test_0.cuda())

            outputs_test_0 = model(inputs_test_0)
            _, predicted = torch.max(outputs_test_0.data, 1)
            correct_prediction_test_0 += (predicted.cpu().numpy() == ground_truth_test_0.data.cpu().numpy()).sum()
            total += ground_truth_test_0.size(0)
        print("ACCURACY TEST_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))
        accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))

    return accuracy_test0


if __name__ == '__main__':
    # Comment between here
    '''
    examples, labels = load_pre_training_dataset.read_data('PreTrainingDataset')
    datasets = [examples, labels]

    np.save("Processed_datasets/saved_pre_training_dataset_spectrogram.npy", datasets)
    '''
    # And here if the pre-training dataset was already processed and saved

    # Comment between here
    '''
    datasets_pre_training = np.load("Processed_datasets/saved_pre_training_dataset_spectrogram.npy", encoding="bytes")
    examples_pre_training, labels_pre_training = datasets_pre_training
    calculate_pre_training(examples_pre_training, labels_pre_training)
    '''
    # Comment between here

    # Comment between here
    '''
    train_examples, train_labels, test_examples, test_labels = load_NinaPro_dataset.get_data(
        "../Dataset/datasetNinaPro/", "raw")

    datasets = [train_examples, train_labels, test_examples, test_labels]
    np.save("Processed_datasets/saved_dataset_NinaPro_RAW_test.npy", datasets)
    '''
    # And here if the pre-training dataset was already processed and saved

    # Comment between here

    datasets = np.load("Processed_datasets/saved_dataset_NinaPro_RAW_test.npy")
    train_examples_segregated_by_domain, train_labels_segregated_by_domain, _, _ = datasets
    source_convNet = target_network_raw_emg_enhanced.SourceNetwork(number_of_class=18, dropout_rate=.35).cuda()
    run_pre_training_for_NinaPro_cycle(train_examples_segregated_by_domain, train_labels_segregated_by_domain)

    # And here if the pre-training of the network was already completed.
    datasets = np.load("Processed_datasets/saved_dataset_NinaPro_RAW_test.npy")
    train_examples_segregated_by_domain, train_labels_segregated_by_domain, test_examples_by_domain, \
    test_labels_by_domain = datasets
    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []
    # learning_rate=1.1288378916846883e-05 (for network as described)
    learning_rate = 0.002335721469090121  # For enhanced network
    for number_cycle in range(4, 5):
        test_0 = []
        for i in range(20):
            accuracy_subject = calculate_fitness(train_examples_segregated_by_domain, train_labels_segregated_by_domain,
                                                 test_examples_by_domain, test_labels_by_domain, number_cycle,
                                                 learning_rate)
            print(accuracy_subject)
            test_0.append(accuracy_subject)
            print("TEST 0 SO FAR: ", test_0)
            print("CURRENT AVERAGE : ", np.mean(test_0))

        print("ACCURACY FINAL TEST 0: ", test_0)
        print("ACCURACY FINAL TEST 0: ", np.mean(test_0))

        with open("results/results_RAW_TARGET_NINAPRO_Adversarial.txt", "a") as myfile:
            myfile.write("ConvNet RAW : " + str(number_cycle) + "\n\n")
            myfile.write("Test: \n")
            myfile.write(str(test_0) + '\n')
            myfile.write(str(np.mean(test_0, axis=0)) + '\n')
            myfile.write(str(np.mean(test_0, axis=1)) + '\n')
            myfile.write(str(np.mean(test_0)) + '\n')

# TODO Use DANN during training (on top of pre-training)