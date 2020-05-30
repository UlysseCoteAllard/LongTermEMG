import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset


def remove_list_gestures_from_cycle(cycle_examples, cycle_labels, gestures_to_remove=(5, 6, 9, 10)):
    list_gestures = list(set(cycle_labels))
    dict_reduced_labels = {}
    new_label = 0
    for i in list_gestures:
        if i not in gestures_to_remove:
            dict_reduced_labels[i] = new_label
            new_label += 1

    reduced_examples_cycles, reduced_labels_cycles = [], []
    for example, label in zip(cycle_examples, cycle_labels):
        if label not in gestures_to_remove:
            reduced_examples_cycles.append(example)
            reduced_labels_cycles.append(dict_reduced_labels[label])

    return reduced_examples_cycles, reduced_labels_cycles


def get_dataloader(examples_datasets, labels_datasets, number_of_cycle_for_first_training=2,
                   number_of_cycles_rest_of_training=2, batch_size=128,
                   drop_last=True, shuffle=True,
                   number_of_cycles_total=4, validation_set_ratio=0.1, get_validation_set=True, cycle_for_test=None,
                   ignore_first=False, gestures_to_remove=None):
    participants_dataloaders, participants_dataloaders_validation, participants_dataloaders_test = [], [], []

    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        dataloaders_trainings = []
        dataloaders_validations = []
        dataloaders_testing = []
        k = 0
        for training_index_examples, training_index_labels in zip(participant_examples, participant_labels):
            print(np.shape(training_index_labels), " ", k)
            cycles_to_add_to_train = number_of_cycle_for_first_training
            if k > 0:
                cycles_to_add_to_train = number_of_cycles_rest_of_training
            X_associated_with_training_i, Y_associated_with_training_i = [], []
            X_test_associated_with_training_i, Y_test_associated_with_training_i = [], []
            for cycle in range(number_of_cycles_total):
                if (ignore_first is True and cycle != 1) or ignore_first is False:
                    examples_cycles = training_index_examples[cycle]
                    labels_cycles = training_index_labels[cycle]
                    if gestures_to_remove is not None:
                        examples_cycles, labels_cycles = remove_list_gestures_from_cycle(examples_cycles, labels_cycles,
                                                                                         gestures_to_remove=
                                                                                         gestures_to_remove)
                    if cycle < cycles_to_add_to_train:
                        X_associated_with_training_i.extend(examples_cycles)
                        Y_associated_with_training_i.extend(labels_cycles)
                    elif cycle_for_test is None:
                        X_test_associated_with_training_i.extend(examples_cycles)
                        Y_test_associated_with_training_i.extend(labels_cycles)
                    elif cycle == cycle_for_test:
                        X_test_associated_with_training_i.extend(examples_cycles)
                        Y_test_associated_with_training_i.extend(labels_cycles)

            k += 1

            if get_validation_set:
                # Shuffle X and Y and separate them in a train and validation set.
                X, X_valid, Y, Y_valid = train_test_split(X_associated_with_training_i, Y_associated_with_training_i,
                                                          test_size=validation_set_ratio, shuffle=True)
            else:
                X, Y = X_associated_with_training_i, Y_associated_with_training_i

            print("SHAPE X: ", np.shape(X))
            train = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                                  torch.from_numpy(np.array(Y, dtype=np.int64)))
            trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle,
                                                      drop_last=drop_last)

            dataloaders_trainings.append(trainloader)

            if get_validation_set:
                validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                           torch.from_numpy(np.array(Y_valid, dtype=np.int64)))
                validationloader = torch.utils.data.DataLoader(validation, batch_size=len(X_valid), shuffle=shuffle,
                                                               drop_last=False)
                dataloaders_validations.append(validationloader)

            if len(X_test_associated_with_training_i) > 0:
                test = TensorDataset(torch.from_numpy(np.array(X_test_associated_with_training_i, dtype=np.float32)),
                                     torch.from_numpy(np.array(Y_test_associated_with_training_i, dtype=np.int64)))
                testLoader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                                         drop_last=False)
                dataloaders_testing.append(testLoader)

        participants_dataloaders.append(dataloaders_trainings)
        if get_validation_set:
            participants_dataloaders_validation.append(dataloaders_validations)
        participants_dataloaders_test.append(dataloaders_testing)

    return participants_dataloaders, participants_dataloaders_validation, participants_dataloaders_test


def load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                       number_of_cycle_for_first_training=2, number_of_cycles_rest_of_training=2,
                                       batch_size=128, drop_last=True, shuffle=True, get_validation_set=True,
                                       cycle_for_test=None, ignore_first=False, gestures_to_remove=None):
    train, validation, test = get_dataloader(examples_datasets_train, labels_datasets_train,
                                             number_of_cycle_for_first_training=
                                             number_of_cycle_for_first_training,
                                             number_of_cycles_rest_of_training=
                                             number_of_cycles_rest_of_training, batch_size=batch_size,
                                             drop_last=drop_last, shuffle=shuffle,
                                             get_validation_set=get_validation_set,
                                             cycle_for_test=cycle_for_test,
                                             ignore_first=ignore_first,
                                             gestures_to_remove=gestures_to_remove)

    return train, validation, test
