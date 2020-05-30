import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset


def get_dataloader(examples_datasets, labels_datasets, number_of_cycle_for_first_training=2,
                   number_of_cycles_rest_of_training=2, batch_size=128,
                   drop_last=True, shuffle=True,
                   number_of_cycles_total=4, validation_set_ratio=0.1, get_validation_set=True, cycle_for_test=None,
                   ignore_first=False):
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
                if (ignore_first is True and cycle != 0) or ignore_first is False:
                    if cycle < cycles_to_add_to_train:
                        X_associated_with_training_i.extend(training_index_examples[cycle])
                        Y_associated_with_training_i.extend(training_index_labels[cycle])
                    elif cycle_for_test is None:
                        X_test_associated_with_training_i.extend(training_index_examples[cycle])
                        Y_test_associated_with_training_i.extend(training_index_labels[cycle])
                    elif cycle == cycle_for_test:
                        X_test_associated_with_training_i.extend(training_index_examples[cycle])
                        Y_test_associated_with_training_i.extend(training_index_labels[cycle])

            k += 1

            if get_validation_set:
                # Shuffle X and Y and separate them in a train and validation set.
                X, X_valid, Y, Y_valid = train_test_split(X_associated_with_training_i, Y_associated_with_training_i,
                                                          test_size=validation_set_ratio, shuffle=True)
            else:
                X, Y = X_associated_with_training_i, Y_associated_with_training_i

            X = np.expand_dims(X, axis=1)
            print("SHAPE X: ", np.shape(X))
            train = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                                  torch.from_numpy(np.array(Y, dtype=np.int64)))
            trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle,
                                                      drop_last=drop_last)
            dataloaders_trainings.append(trainloader)

            if get_validation_set:
                X_valid = np.expand_dims(X_valid, axis=1)
                validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                           torch.from_numpy(np.array(Y_valid, dtype=np.int64)))
                validationloader = torch.utils.data.DataLoader(validation, batch_size=len(X_valid), shuffle=shuffle,
                                                               drop_last=False)
                dataloaders_validations.append(validationloader)

            if len(X_test_associated_with_training_i) > 0:
                X_test = np.expand_dims(X_test_associated_with_training_i, axis=1)
                test = TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
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
                                       cycle_for_test=None,
                                       ignore_first=False):
    train, validation, test = get_dataloader(examples_datasets_train, labels_datasets_train,
                                             number_of_cycle_for_first_training=
                                             number_of_cycle_for_first_training,
                                             number_of_cycles_rest_of_training=
                                             number_of_cycles_rest_of_training, batch_size=batch_size,
                                             drop_last=drop_last, shuffle=shuffle,
                                             get_validation_set=get_validation_set,
                                             cycle_for_test=cycle_for_test,
                                             ignore_first=ignore_first)

    return train, validation, test


def get_dataloader_evaluation(examples_datasets, labels_datasets, batch_size=128,
                              drop_last=False, shuffle=False, get_validation=False, validation_ratio=0.1):
    participants_datasets_evaluation_formatted, participants_dataloaders_validation = [], []
    print(np.shape(examples_datasets))
    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        print("Participant")
        print(np.shape(participant_examples))
        print(np.shape(participant_labels))
        dataloaders_participant_evaluation, dataloaders_participant_validation = [], []
        for sessions_examples, sessions_labels in zip(participant_examples, participant_labels):
            print("Session")
            print(np.shape(sessions_labels))
            print(np.shape(sessions_examples))

            #X = np.expand_dims(np.array(sessions_examples, dtype=np.float32), axis=1)
            X = np.array(sessions_examples, dtype=np.float32)
            Y = np.array(sessions_labels, dtype=np.int64)
            if get_validation:
                # Shuffle X and Y and separate them in a train and validation set.
                X, X_valid, Y, Y_valid = train_test_split(X, Y, test_size=validation_ratio, shuffle=True)

                validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                           torch.from_numpy(np.array(Y_valid, dtype=np.int64)))
                validationloader = torch.utils.data.DataLoader(validation, batch_size=len(X_valid), shuffle=shuffle,
                                                               drop_last=drop_last)
                dataloaders_participant_validation.append(validationloader)

            session_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
            session_dataloader = torch.utils.data.DataLoader(session_dataset, batch_size=batch_size, shuffle=shuffle,
                                                             drop_last=drop_last)
            dataloaders_participant_evaluation.append(session_dataloader)
        participants_datasets_evaluation_formatted.append(dataloaders_participant_evaluation)
        participants_dataloaders_validation.append(dataloaders_participant_validation)

    if get_validation:
        return participants_datasets_evaluation_formatted, participants_dataloaders_validation
    else:
        return participants_datasets_evaluation_formatted


def load_dataloaders_test_sessions(examples_datasets_evaluation, labels_datasets_evaluation, batch_size=128,
                                   drop_last=False, shuffle=False,  get_validation=False,
                                   validation_ratio=0.1):
    dataloaders_TCN_validation = None
    if get_validation:
        dataloaders_TCN, dataloaders_TCN_validation = get_dataloader_evaluation(examples_datasets_evaluation,
                                                                                labels_datasets_evaluation,
                                                                                batch_size=batch_size,
                                                                                drop_last=drop_last, shuffle=shuffle,
                                                                                get_validation=get_validation,
                                                                                validation_ratio=validation_ratio)
    else:
        dataloaders_TCN = get_dataloader_evaluation(examples_datasets_evaluation, labels_datasets_evaluation,
                                                    batch_size=batch_size, drop_last=drop_last,
                                                    shuffle=shuffle)

    if get_validation:
        return dataloaders_TCN, dataloaders_TCN_validation
    else:
        return dataloaders_TCN
