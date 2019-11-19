import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset

def get_dataloader(examples_datasets, labels_datasets, number_of_cycle_for_first_training=2,
                   number_of_cycles_rest_of_training=2, batch_size=128,
                   drop_last=True, shuffle=True, get_convNet_dataloader=False, sub_examples_per_examples=5,
                   number_of_cycles_total=4, validation_set_ratio=0.1, get_validation_set=True):
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
                if cycle < cycles_to_add_to_train:
                    X_associated_with_training_i.extend(training_index_examples[cycle])
                    Y_associated_with_training_i.extend(training_index_labels[cycle])
                else:
                    X_test_associated_with_training_i.extend(training_index_examples[cycle])
                    Y_test_associated_with_training_i.extend(training_index_labels[cycle])
            k += 1

            if get_validation_set:
                # Shuffle X and Y and separate them in a train and validation set.
                X, X_valid, Y, Y_valid = train_test_split(X_associated_with_training_i, Y_associated_with_training_i,
                                                          test_size=validation_set_ratio, shuffle=True)
            else:
                X, Y = X_associated_with_training_i, Y_associated_with_training_i

            if get_convNet_dataloader:
                # Separate the examples into sub-examples that will be use to train the ConvNet
                X = np.stack(np.split(np.array(X), sub_examples_per_examples, axis=2), axis=1)
                if get_validation_set:
                    X_valid = np.stack(np.split(np.array(X_valid), sub_examples_per_examples, axis=2), axis=1)
                #Y = np.tile(np.array(Y), sub_examples_per_examples)
                #Y_valid = np.tile(np.array(Y_valid), sub_examples_per_examples)
                if len(X_test_associated_with_training_i) > 0:
                    X_test = np.hstack(np.split(np.array(X_test_associated_with_training_i),
                                                sub_examples_per_examples, axis=2))
                    #Y_test = np.tile(np.array(Y_test_associated_with_training_i),
                    #                 sub_examples_per_examples)
                    Y_test = np.array(Y_test_associated_with_training_i)

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
                                                               drop_last=drop_last)
                dataloaders_validations.append(validationloader)

            if len(X_test_associated_with_training_i) > 0:
                if get_convNet_dataloader is False:
                    X_test = X_test_associated_with_training_i
                    Y_test = Y_test_associated_with_training_i
                X_test = np.expand_dims(X_test, axis=1)
                test = TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
                                     torch.from_numpy(np.array(Y_test, dtype=np.int64)))
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
                                       batch_size=128, drop_last=True, shuffle=True,
                                       sub_examples_per_examples=3, get_validation_set=True):

    '''
    train_convNet, validation_convNet, test_convNet = get_dataloader(examples_datasets_train, labels_datasets_train,
                                                                     number_of_cycle_for_first_training=
                                                                     number_of_cycle_for_first_training,
                                                                     number_of_cycles_rest_of_training=
                                                                     number_of_cycles_rest_of_training,
                                                                     batch_size=batch_size, drop_last=drop_last,
                                                                     shuffle=shuffle, get_convNet_dataloader=True,
                                                                     sub_examples_per_examples=
                                                                     sub_examples_per_examples)
    '''

    train_lstm, validation_lstm, test_lstm = get_dataloader(examples_datasets_train, labels_datasets_train,
                                                            number_of_cycle_for_first_training=
                                                            number_of_cycle_for_first_training,
                                                            number_of_cycles_rest_of_training=
                                                            number_of_cycles_rest_of_training, batch_size=batch_size,
                                                            drop_last=drop_last, shuffle=shuffle,
                                                            get_convNet_dataloader=False,
                                                            get_validation_set=get_validation_set)

    #return train_convNet, validation_convNet, test_convNet, train_lstm, validation_lstm, test_lstm
    return train_lstm, validation_lstm, test_lstm


def get_dataloader_evaluation(examples_datasets, labels_datasets, batch_size=128,
                              drop_last=False, shuffle=False, sub_examples_per_examples=5,
                              get_convNet_dataloader=False):
    participants_datasets_evaluation_formatted = []
    print(np.shape(examples_datasets))
    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        print("Participant")
        print(np.shape(participant_examples))
        print(np.shape(participant_labels))
        dataloaders_participant_evaluation = []
        for sessions_examples, sessions_labels in zip(participant_examples, participant_labels):
            print("Session")
            print(np.shape(sessions_labels))
            print(np.shape(sessions_examples))
            if get_convNet_dataloader:
                # Separate the examples into sub-examples that will be use to train the ConvNet
                sessions_examples = np.vstack(np.split(np.array(sessions_examples), sub_examples_per_examples, axis=2))
                sessions_labels = np.tile(np.array(sessions_labels), sub_examples_per_examples)

            X = np.expand_dims(np.array(sessions_examples, dtype=np.float32), axis=1)
            Y = np.array(sessions_labels, dtype=np.int64)
            session_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
            session_dataloader = torch.utils.data.DataLoader(session_dataset, batch_size=batch_size, shuffle=shuffle,
                                                             drop_last=drop_last)
            dataloaders_participant_evaluation.append(session_dataloader)
        participants_datasets_evaluation_formatted.append(dataloaders_participant_evaluation)

    return participants_datasets_evaluation_formatted


def load_dataloaders_test_sessions(examples_datasets_evaluation, labels_datasets_evaluation, batch_size=128,
                                   drop_last=False, shuffle=False, sub_examples_per_examples=3):
    '''
    test_convNet = get_dataloader_evaluation(examples_datasets_evaluation, labels_datasets_evaluation,
                                             batch_size=batch_size, drop_last=drop_last,  shuffle=shuffle,
                                             get_convNet_dataloader=True,
                                             sub_examples_per_examples=sub_examples_per_examples)
    '''
    test_lstm = get_dataloader_evaluation(examples_datasets_evaluation, labels_datasets_evaluation,
                                          batch_size=batch_size, drop_last=drop_last,  shuffle=shuffle,
                                          get_convNet_dataloader=False,
                                          sub_examples_per_examples=sub_examples_per_examples)

    #return test_convNet, test_lstm
    return test_lstm


if __name__ == "__main__":
    with open("../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)
    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    load_dataloaders_test_sessions(examples_datasets_evaluation=examples_datasets_evaluation,
                                   labels_datasets_evaluation=labels_datasets_evaluation)

    with open("../Processed_datasets/LongTermDataset_training_session.pickle", 'rb') as f:
       dataset_training = pickle.load(file=f)
    examples_datasets_train = dataset_training['examples_training']
    labels_datasets_train = dataset_training['labels_training']
    _, _, _, _, _, _ = load_dataloaders_training_sessions(examples_datasets_train=examples_datasets_train,
                                                          labels_datasets_train=labels_datasets_train)

