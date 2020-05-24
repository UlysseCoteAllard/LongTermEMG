import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_in_dataloader import load_dataloaders_training_sessions


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


def get_dataset(examples_datasets, labels_datasets, number_of_cycles_training, ignore_first=True,
                number_of_cycles_total=4, gestures_to_remove=None):
    participants_dataset_train, participants_dataset_test = [], []
    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        dataset_training_X, dataset_training_Y = [], []
        dataset_testing_X, dataset_testing_Y = [], []
        session = 0
        for training_index_examples, training_index_labels in zip(participant_examples, participant_labels):
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
                    if cycle < number_of_cycles_training:
                        X_associated_with_training_i.extend(examples_cycles)
                        Y_associated_with_training_i.extend(labels_cycles)
                    else:
                        X_test_associated_with_training_i.extend(examples_cycles)
                        Y_test_associated_with_training_i.extend(labels_cycles)
            dataset_training_X.append(X_associated_with_training_i)
            dataset_training_Y.append(Y_associated_with_training_i)

            dataset_testing_X.append(X_test_associated_with_training_i)
            dataset_testing_Y.append(Y_test_associated_with_training_i)
            session += 1
            if session >= 3:
                break
        participants_dataset_train.append({"Examples": dataset_training_X, "Labels": dataset_training_Y})
        participants_dataset_test.append({"Examples": dataset_testing_X, "Labels": dataset_testing_Y})
    return participants_dataset_train, participants_dataset_test


def train_test_technique(dataset_train, dataset_test, name_technique, number_of_class=7):
    average_accuracy = []
    predictions_all = []
    ground_truths_all = []
    for participant_i in range(len(dataset_train)):
        predictions_all_sessions = []
        groundtruth_all_session = []
        for session_j in range(len(dataset_train[participant_i]["Examples"])):
            lda = LinearDiscriminantAnalysis()
            lda.fit(np.nan_to_num(dataset_train[participant_i]["Examples"][session_j]),
                    dataset_train[participant_i]["Labels"][session_j])
            predictions = lda.predict(np.nan_to_num(dataset_test[participant_i]["Examples"][session_j]))
            predictions_all_sessions.extend(predictions)
            groundtruth_all_session.extend(dataset_test[participant_i]["Labels"][session_j])
            average_accuracy.append(accuracy_score(dataset_test[participant_i]["Labels"][session_j], predictions))
        predictions_all.append(predictions_all_sessions)
        ground_truths_all.append(groundtruth_all_session)

    np.save("../../results/results_lda/" + name_technique + "_%d_gestures" % number_of_class,
            (ground_truths_all,
             predictions_all,
             average_accuracy))

    print(np.array2string(100*np.array(average_accuracy), max_line_width=1))
    return average_accuracy


if __name__ == "__main__":
    dataset_techniques_names = ["TD", "NinaPro", "SampEn Pipeline", "LSF9", "TDPSD", "FSD"]
    dataset_techniques = ["TD_features_set_training_session", "NinaPro_features_set_training_session",
                          "Sampen_features_set_training_session", "LSF9_features_set_training_session",
                          "TDPSD_features_set_training_session", "FSD_features_set_training_session"]

    for name, dataset in zip(dataset_techniques_names, dataset_techniques):
        print(os.listdir("../../"))
        with open("../../Processed_datasets/%s.pickle" % dataset, 'rb') as f:
            dataset_training = pickle.load(file=f)

        examples_datasets_train = dataset_training['examples_training']
        labels_datasets_train = dataset_training['labels_training']


        participants_train, participants_test = get_dataset(
            examples_datasets_train, labels_datasets_train,
            number_of_cycles_training=3, ignore_first=True, gestures_to_remove=(5, 6, 9, 10))

        accuracies = train_test_technique(participants_train, participants_test, name, number_of_class=7)

        print("%s : 7 GESTURES: Average accuracy score: %f, STD: %f" % (name, np.mean(accuracies), np.std(accuracies)))

        participants_train, participants_test = get_dataset(
            examples_datasets_train, labels_datasets_train,
            number_of_cycles_training=3, ignore_first=True)

        accuracies = train_test_technique(participants_train, participants_test, name, number_of_class=11)

        print("%s : 11 GESTURES: Average accuracy score: %f, STD: %f" % (name, np.mean(accuracies), np.std(accuracies)))
