import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from LongTermClassificationMain.TrainingsAndEvaluations.ForTrainingSessions.train_lda_standard_training import \
    get_dataset


def prepare_dataset_evaluation_session(examples_datasets_evaluation, labels_datasets_evaluation):
    participants_datasets_evaluation_formatted, participants_dataloaders_validation = [], []
    print(np.shape(examples_datasets_evaluation))
    for participant_examples, participant_labels in zip(examples_datasets_evaluation, labels_datasets_evaluation):
        print("Participant")
        print(np.shape(participant_examples))
        print(np.shape(participant_labels))
        dataloaders_participant_evaluation, dataloaders_participant_validation = [], []
        for sessions_examples, sessions_labels in zip(participant_examples, participant_labels):
            print("Session")
            print(np.shape(sessions_labels))
            print(np.shape(sessions_examples))

            X = np.array(sessions_examples, dtype=np.float32)
            Y = np.array(sessions_labels, dtype=np.int64)


def evaluate_on_evaluation_session(participants_train, participants_evaluation_examples,
                                   participants_evaluation_labels):
    average_accuracy = []
    predictions_all = []
    ground_truths_all = []
    for participant_i in range(len(participants_train)):
        predictions_all_sessions = []
        groundtruth_all_session = []
        accuracy_sessions = []
        # Train using the first session
        print(np.shape(participants_evaluation_examples[participant_i]))
        for session_j in range(0, 6):
            # For comparison purposes, only consider the second session of the day for each participant and only the
            # first three of these sessions
            if session_j % 2 != 0:
                lda = LinearDiscriminantAnalysis()
                lda.fit(np.nan_to_num(participants_train[participant_i]["Examples"][int(session_j/2)]),
                        participants_train[participant_i]["Labels"][int(session_j/2)])

                predictions = lda.predict(np.nan_to_num(participants_evaluation_examples[participant_i][session_j]))
                predictions_all_sessions.extend(predictions)
                groundtruth_all_session.extend(participants_evaluation_labels[participant_i][session_j])
                accuracy_sessions.append(
                    accuracy_score(participants_evaluation_labels[participant_i][session_j], predictions))
        average_accuracy.append(accuracy_sessions)
    print(np.mean(average_accuracy, axis=0))
    print(np.mean(average_accuracy))
    print(np.std(average_accuracy))


if __name__ == '__main__':
    dataset_techniques_names = ["TD", "TSD"]
    dataset_techniques = ["TD_features_set", "TSD_features_set"]
    for name, dataset in zip(dataset_techniques_names, dataset_techniques):
        print(os.listdir("../../"))
        with open("../../Processed_datasets/%s_training_session.pickle" % dataset, 'rb') as f:
            dataset_training = pickle.load(file=f)
        examples_datasets_train = dataset_training['examples_training']
        labels_datasets_train = dataset_training['labels_training']

        with open("../../Processed_datasets/%s_evaluation_session.pickle" % dataset, 'rb') as f:
            dataset_evaluation = pickle.load(file=f)
        examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
        labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
        evaluation_datetimes = dataset_evaluation['evaluation_datetimes']

        participants_train, _ = get_dataset(examples_datasets_train, labels_datasets_train, number_of_cycles_training=4,
                                            ignore_first=True)

        evaluate_on_evaluation_session(participants_train, examples_datasets_evaluation, labels_datasets_evaluation)
