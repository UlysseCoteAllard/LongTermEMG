import os
import pickle
import numpy as np
from datetime import datetime, timedelta

from LongTermClassificationMain.PrepareAndLoadDataLongTerm.prepare_dataset_utils import butter_bandpass_filter, \
    show_filtered_signal, load_timestamps_from_participant, get_angles_from_positions_3d_arm
from LongTermClassificationMain.PrepareAndLoadDataLongTerm import feature_extraction

list_participant_training_1_to_skip = ["Participant0/Training1", "Participant0/Evaluation2", "Participant0/Evaluation3",
                                       "Participant2/Training1", "Participant2/Evaluation2", "Participant2/Evaluation3"]


def get_highest_average_emg_window(emg_signal, window_for_moving_average):
    max_average = 0.
    example = []
    for emg_vector in emg_signal:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))

        if len(example) >= window_for_moving_average:
            example = example.transpose()
            example_filtered = []
            for channel in example:
                channel_filtered = butter_bandpass_filter(channel, lowcut=20, highcut=495, fs=1000, order=4)
                # show_filtered_signal(channel, channel_filtered)
                example_filtered.append(channel_filtered)
            average = np.mean(np.abs(example_filtered))
            if average > max_average:
                max_average = average
            example = example.transpose()
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[1:]
    return max_average


def format_examples(emg_examples, feature_set_function, window_size=150, size_non_overlap=50):
    examples_to_calculate_features_set_from = []
    example = []
    for emg_vector in emg_examples:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))

        if len(example) >= window_size:
            # The example is of the shape TIME x CHANNEL. Make it of the shape CHANNEL x TIME
            example = example.transpose()
            # Go over each channel and bandpass filter it between 20 and 495 Hz.
            example_filtered = []
            for channel in example:
                channel_filtered = butter_bandpass_filter(channel, lowcut=20, highcut=495, fs=1000, order=4)
                # show_filtered_signal(channel, channel_filtered)
                example_filtered.append(channel_filtered)
            # Add the filtered example to the list of examples to return and transpose the example array again to go
            # back to TIME x CHANNEL
            examples_to_calculate_features_set_from.append(example_filtered)
            example = example.transpose()
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[size_non_overlap:]
    examples_features_set_calculated = feature_extraction.get_dataset_with_features_set(
        dataset=examples_to_calculate_features_set_from, features_set_function=feature_set_function)
    return examples_features_set_calculated


def format_examples_and_timestamps(emg_examples, timestamps, features_set_function, window_size=150,
                                   size_non_overlap=50):
    examples_formatted_timestamps = []
    dataset_to_calculate_features_set_from = []
    example, example_timestamp = [], []
    for emg_vector, timestamp in zip(emg_examples, timestamps):
        if len(example) == 0:
            example = emg_vector
            example_timestamp = timestamp
        else:
            example = np.row_stack((example, emg_vector))
            example_timestamp = np.row_stack((example_timestamp, timestamp))

        if len(example) >= window_size:
            # The example is of the shape TIME x CHANNEL. Make it of the shape CHANNEL x TIME
            example = example.transpose()
            # Go over each channel and bandpass filter it between 20 and 495 Hz.
            example_filtered = []
            for channel in example:
                channel_filtered = butter_bandpass_filter(channel, lowcut=20, highcut=495, fs=1000, order=4)
                # show_filtered_signal(channel, channel_filtered)
                example_filtered.append(channel_filtered)
            # print("SHAPE SPECTROGRAM: ", np.shape(spectrogram))
            # Add the filtered example to the list of examples to return and transpose the example array again to go
            # back to TIME x CHANNEL
            dataset_to_calculate_features_set_from.append(example_filtered)
            example = example.transpose()

            # Add the timestamp of this example as the timestamp of the middle of the current timestamp example
            examples_formatted_timestamps.extend(example_timestamp[int(len(example_timestamp) / 2)])

            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[size_non_overlap:]
            example_timestamp = example_timestamp[size_non_overlap:]
    datasets_with_features_set = feature_extraction.get_dataset_with_features_set(
        dataset=dataset_to_calculate_features_set_from, features_set_function=features_set_function)
    return datasets_with_features_set, examples_formatted_timestamps


def read_files_to_format_evaluation_session(path_folder_examples, window_size, size_non_overlap):
    examples_evaluation, labels_evaluation, examples_timestamps = [], [], []

    # Check if the folder is empty, if so, skip it
    # path_folder_examples = path + "/" + folder_participant + "/" + training_directory + "/EMG"
    if len(os.listdir(path_folder_examples)) == 0:
        return [], [], []
    print(os.listdir(path_folder_examples))
    with open(path_folder_examples + "/3dc_EMG_gesture.txt") as emgFile, \
            open(path_folder_examples + "/3dc_EMG_Timestamp_gesture.txt") as timestampFile:
        examples_to_format, timestamps = [], []
        for line_emg in emgFile:
            #  strip() remove the "\n" character, split separate the data in a list. np.float
            #  transform each element of the list from a str to a float
            emg_signal = np.float32(line_emg.strip().split(","))
            examples_to_format.append(emg_signal)
        for line_timestamp in timestampFile:
            # [:len(line_timestamp) - 1], the -1 is to remove the '\n' character
            date_and_time_emg_signal = datetime.fromisoformat(line_timestamp.strip())
            timestamps.append(date_and_time_emg_signal)
        # In the unity software, the EMG class was activated before the EMG armband was actually recording, making it
        # so that timestamps where registered with no actual emg link (this happen at the beginning). Remove these
        # timestamps at the beginning so that both array matches
        timestamps = timestamps[len(timestamps) - len(examples_to_format):]
        examples, timestamps_from_examples = format_examples_and_timestamps(examples_to_format,
                                                                            features_set_function=feature_set_function,
                                                                            timestamps=timestamps,
                                                                            window_size=window_size,
                                                                            size_non_overlap=size_non_overlap)

    # Get the arm's angles
    with open(path_folder_examples + "/arm_positions.pickle", 'rb') as f:
        arm_positions = pickle.load(file=f)
        angles_arm = get_angles_from_positions_3d_arm(arm_positions)
    # Sync the arm angles with the examples
    first_timestamp = int(angles_arm[0]['timestamp'])
    datetime_arm_original = timestamps_from_examples[0]
    angles_and_timestamps_synchronized = []

    previous_time_difference = datetime_arm_original - datetime_arm_original
    previous_time_arm_index = 0
    previous_time_arm = datetime_arm_original + timedelta(microseconds=(
            int(angles_arm[0]['timestamp']) - first_timestamp))

    index_arm_angles = 1
    biggest_time_difference = timedelta(0)
    for index_timestamp_example in range(len(timestamps_from_examples)):
        while True:
            current_time_arm = datetime_arm_original + timedelta(
                microseconds=(int(angles_arm[index_arm_angles]['timestamp']) - first_timestamp))
            if current_time_arm > timestamps_from_examples[index_timestamp_example]:
                current_time_difference = current_time_arm - timestamps_from_examples[index_timestamp_example]
            else:
                current_time_difference = timestamps_from_examples[index_timestamp_example] - current_time_arm
            if current_time_arm >= timestamps_from_examples[index_timestamp_example]:
                if previous_time_difference < current_time_difference:
                    index_arm_angles = previous_time_arm_index
                    current_time_arm = previous_time_arm
                break
            else:
                previous_time_arm = current_time_arm
                previous_time_difference = current_time_difference
                previous_time_arm_index = index_arm_angles
                if index_arm_angles + 1 < len(angles_arm):
                    index_arm_angles += 1
                else:
                    break
        angles_associated = {"pitch": angles_arm[index_arm_angles]['pitch'], "yaw": angles_arm[index_arm_angles]['yaw'],
                             "timestamp": current_time_arm}

        if current_time_arm > timestamps_from_examples[index_timestamp_example]:
            time_difference = current_time_arm - timestamps_from_examples[index_timestamp_example]
        else:
            time_difference = timestamps_from_examples[index_timestamp_example] - current_time_arm
        if biggest_time_difference < time_difference:
            biggest_time_difference = time_difference

        angles_and_timestamps_synchronized.append(angles_associated)

    print("LEN ARM ANGLES : ", np.shape(angles_and_timestamps_synchronized), " BIGGEST TIME DIFFERENCE: ",
          biggest_time_difference)

    timestamps_for_gestures, gestures_array = [], []
    with open(path_folder_examples + "/timestamps.txt") as timestamps_gestures, \
            open(path_folder_examples + "/gesture_asked.txt") as gestures:
        for gesture, timestamp_gesture in zip(gestures, timestamps_gestures):
            gestures_array.append(np.float32(gesture.strip()))
            timestamps_for_gestures.append(datetime.fromisoformat(timestamp_gesture.strip()))

    "Build the labels according to the timestamps"
    current_gesture_index = -1
    for example, timestamp in zip(examples, timestamps_from_examples):
        if current_gesture_index + 1 < len(timestamps_for_gestures) and \
                timestamps_for_gestures[current_gesture_index + 1] < timestamp:
            current_gesture_index += 1
        if current_gesture_index > -1:
            examples_evaluation.append(example)
            labels_evaluation.append(int(gestures_array[current_gesture_index]))
            examples_timestamps.append(timestamp)
    return examples_evaluation, labels_evaluation, examples_timestamps, angles_and_timestamps_synchronized


def read_files_to_format_training_session(path_folder_examples, feature_set_function,
                                          number_of_cycles, number_of_gestures, window_size,
                                          size_non_overlap):
    examples_training, labels_training = [], []
    # Check if the folder is empty, if so, skip it
    # path_folder_examples = path + "/" + folder_participant + "/" + training_directory + "/EMG"
    if len(os.listdir(path_folder_examples)) == 0:
        return [], [], []
    print(os.listdir(path_folder_examples))
    highest_activation_per_gesture = []
    for cycle in range(number_of_cycles):
        # path_folder_examples = path + folder_participant + "/" + training_directory + "/EMG"
        # This one instance, the participant only recorded one cycle of training. Skip it
        for participant_session_to_skip in list_participant_training_1_to_skip:
            if participant_session_to_skip in path_folder_examples:
                return [], [], []
        path_emg = path_folder_examples + "/3dc_EMG_gesture_%d_" % cycle
        examples, labels = [], []
        for gesture_index in range(number_of_gestures):
            examples_to_format = []
            for line in open(path_emg + '%d.txt' % gesture_index):
                #  strip() remove the "\n" character, split separate the data in a list. np.float
                #  transform each element of the list from a str to a float
                emg_signal = np.float32(line.strip().split(","))
                examples_to_format.append(emg_signal)
            if cycle == 1:  # This cycle is the second cycle and correspond to the highest effort baseline. Record it.
                if gesture_index == 0:
                    highest_activation_per_gesture.append(0)
                else:
                    highest_activation_per_gesture.append(get_highest_average_emg_window(
                        examples_to_format, window_for_moving_average=window_size))

            examples_formatted = format_examples(examples_to_format,
                                                 feature_set_function=feature_set_function, window_size=window_size,
                                                 size_non_overlap=size_non_overlap)
            examples.extend(examples_formatted)
            labels.extend(np.ones(len(examples_formatted)) * gesture_index)
        print("SHAPE SESSION EXAMPLES: ", np.shape(examples))
        examples_training.append(examples)
        labels_training.append(labels)

    return examples_training, labels_training, highest_activation_per_gesture


def get_data_and_process_it_from_file(path, feature_set_function, number_of_gestures=11, number_of_cycles=4,
                                      window_size=150, size_non_overlap=50):
    examples_training_sessions_datasets, labels_training_sessions_datasets = [], []
    highest_activation_participants = []
    examples_evaluation_sessions_datasets, labels_evaluation_sessions_datasets, timestamps_emg_evaluation = [], [], []
    angles_with_timestamps_emg_evaluation = []

    training_datetimes, evaluation_datetimes = [], []
    for index_participant in range(22):
        # Those two participant did not complete the experiment
        if index_participant != 10 and index_participant != 11:
            folder_participant = "/Participant" + str(index_participant)
            sessions_directories = os.listdir(path + folder_participant)

            training_datetime, evaluation_datetime = load_timestamps_from_participant(path + folder_participant)
            training_datetimes.append(training_datetime)
            evaluation_datetimes.append(evaluation_datetime)

            examples_participant_training_sessions, labels_participant_training_sessions = [], []
            highest_activation_per_session = []
            examples_participant_evaluation_sessions, labels_participant_evaluation_sessions = [], []
            timestamps_evaluation_participant_sessions, angles_with_timestamps_participant_sessions = [], []
            for session_directory in sessions_directories:

                if "Training" in session_directory:
                    path_folder_examples = path + "/" + folder_participant + "/" + session_directory + "/EMG"
                    examples_training, labels_training, highest_activation_per_gesture = \
                        read_files_to_format_training_session(path_folder_examples=path_folder_examples,
                                                              number_of_cycles=number_of_cycles,
                                                              number_of_gestures=number_of_gestures,
                                                              window_size=window_size,
                                                              size_non_overlap=size_non_overlap,
                                                              feature_set_function=feature_set_function)
                    if len(examples_training) > 0:
                        # These instances, the participant only recorded one cycle of training. Skip it
                        skip_it = False
                        for participant_session_to_skip in list_participant_training_1_to_skip:
                            if participant_session_to_skip in path_folder_examples:
                                skip_it = True
                        if skip_it is False:
                            examples_participant_training_sessions.append(examples_training)
                            labels_participant_training_sessions.append(labels_training)
                            highest_activation_per_session.append(highest_activation_per_gesture)
                if "Evaluation" in session_directory:
                    path_folder_examples = path + "/" + folder_participant + "/" + session_directory
                    examples_evaluation, labels_evaluation, timestamps_emg, \
                    angles_with_timestamps = read_files_to_format_evaluation_session(
                        path_folder_examples, window_size, size_non_overlap)
                    if len(examples_evaluation) > 0:
                        # These instances, the participant only recorded one cycle of training. Skip it
                        skip_it = False
                        for participant_session_to_skip in list_participant_training_1_to_skip:
                            if participant_session_to_skip in path_folder_examples:
                                skip_it = True
                        if skip_it is False:
                            examples_participant_evaluation_sessions.append(examples_evaluation)
                            labels_participant_evaluation_sessions.append(labels_evaluation)
                            timestamps_evaluation_participant_sessions.append(timestamps_emg)
                            angles_with_timestamps_participant_sessions.append(angles_with_timestamps)

            print(np.shape(examples_participant_evaluation_sessions))
            examples_training_sessions_datasets.append(examples_participant_training_sessions)
            labels_training_sessions_datasets.append(labels_participant_training_sessions)
            highest_activation_participants.append(highest_activation_per_session)

            examples_evaluation_sessions_datasets.append(examples_participant_evaluation_sessions)
            labels_evaluation_sessions_datasets.append(labels_participant_evaluation_sessions)
            timestamps_emg_evaluation.append(timestamps_evaluation_participant_sessions)
            angles_with_timestamps_emg_evaluation.append(angles_with_timestamps_participant_sessions)

    dataset_dictionnary = {"examples_training": np.array(examples_training_sessions_datasets),
                           "labels_training": np.array(labels_training_sessions_datasets),
                           "training_datetimes": np.array(training_datetimes),
                           "highest_activations": np.array(highest_activation_participants),
                           "examples_evaluation": np.array(examples_evaluation_sessions_datasets),
                           "labels_evaluation": np.array(labels_evaluation_sessions_datasets),
                           "evaluation_emg_timestamps": np.array(timestamps_emg_evaluation),
                           "angles_and_timestamps": np.array(angles_with_timestamps_emg_evaluation),
                           "evaluation_datetimes": np.array(evaluation_datetimes)}

    return dataset_dictionnary


def read_data_training(path, features_set_name, feature_set_function, number_of_gestures=11, number_of_cycles=4,
                       window_size=150, size_non_overlap=50):
    print("Loading and preparing Training datasets...")
    'Get and process the train data'
    print("Taking care of the training data...")
    dataset_dictionnary = get_data_and_process_it_from_file(path=path, number_of_gestures=number_of_gestures,
                                                            number_of_cycles=number_of_cycles, window_size=window_size,
                                                            size_non_overlap=size_non_overlap,
                                                            feature_set_function=feature_set_function)

    training_session_dataset_dictionnary = {}
    training_session_dataset_dictionnary["examples_training"] = dataset_dictionnary["examples_training"]
    training_session_dataset_dictionnary["labels_training"] = dataset_dictionnary["labels_training"]
    training_session_dataset_dictionnary["training_datetimes"] = dataset_dictionnary["training_datetimes"]
    training_session_dataset_dictionnary["highest_activations"] = dataset_dictionnary["highest_activations"]

    evaluation_session_dataset_dictionnary = {}
    evaluation_session_dataset_dictionnary["examples_evaluation"] = dataset_dictionnary["examples_evaluation"]
    evaluation_session_dataset_dictionnary["labels_evaluation"] = dataset_dictionnary["labels_evaluation"]
    evaluation_session_dataset_dictionnary["evaluation_emg_timestamps"] = dataset_dictionnary[
        "evaluation_emg_timestamps"]
    evaluation_session_dataset_dictionnary["evaluation_datetimes"] = dataset_dictionnary["evaluation_datetimes"]
    evaluation_session_dataset_dictionnary["angles_and_timestamps"] = dataset_dictionnary["angles_and_timestamps"]

    with open("../Processed_datasets/%s_training_session.pickle" % features_set_name, 'wb') as f:
        pickle.dump(training_session_dataset_dictionnary, f, pickle.HIGHEST_PROTOCOL)
    with open("../Processed_datasets/%s_evaluation_session.pickle" % features_set_name, 'wb') as f:
        pickle.dump(evaluation_session_dataset_dictionnary, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    features_set_name = "TD_features_set_"
    feature_set_function = feature_extraction.get_TD_features_set
    read_data_training(path="../../datasets/longterm_dataset_3DC", features_set_name=features_set_name,
                       feature_set_function=feature_set_function)
    
    features_set_name = "EnhancedTD_features_set"
    feature_set_function = feature_extraction.get_enhancedTD_feature_set
    read_data_training(path="../../datasets/longterm_dataset_3DC", features_set_name=features_set_name,
                       feature_set_function=feature_set_function)
    
    features_set_name = "Sampen_features_set"
    feature_set_function = feature_extraction.get_Sampen_pipeline_features_set
    read_data_training(path="../../datasets/longterm_dataset_3DC", features_set_name=features_set_name,
                       feature_set_function=feature_set_function)
    
    features_set_name = "LSF9_features_set"
    feature_set_function = feature_extraction.get_LSF9
    read_data_training(path="../../datasets/longterm_dataset_3DC", features_set_name=features_set_name,
                       feature_set_function=feature_set_function)
    
    features_set_name = "TDPSD_features_set"
    feature_set_function = feature_extraction.get_TDPSD
    read_data_training(path="../../datasets/longterm_dataset_3DC", features_set_name=features_set_name,
                       feature_set_function=feature_set_function)
    
    features_set_name = "NinaPro_features_set"
    feature_set_function = feature_extraction.get_NinaPro_best
    read_data_training(path="../../datasets/longterm_dataset_3DC", features_set_name=features_set_name,
                       feature_set_function=feature_set_function)

    features_set_name = "TSD_features_set"
    feature_set_function = feature_extraction.getTSD
    read_data_training(path="../../datasets/longterm_dataset_3DC", features_set_name=features_set_name,
                       feature_set_function=feature_set_function)
