import os
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    lowcut_normalized = lowcut / nyq
    highcut_normalized = highcut / nyq
    b, a = signal.butter(N=order, Wn=[lowcut_normalized, highcut_normalized], btype='band', output="ba")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def applies_high_pass_for_dataset(dataset, frequency):
    dataset_to_return = []
    for example in dataset:
        example_formatted = []
        for vector_electrode in example:
            filtered_signal = butter_bandpass_filter(vector_electrode, 20, 495, frequency)
            example_formatted.append(filtered_signal)
        dataset_to_return.append(example_formatted)
    return dataset_to_return


def show_filtered_signal(noisy_signal, filtered_signal, fs=1000):
    plt.plot(noisy_signal, label='Noisy signal (%g Hz)' % fs)
    plt.plot(filtered_signal, label='Filtered signal (%g Hz)' % fs)
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


def load_timestamps_from_participant(path_participant):
    list_participant_training_1_to_skip = ["Participant0/", "Participant2/"]
    path_participant += "/"
    training_datetime = []
    index_training = 0
    evaluation_datetime = []
    index_evaluation = 0

    trainings_and_evaluations_directories = os.listdir(path_participant)
    for directory in trainings_and_evaluations_directories:
        if "Training" in directory:
            print("PATH TIMESTAMP: ", path_participant)
            # Check to see if there's file in the directory
            if (list_participant_training_1_to_skip[0] not in path_participant and
                    list_participant_training_1_to_skip[1] not in path_participant) or index_training != 1:
                if len(os.listdir(path_participant + directory + "/EMG")) > 0:
                    path_timestamp = path_participant + directory + "/EMG/3dc_EMG_Timestamp_gesture_0_0.txt"
                    for line in open(path_timestamp):
                        # [:len(line_timestamp) - 1], the -1 is to remove the '\n' character
                        date_and_time = datetime.fromisoformat(line.strip())
                        training_datetime.append(date_and_time)
                        # We only need the first line
                        break
            index_training += 1
        elif "Evaluation" in directory:
            # Check to see if there's file in the directory
            if len(os.listdir(path_participant + directory)) > 0:
                path_timestamp = path_participant + directory + "/3dc_EMG_Timestamp_gesture.txt"
                for line in open(path_timestamp):
                    # [:len(line_timestamp) - 1], the -1 is to remove the '\n' character
                    date_and_time = datetime.fromisoformat(line.strip())
                    evaluation_datetime.append(date_and_time)
                    # We only need the first line
                    break
                index_evaluation += 1
    return training_datetime, evaluation_datetime


# Get the pitch and yaw (in Degrees) for all the participant's arm during the evaluation session
def get_angles_from_positions_3d_arm(list_positions):
    original_wrist_position = np.array(list_positions[0]['wristPosition'])
    original_elbow_position = np.array(list_positions[0]['elbowPosition'])
    vector_arm_neutral = original_elbow_position-original_wrist_position
    vector_arm_neutral_normalized = vector_arm_neutral/np.linalg.norm(vector_arm_neutral)
    i = 0
    list_angles_arm = []
    for arm in list_positions:
        wrist_position = np.array(arm['wristPosition'])
        elbow_position = np.array(arm['elbowPosition'])
        vector_arm = elbow_position - wrist_position
        vector_arm_normalized = vector_arm/np.linalg.norm(vector_arm)
        pitch = math.asin(vector_arm_normalized[1])
        # Output in radian, get degrees
        pitch_degree = pitch*(180/math.pi)

        vector_arm_normalized_compare_to_neutral = vector_arm_normalized - vector_arm_neutral_normalized
        norm_vector_comparing = np.linalg.norm(vector_arm_normalized_compare_to_neutral)
        if norm_vector_comparing > 0:
            vector_arm_normalized_compare_to_neutral /= norm_vector_comparing
        yaw = math.asin(
            vector_arm_normalized_compare_to_neutral[2] /
            math.sqrt(vector_arm_normalized_compare_to_neutral[2]**2+vector_arm_normalized_compare_to_neutral[0]**2))
        # Output in radian, get degrees
        yaw_degree = yaw*(180/math.pi)
        angles_degrees = {"pitch": pitch_degree, "yaw": yaw_degree, "timestamp": arm['timestamp']}
        list_angles_arm.append(angles_degrees)
    return list_angles_arm


if __name__ == '__main__':
    load_timestamps_from_participant("../../datasets/longterm_dataset_3DC/Participant2")
