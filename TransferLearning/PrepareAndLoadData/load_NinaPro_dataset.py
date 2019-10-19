import scipy.io as sio
import numpy as np
import os
from scipy import signal

number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5

def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    example = []
    for value_armband in vector_to_format:
        if (example == []):
            example = value_armband
        else:
            example = np.row_stack((example, value_armband))
        if len(example) >= number_of_vector_per_example:
            example = example.transpose()
            dataset_example_formatted.append(example)
            example = example.transpose()
            example = example[size_non_overlap:]
    # Apply the butterworth high pass filter at 2Hz
    print(np.shape(dataset_example_formatted))
    dataset_high_pass_filtered = []
    for example in dataset_example_formatted:
        example_filtered = []
        for channel_example in example:
            example_filtered.append(butter_bandpass_filter(channel_example, 5, 99, 200))
        dataset_high_pass_filtered.append([example_filtered])
    return np.array(dataset_high_pass_filtered)

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

def get_data(path, modality="cwt"):
    print(os.walk(path))
    all_subjects_train_examples = []
    all_subjects_train_labels = []
    all_subjects_test_examples = []
    all_subjects_test_labels = []
    for subject_index in range(1, 11):
        paths_data = [path + 's' + str(subject_index) + '/S' + str(subject_index) + '_E2_A1.mat']
        train_examples = []
        train_labels = []
        test_examples = []
        test_labels = []
        number_class = 0
        for k, path_data in enumerate(paths_data):
            mat_contents = sio.loadmat(path_data)
            emg_data = mat_contents['emg'][:, 8::]  # Get the data from the second EMG
            labels_total = np.squeeze(mat_contents['restimulus'])
            print(np.shape(labels_total))

            dictionnary_regrouped_emg_repetitions = {}
            emg_big_window = []
            last_label = labels_total[0]
            for emg_entry, label_entry in zip(emg_data, labels_total):
                if label_entry != last_label:
                    if last_label in dictionnary_regrouped_emg_repetitions:
                        dictionnary_regrouped_emg_repetitions[last_label].append(emg_big_window)
                    else:
                        dictionnary_regrouped_emg_repetitions[last_label] = [emg_big_window]
                    emg_big_window = []

                last_label = label_entry
                emg_big_window.append(emg_entry)

            dictionnary_regrouped_emg_repetitions[last_label].append(emg_big_window)

            print(np.shape(dictionnary_regrouped_emg_repetitions[17]))

            keys_labels = dictionnary_regrouped_emg_repetitions.keys()
            print(keys_labels)
            if k == 0:
                number_class = len(labels_total)

            for key in keys_labels:
                for i, examples in enumerate(dictionnary_regrouped_emg_repetitions[key]):
                    if key == 0 and i > 5:
                        break
                    # Calculate the examples and labels according to the required window
                    if modality == "raw":
                        dataset_example = format_data_to_train(examples)
                    elif modality == "spectrogram":
                        dataset_example = format_data_to_train(examples)
                    else:
                        dataset_example = format_data_to_train(examples)
                    if i in [0, 1, 2, 3]:  # To follow the NinaPro article
                        train_examples.append(dataset_example)
                        train_labels.append(key+np.zeros(len(dataset_example)))
                    else:
                        test_examples.append(dataset_example)
                        test_labels.append(key+np.zeros(len(dataset_example)))
                        print(key+np.zeros(len(dataset_example)))
        print(np.shape(train_examples))
        all_subjects_train_examples.append(train_examples)
        all_subjects_train_labels.append(train_labels)
        all_subjects_test_examples.append(test_examples)
        all_subjects_test_labels.append(test_labels)

    return all_subjects_train_examples, all_subjects_train_labels, all_subjects_test_examples, all_subjects_test_labels

if __name__ == '__main__':
    get_data("datasetNinaPro\\")