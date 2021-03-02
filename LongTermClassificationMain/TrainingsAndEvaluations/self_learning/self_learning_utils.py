import numpy as np
from scipy.stats import mode
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset

np.set_printoptions(precision=3, suppress=True)


def look_back_and_re_label(network_outputs, index_start_look_back, len_look_back=10):
    data_uncertain = network_outputs[np.max((0, index_start_look_back - len_look_back)):index_start_look_back]
    index_associated_with_max_entropy = np.max((0, index_start_look_back - len_look_back))
    entropies = []
    for output in data_uncertain:
        entropies.append(entropy(output))
    discrete_entropy_derivative = np.diff(entropies)

    for index in range(len(discrete_entropy_derivative)):
        if discrete_entropy_derivative[index] > 0.25:
            index_associated_with_max_entropy = index_associated_with_max_entropy + index
            break
    return int(index_associated_with_max_entropy)


def pseudo_labels_heuristic_training_sessions(predictions, model_outputs, window_stable_mode_length=30,
                                              percentage_same_gesture_now_stable=0.85,
                                              maximum_length_instability_same_gesture=50,
                                              maximum_length_instability_gesture_transition=50, use_look_back=False,
                                              len_look_back=10):
    pseudo_labels_sessions = []
    indexes_associated_with_pseudo_labels_sessions = []
    for session_index in range(len(predictions)):
        print(predictions[session_index])
        pseudo_labels, indexes_associated = pseudo_labels_heuristic(
            predictions[session_index], model_outputs[session_index],
            window_stable_mode_length=window_stable_mode_length,
            percentage_same_gesture_now_stable=percentage_same_gesture_now_stable,
            maximum_length_instability_same_gesture=maximum_length_instability_same_gesture,
            maximum_length_instability_gesture_transition=maximum_length_instability_gesture_transition,
            use_look_back=use_look_back,
            len_look_back=len_look_back)
        pseudo_labels_sessions.append(pseudo_labels)
        indexes_associated_with_pseudo_labels_sessions.append(indexes_associated)
    return pseudo_labels_sessions, indexes_associated_with_pseudo_labels_sessions


def pseudo_labels_heuristic(predictions, model_outputs, window_stable_mode_length=30,
                            percentage_same_gesture_now_stable=0.85, maximum_length_instability_same_gesture=50,
                            maximum_length_instability_gesture_transition=50, use_look_back=False,
                            len_look_back=10):
    predictions_heuristic = []
    predictions_heuristic_index = []
    current_class = np.argmax(np.median(model_outputs[0:window_stable_mode_length], axis=0))
    stable_predictions = True

    index_unstable_start = 0
    model_output_in_unstable_mode = []
    for index, (prediction, model_output) in enumerate(zip(predictions,
                                                           model_outputs)):
        if prediction != current_class and stable_predictions:
            stable_predictions = False
            index_unstable_start = index

            model_output_in_unstable_mode = []

        if stable_predictions is False:
            model_output_in_unstable_mode.append(model_output)
        if len(model_output_in_unstable_mode) > window_stable_mode_length:
            model_output_in_unstable_mode = model_output_in_unstable_mode[1:]

        if stable_predictions is False:
            if len(model_output_in_unstable_mode) >= window_stable_mode_length:
                '''
                mode_gesture_frequency = mode(model_output_in_unstable_mode)[1][0]
                most_prevalent_gesture = mode(model_output_in_unstable_mode)[0][0]

                print(mode_gesture)
                '''
                # print(np.shape(model_output_in_unstable_mode))
                medians = np.median(np.array(model_output_in_unstable_mode), axis=0)
                medians_percentage = medians / np.sum(medians)
                most_prevalent_gesture = np.argmax(medians_percentage)
                if medians_percentage[most_prevalent_gesture] > percentage_same_gesture_now_stable:
                    stable_predictions = True
                    # Determine if this period of instability was due to a gesture transition or if it was due to
                    # a bad classification from the classifier
                    old_class = current_class
                    current_class = most_prevalent_gesture
                    # After the unstability, we are still on the same gesture. Check if it took too long to solve
                    # If it was ignore all the predictions up to that point. If the period of instability did not
                    # last too long, correct all predictions to be of the current class
                    if current_class == old_class:
                        if index - index_unstable_start < maximum_length_instability_same_gesture:
                            # Add all the predictions made during instability to the pseudo labels and set them to
                            # the current class to the current class
                            for index_to_add in range(index_unstable_start, index + 1):
                                predictions_heuristic.append(current_class)
                                predictions_heuristic_index.append(index_to_add)
                        # Else: Ignore all the predictions made during the period of instability.
                    else:  # current class != old_class, therefore we just experienced a transition. Act accordingly
                        if index - index_unstable_start < maximum_length_instability_gesture_transition:
                            # Add all the predictions made during instability to the pseudo labels and set them to
                            # to the new class as they were all part of a transition between two gestures
                            for index_to_add in range(index_unstable_start, index + 1):
                                predictions_heuristic.append(current_class)
                                predictions_heuristic_index.append(index_to_add)
                            if use_look_back:
                                index_from_where_to_change = look_back_and_re_label(
                                    network_outputs=model_outputs,
                                    index_start_look_back=index_unstable_start,
                                    len_look_back=len_look_back)

                                index_from_where_to_change_in_pseudolabels = None
                                for i in range(index_from_where_to_change,
                                               index_unstable_start):
                                    if i in predictions_heuristic_index:
                                        index_from_where_to_change_in_pseudolabels = \
                                            predictions_heuristic_index.index(i)
                                        break
                                # print(predictions_heuristic)
                                if index_from_where_to_change_in_pseudolabels is not None:
                                    for index_to_change in range(index_from_where_to_change_in_pseudolabels,
                                                                 np.min((
                                                                         index_from_where_to_change_in_pseudolabels +
                                                                         len_look_back,
                                                                         len(predictions_heuristic)))):
                                        predictions_heuristic[index_to_change] = current_class
                                # print(predictions_heuristic)

                        # Else: Ignore all the predictions made during the period of instability.
                    model_output_in_unstable_mode = []
        else:
            predictions_heuristic.append(prediction)
            predictions_heuristic_index.append(index)

    # Add the remaining few predictions to the current gestures (in actual real-time scenario, that last part would
    # not exist. But we are working with a finite dataset
    if stable_predictions is False and \
            len(predictions) - index_unstable_start < window_stable_mode_length * percentage_same_gesture_now_stable:
        gesture_to_add = np.argmax(np.median(model_outputs, axis=0))
        for index_to_add in range(index_unstable_start, len(predictions)):
            predictions_heuristic.append(gesture_to_add)
            predictions_heuristic_index.append(index_to_add)

    return predictions_heuristic, predictions_heuristic_index


def get_index_that_started_the_class_change(data_uncertain):
    data_uncertain = np.array(data_uncertain)
    index_associated_with_max_entropy = data_uncertain[0, 0]

    discret_entropy_derivative = np.diff(data_uncertain[:, 2])

    for index in range(len(discret_entropy_derivative)):
        if discret_entropy_derivative[index] > 0.25:
            index_associated_with_max_entropy = data_uncertain[index + 1, 0]
            break
    return int(index_associated_with_max_entropy)


def generate_pseudo_labels_from_predictions(predictions, model_outputs, mode_predictions_smoothing_size=30,
                                            use_anomaly_detection=True):
    pseudo_labels_sessions = []
    for session_index in range(len(predictions)):
        predictions_heuristic = []
        array_to_consider = np.array(model_outputs[session_index][0: len(model_outputs[session_index])])
        current_class_mode = np.argmax(np.median(array_to_consider, axis=0))
        current_class = current_class_mode
        data_uncertain = []
        rolling_median_argmax = []
        for index, (prediction, model_output) in enumerate(zip(predictions[session_index],
                                                               model_outputs[session_index])):
            entropy_current_prediction = entropy(model_output)

            if len(data_uncertain) > mode_predictions_smoothing_size:
                data_uncertain = data_uncertain[1:]
            data_uncertain.append([index, prediction, entropy_current_prediction])

            if len(rolling_median_argmax) > mode_predictions_smoothing_size:
                rolling_median_argmax = rolling_median_argmax[1:]
            rolling_median_argmax.append(model_output)

            array_to_consider = np.array(model_outputs[session_index][
                                         np.max((0, index - mode_predictions_smoothing_size)):
                                         np.min((len(model_outputs[session_index]),
                                                 index + mode_predictions_smoothing_size))])
            current_class_mode = np.argmax(np.median(array_to_consider, axis=0))

            if current_class != current_class_mode:
                current_class = current_class_mode
                if use_anomaly_detection:
                    index_from_where_to_change = get_index_that_started_the_class_change(data_uncertain=
                                                                                         data_uncertain)
                    for index_to_change in range(index_from_where_to_change, index):
                        predictions_heuristic[index_to_change] = current_class

            predictions_heuristic.append(current_class)
        pseudo_labels_sessions.append(predictions_heuristic)
    return pseudo_labels_sessions


def percentage_mode_heuristic(predictions, percentage_needed=.45):
    pseudo_labels_sessions = []
    for session_index in range(len(predictions)):
        most_frequent_gesture, frequency = mode(predictions[session_index])
        most_frequent_gesture, frequency = most_frequent_gesture[0], frequency[0]
        print("GESTURE: ", most_frequent_gesture, "Frequency percentage: ", frequency / len(predictions[session_index]))
        print(predictions[session_index])
        if frequency / len(predictions[session_index]) > percentage_needed:
            pseudo_labels = []
            for index in range(len(predictions[session_index])):
                pseudo_labels.append(most_frequent_gesture)
            pseudo_labels_sessions.append(pseudo_labels)
        else:
            pseudo_labels_sessions.append([])
    return pseudo_labels_sessions


def generate_pseudo_labels_MultipleVotes(predictions, model_outputs,
                                         mode_predictions_smoothing_size=20):
    predictions_heuristic = []
    for index in range(len(predictions)):
        array_to_consider = np.array(model_outputs[np.max((0, index - mode_predictions_smoothing_size)):
                                                   np.min((len(model_outputs),
                                                           index + mode_predictions_smoothing_size))])
        predictions_heuristic.append(np.argmax(np.median(array_to_consider, axis=0)))
    return predictions_heuristic


def generate_pseudo_labels_MultipleVotes_training_session(predictions, model_outputs,
                                                          mode_predictions_smoothing_size=20):
    pseudo_labels_sessions = []
    for session_index in range(len(predictions)):
        predictions_heuristic = generate_pseudo_labels_MultipleVotes(predictions[session_index],
                                                                     model_outputs[session_index],
                                                                     mode_predictions_smoothing_size=
                                                                     mode_predictions_smoothing_size)
        pseudo_labels_sessions.append(predictions_heuristic)
    return pseudo_labels_sessions


def segment_dataset_by_gesture_to_remove_transitions(ground_truths, predictions, model_outputs, examples):
    ground_truth_segmented_session = []
    predictions_segmented_session = []
    model_outputs_segmented_session = []
    examples_segmented_session = []

    ground_truth_segmented_gesture = []
    predictions_segmented_gesture = []
    model_outputs_segmented_gesture = []
    examples_segmented_gesture = []
    current_label = ground_truths[0]
    for example_index in range(len(ground_truths)):
        if current_label != ground_truths[example_index]:
            ground_truth_segmented_session.append(ground_truth_segmented_gesture)
            predictions_segmented_session.append(predictions_segmented_gesture)
            model_outputs_segmented_session.append(model_outputs_segmented_gesture)
            examples_segmented_session.append(examples_segmented_gesture)
            ground_truth_segmented_gesture = []
            predictions_segmented_gesture = []
            model_outputs_segmented_gesture = []
            examples_segmented_gesture = []
            current_label = ground_truths[example_index]
        ground_truth_segmented_gesture.append(ground_truths[example_index])
        predictions_segmented_gesture.append(predictions[example_index])
        model_outputs_segmented_gesture.append(model_outputs[example_index])
        examples_segmented_gesture.append(examples[example_index])
    ground_truth_segmented_session.append(ground_truth_segmented_gesture)
    predictions_segmented_session.append(predictions_segmented_gesture)
    model_outputs_segmented_session.append(model_outputs_segmented_gesture)
    examples_segmented_session.append(examples_segmented_gesture)
    return ground_truth_segmented_session, predictions_segmented_session, model_outputs_segmented_session, \
           examples_segmented_session


def generate_dataloaders_for_MultipleVote_Evaluation(dataloader_session_training, dataloader_sessions_evaluation, model,
                                                     current_session, validation_set_ratio=0.2, batch_size=256):
    examples_all_session = []
    labels_all_session = []

    # Create the training and validation dataset to train on
    for batch in dataloader_session_training:
        with torch.no_grad():
            inputs, labels_batch = batch
            labels_all_session.extend(labels_batch.numpy())
            examples_all_session.extend(inputs.numpy())

    for session_index in range(current_session + 1):
        print("HANDLING NEW SESSION")
        # This is the first session where we have real labels. Use them
        model.eval()
        if session_index % 2 == 0:
            examples_self_learning = []
            ground_truth_self_learning = []
            model_outputs_self_learning = []
            predicted_labels_self_learning = []
            # Get the predictions and model output for the pseudo label dataset
            model.eval()
            for batch in dataloader_sessions_evaluation[session_index]:
                with torch.no_grad():
                    inputs_batch, labels_batch = batch
                    inputs_batch = inputs_batch.cuda()

                    outputs = model(inputs_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    model_outputs_self_learning.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                    predicted_labels_self_learning.extend(predicted.cpu().numpy())
                    ground_truth_self_learning.extend(labels_batch.numpy())
                    examples_self_learning.extend(inputs_batch.cpu().numpy())

            # equivalent to 1second of data as selected in https://www.frontiersin.org/articles/10.3389/fnins.2017.00379/full
            pseudo_labels = generate_pseudo_labels_MultipleVotes(
                predicted_labels_self_learning, model_outputs_self_learning, mode_predictions_smoothing_size=20)

            examples = []
            print("BEFORE: ", np.mean(np.array(ground_truth_self_learning) ==
                                      np.array(predicted_labels_self_learning)), "  AFTER: ", np.mean(
                np.array(ground_truth_self_learning) ==
                np.array(pseudo_labels)), " len before: ",
                  len(ground_truth_self_learning), "  len after: ",
                  len(pseudo_labels))
            examples.extend(np.array(examples_self_learning).tolist())

            examples_all_session.extend(examples)
            labels_all_session.extend(pseudo_labels)

    # Create the dataloaders associated with the NEW session
    X_pseudo, X_valid_pseudo, Y_pseudo, Y_valid_pseudo = train_test_split(examples_all_session, labels_all_session,
                                                                          test_size=validation_set_ratio, shuffle=True)

    validation_pseudo = TensorDataset(torch.from_numpy(np.array(X_valid_pseudo, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_pseudo, dtype=np.int64)))
    validationloader_pseudo = torch.utils.data.DataLoader(validation_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=False)
    train_pseudo = TensorDataset(torch.from_numpy(np.array(X_pseudo, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_pseudo, dtype=np.int64)))
    train_dataloader_pseudo = torch.utils.data.DataLoader(train_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    return train_dataloader_pseudo, validationloader_pseudo


def generate_dataloaders_for_MultipleVote(dataloader_sessions, model, current_session, validation_set_ratio=0.2,
                                          batch_size=256, use_recalibration_data=True):
    examples_all_session = []
    labels_all_session = []
    for session_index in range(current_session + 1):
        print("HANDLING NEW SESSION")
        # This is the first session where we have real labels. Use them
        model.eval()
        if session_index == 0:
            # Create the training and validation dataset to train on
            for batch in dataloader_sessions[0]:
                with torch.no_grad():
                    inputs, labels_batch = batch
                    labels_all_session.extend(labels_batch.numpy())
                    examples_all_session.extend(inputs.numpy())
        elif (session_index % 2 == 0 and use_recalibration_data is False) or session_index == current_session:
            examples_self_learning = []
            ground_truth_self_learning = []
            model_outputs_self_learning = []
            predicted_labels_self_learning = []
            # Get the predictions and model output for the pseudo label dataset
            model.eval()
            for batch in dataloader_sessions[session_index]:
                with torch.no_grad():
                    inputs_batch, labels_batch = batch
                    inputs_batch = inputs_batch.cuda()

                    outputs = model(inputs_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    model_outputs_self_learning.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                    predicted_labels_self_learning.extend(predicted.cpu().numpy())
                    ground_truth_self_learning.extend(labels_batch.numpy())
                    examples_self_learning.extend(inputs_batch.cpu().numpy())

            # Segment the data in respect to the recorded gestures
            ground_truth_segmented, predictions_segmented, model_outputs_segmented, examples_segmented = \
                segment_dataset_by_gesture_to_remove_transitions(ground_truths=ground_truth_self_learning,
                                                                 predictions=predicted_labels_self_learning,
                                                                 model_outputs=model_outputs_self_learning,
                                                                 examples=examples_self_learning)

            # equivalent to 1second of data as selected in https://www.frontiersin.org/articles/10.3389/fnins.2017.00379/full
            pseudo_labels_segmented = generate_pseudo_labels_MultipleVotes_training_session(
                predictions_segmented, model_outputs_segmented, mode_predictions_smoothing_size=20)

            pseudo_labels = []
            examples = []
            for index_segment in range(len(pseudo_labels_segmented)):
                pseudo_labels.extend(pseudo_labels_segmented[index_segment])
                print("BEFORE: ", np.mean(np.array(ground_truth_segmented[index_segment]) ==
                                          np.array(predictions_segmented[index_segment])), "  AFTER: ", np.mean(
                    np.array(ground_truth_segmented[index_segment]) ==
                    np.array(pseudo_labels_segmented[index_segment])), " len before: ",
                      len(ground_truth_segmented[index_segment]), "  len after: ",
                      len(pseudo_labels_segmented[index_segment]))
                examples.extend(np.array(examples_segmented[index_segment]).tolist())

            examples_all_session.extend(examples)
            labels_all_session.extend(pseudo_labels)

    # Create the dataloaders associated with the NEW session
    X_pseudo, X_valid_pseudo, Y_pseudo, Y_valid_pseudo = train_test_split(examples_all_session, labels_all_session,
                                                                          test_size=validation_set_ratio, shuffle=True)

    validation_pseudo = TensorDataset(torch.from_numpy(np.array(X_valid_pseudo, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_pseudo, dtype=np.int64)))
    validationloader_pseudo = torch.utils.data.DataLoader(validation_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=False)
    train_pseudo = TensorDataset(torch.from_numpy(np.array(X_pseudo, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_pseudo, dtype=np.int64)))
    train_dataloader_pseudo = torch.utils.data.DataLoader(train_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    return train_dataloader_pseudo, validationloader_pseudo


def generate_dataloaders_evaluation_for_SCADANN(dataloader_session_training, dataloader_sessions_evaluation, models,
                                                current_session, validation_set_ratio=0.2, batch_size=256,
                                                use_recalibration_data=True):
    examples_replay = []
    labels_replay = []

    examples_new_session = []
    labels_new_session = []
    # Create the training and validation dataset to train on
    for batch in dataloader_session_training:
        with torch.no_grad():
            inputs, labels_batch = batch
            labels_replay.extend(labels_batch.numpy())
            examples_replay.extend(inputs.numpy())

    for session_index in range(0, current_session + 1):
        print("HANDLING NEW SESSION")
        # This is the first session where we have real labels. Use them
        models[int(session_index / 2)].eval()
        # There is two evaluation sessions per day, we only have access to the first one.
        # if session_index % 2 == 0:
        if (session_index % 2 == 0 and use_recalibration_data is False) or session_index == current_session:
            # if session_index == current_session:
            examples_self_learning = []
            ground_truth_self_learning = []
            model_outputs_self_learning = []
            predicted_labels_self_learning = []
            # Get the predictions and model output for the pseudo label dataset
            models[int(session_index / 2)].eval()
            # model.apply(model.apply_dropout)
            for batch in dataloader_sessions_evaluation[session_index]:
                with torch.no_grad():
                    inputs_batch, labels_batch = batch
                    inputs_batch = inputs_batch.cuda()

                    outputs = models[int(session_index / 2)](inputs_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    model_outputs_self_learning.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                    predicted_labels_self_learning.extend(predicted.cpu().numpy())
                    ground_truth_self_learning.extend(labels_batch.numpy())
                    examples_self_learning.extend(inputs_batch.cpu().numpy())
            pseudo_labels, indexes_pseudo = pseudo_labels_heuristic(
                predicted_labels_self_learning, model_outputs_self_learning, window_stable_mode_length=30,
                percentage_same_gesture_now_stable=.65,
                maximum_length_instability_gesture_transition=40,
                maximum_length_instability_same_gesture=40, use_look_back=True)

            examples = []
            ground_truth_reduced = []

            examples.extend(np.array(examples_self_learning)[
                                indexes_pseudo].tolist())
            ground_truth_reduced.extend(np.array(ground_truth_self_learning)[
                                            indexes_pseudo].tolist())

            accuracy_before = np.mean(np.array(ground_truth_self_learning) ==
                                      np.array(predicted_labels_self_learning))
            accuracy_after = np.mean(np.array(ground_truth_reduced) ==
                                     np.array(pseudo_labels))
            print("ACCURACY MODEL: ", accuracy_before, "  Accuracy pseudo:", accuracy_after, " len pseudo: ",
                  len(pseudo_labels), "   len predictions", len(predicted_labels_self_learning))
            print("CURRENT SESSION: ", current_session, " Session Index: ", session_index)
            if session_index == current_session:
                examples_new_session.extend(examples)
                labels_new_session.extend(pseudo_labels)
            else:
                examples_replay.extend(examples)
                labels_replay.extend(pseudo_labels)

    # Create the dataloaders associated with the REPLAY data
    X_replay, X_valid_replay, Y_replay, Y_valid_replay = train_test_split(examples_replay, labels_replay,
                                                                          test_size=0.01,
                                                                          shuffle=True)

    validation_replay = TensorDataset(torch.from_numpy(np.array(X_valid_replay, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_replay, dtype=np.int64)))
    validationloader_replay = torch.utils.data.DataLoader(validation_replay, batch_size=batch_size, shuffle=True,
                                                          drop_last=False)
    train_replay = TensorDataset(torch.from_numpy(np.array(X_replay, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_replay, dtype=np.int64)))
    train_dataloader_replay = torch.utils.data.DataLoader(train_replay, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    # Create the dataloaders associated with the NEW session
    X_pseudo, X_valid_pseudo, Y_pseudo, Y_valid_pseudo = train_test_split(examples_new_session, labels_new_session,
                                                                          test_size=validation_set_ratio,
                                                                          shuffle=True)

    validation_pseudo = TensorDataset(torch.from_numpy(np.array(X_valid_pseudo, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_pseudo, dtype=np.int64)))
    validationloader_pseudo = torch.utils.data.DataLoader(validation_pseudo, batch_size=len(validation_pseudo),
                                                          shuffle=True, drop_last=False)
    train_pseudo = TensorDataset(torch.from_numpy(np.array(X_pseudo, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_pseudo, dtype=np.int64)))
    train_dataloader_pseudo = torch.utils.data.DataLoader(train_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    return train_dataloader_replay, validationloader_replay, train_dataloader_pseudo, validationloader_pseudo


def generate_dataloaders_for_SCADANN(dataloader_sessions, models, current_session, validation_set_ratio=0.2,
                                     batch_size=256, percentage_same_gesture_stable=0.65):
    examples_replay = []
    labels_replay = []

    examples_new_session = []
    labels_new_session = []
    for session_index in range(current_session + 1):
        print("HANDLING NEW SESSION")
        # This is the first session where we have real labels. Use them
        models[session_index].eval()
        if session_index == 0:
            # Create the training and validation dataset to train on
            for batch in dataloader_sessions[0]:
                with torch.no_grad():
                    inputs, labels_batch = batch
                    labels_replay.extend(labels_batch.numpy())
                    examples_replay.extend(inputs.numpy())
        # We don't have true labels for these sessions, generated them
        # elif session_index == current_session:
        else:
            examples_self_learning = []
            ground_truth_self_learning = []
            model_outputs_self_learning = []
            predicted_labels_self_learning = []
            # Get the predictions and model output for the pseudo label dataset
            models[session_index].eval()
            # model.apply(model.apply_dropout)
            for batch in dataloader_sessions[session_index]:
                with torch.no_grad():
                    inputs_batch, labels_batch = batch
                    inputs_batch = inputs_batch.cuda()

                    outputs = models[session_index](inputs_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    model_outputs_self_learning.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                    predicted_labels_self_learning.extend(predicted.cpu().numpy())
                    ground_truth_self_learning.extend(labels_batch.numpy())
                    examples_self_learning.extend(inputs_batch.cpu().numpy())
            # Segment the data in respect to the recorded gestures
            ground_truth_segmented, predictions_segmented, model_outputs_segmented, examples_segmented = \
                segment_dataset_by_gesture_to_remove_transitions(ground_truths=ground_truth_self_learning,
                                                                 predictions=predicted_labels_self_learning,
                                                                 model_outputs=model_outputs_self_learning,
                                                                 examples=examples_self_learning)

            pseudo_labels_segmented, indexes_pseudo_labels_segmented = pseudo_labels_heuristic_training_sessions(
                predictions_segmented, model_outputs_segmented, window_stable_mode_length=30,
                percentage_same_gesture_now_stable=percentage_same_gesture_stable,
                maximum_length_instability_gesture_transition
                =40,
                maximum_length_instability_same_gesture=40)

            pseudo_labels = []
            examples = []
            ground_truth_reduced = []
            for index_segment in range(len(pseudo_labels_segmented)):
                pseudo_labels.extend(pseudo_labels_segmented[index_segment])
                print("BEFORE: ", np.mean(np.array(ground_truth_segmented[index_segment]) ==
                                          np.array(predictions_segmented[index_segment])), "  AFTER: ", np.mean(
                    np.array(ground_truth_segmented[index_segment])[indexes_pseudo_labels_segmented[index_segment]] ==
                    np.array(pseudo_labels_segmented[index_segment])), " len before: ",
                      len(ground_truth_segmented[index_segment]), "  len after: ",
                      len(pseudo_labels_segmented[index_segment]))
                examples.extend(np.array(examples_segmented[index_segment])[
                                    indexes_pseudo_labels_segmented[index_segment]].tolist())
                ground_truth_reduced.extend(np.array(ground_truth_segmented[index_segment])[
                                                indexes_pseudo_labels_segmented[index_segment]].tolist())

            accuracy_before = np.mean(np.array(ground_truth_self_learning) ==
                                      np.array(predicted_labels_self_learning))
            accuracy_after = np.mean(np.array(ground_truth_reduced) ==
                                     np.array(pseudo_labels))
            print("ACCURACY MODEL: ", accuracy_before, "  Accuracy pseudo:", accuracy_after, " len pseudo: ",
                  len(pseudo_labels), "   len predictions", len(predicted_labels_self_learning))

            if session_index == current_session:
                examples_new_session.extend(examples)
                labels_new_session.extend(pseudo_labels)
            else:
                examples_replay.extend(examples)
                labels_replay.extend(pseudo_labels)

    # Create the dataloaders associated with the REPLAY data
    X_replay, X_valid_replay, Y_replay, Y_valid_replay = train_test_split(examples_replay, labels_replay,
                                                                          test_size=validation_set_ratio, shuffle=True)

    validation_replay = TensorDataset(torch.from_numpy(np.array(X_valid_replay, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_replay, dtype=np.int64)))
    validationloader_replay = torch.utils.data.DataLoader(validation_replay, batch_size=batch_size, shuffle=True,
                                                          drop_last=False)
    train_replay = TensorDataset(torch.from_numpy(np.array(X_replay, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_replay, dtype=np.int64)))
    train_dataloader_replay = torch.utils.data.DataLoader(train_replay, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    # Create the dataloaders associated with the NEW session
    X_pseudo, X_valid_pseudo, Y_pseudo, Y_valid_pseudo = train_test_split(examples_new_session, labels_new_session,
                                                                          test_size=validation_set_ratio, shuffle=True)

    validation_pseudo = TensorDataset(torch.from_numpy(np.array(X_valid_pseudo, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_pseudo, dtype=np.int64)))
    validationloader_pseudo = torch.utils.data.DataLoader(validation_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=False)
    train_pseudo = TensorDataset(torch.from_numpy(np.array(X_pseudo, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_pseudo, dtype=np.int64)))
    train_dataloader_pseudo = torch.utils.data.DataLoader(train_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    return train_dataloader_replay, validationloader_replay, train_dataloader_pseudo, validationloader_pseudo


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score

    ground_truths_no_retraining, predictions_no_retraining, output_network = np.load(
        "../../results/predictions_training_session_MultipleVote_11Gestures__2_CYCLES_no_retraining.npy",
        allow_pickle=True)
    index_participant, index_cycle = 6, 0
    # First dimensions is the participants, second dimension is the cycle
    predictions_MV = generate_pseudo_labels_MultipleVotes(
        predictions=predictions_no_retraining[index_participant][index_cycle],
        model_outputs=output_network[index_participant][index_cycle])

    print("Accuracy original: %f , accuracy MV: %f" % (
        accuracy_score(ground_truths_no_retraining[index_participant][index_cycle],
                       predictions_no_retraining[index_participant][index_cycle]),
        accuracy_score(ground_truths_no_retraining[index_participant][index_cycle], predictions_MV)))

    pseudo_labels, indexes_pseudo = pseudo_labels_heuristic(
        predictions_no_retraining[index_participant][index_cycle], output_network[index_participant][index_cycle],
        window_stable_mode_length=30,
        percentage_same_gesture_now_stable=.65,
        maximum_length_instability_gesture_transition=40,
        maximum_length_instability_same_gesture=40, use_look_back=True)

    examples = []
    ground_truth_reduced = []

    ground_truth_reduced.extend(np.array(ground_truths_no_retraining[index_participant][index_cycle])[
                                    indexes_pseudo].tolist())

    accuracy_before = np.mean(np.array(ground_truths_no_retraining[index_participant][index_cycle]) ==
                              np.array(predictions_no_retraining[index_participant][index_cycle]))
    accuracy_after = np.mean(np.array(ground_truth_reduced) ==
                             np.array(pseudo_labels))
    print("ACCURACY MODEL: ", accuracy_before, "  Accuracy pseudo:", accuracy_after, " len pseudo: ",
          len(pseudo_labels), "   len predictions", len(predictions_no_retraining[index_participant][index_cycle]))
