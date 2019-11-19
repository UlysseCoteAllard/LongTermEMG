import numpy as np
import pandas as pd
import seaborn as sns
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix as confusion_matrix_function


# Skipping 30 examples is equivalent in skipping the first 1 seconds as the armband is candenced at 1000Hz
def remove_transition_period_evaluation(ground_truths, predictions, third_array=None,
                                        number_of_examples_to_skip_after_change=30):
    number_of_examples_skipped = 0
    stop_adding = False
    index_without_transition = []
    for participant_ground_truth in ground_truths:
        participant_index_without_transition = []
        for session_ground_truth in participant_ground_truth:
            session_index_without_transition = []
            last_label = session_ground_truth[0]
            print(session_ground_truth)
            for index_ground_truth, ground_truth in enumerate(session_ground_truth):
                if stop_adding is False:
                    if ground_truth == last_label:
                        session_index_without_transition.append(index_ground_truth)
                        number_of_examples_skipped += 1
                    else:
                        stop_adding = True
                        last_label = ground_truth
                        print(number_of_examples_skipped)
                        number_of_examples_skipped = 0
                else:
                    number_of_examples_skipped += 1
                    if number_of_examples_skipped >= number_of_examples_to_skip_after_change:
                        stop_adding = False
                        last_label = ground_truth

            stop_adding = False
            participant_index_without_transition.append(session_index_without_transition)
        stop_adding = False
        index_without_transition.append(participant_index_without_transition)


    if third_array is None:
        ground_truth_without_transition = []
        predictions_without_transition = []
        for participant_index_to_keep, participant_ground_truths, participant_predictions in zip(
                index_without_transition, ground_truths, predictions):
            participant_ground_truth_without_transition = []
            participant_predictions_without_transition = []
            for session_index_to_keep, session_ground_truths, session_predictions in zip(
                    participant_index_to_keep, participant_ground_truths, participant_predictions):
                participant_ground_truth_without_transition.append(np.array(session_ground_truths)[
                                                                       session_index_to_keep])
                participant_predictions_without_transition.append(np.array(session_predictions)[session_index_to_keep])
            ground_truth_without_transition.append(participant_ground_truth_without_transition)
            predictions_without_transition.append(participant_predictions_without_transition)
        return ground_truth_without_transition, predictions_without_transition
    else:
        ground_truth_without_transition = []
        predictions_without_transition = []
        third_array_without_transition = []
        for participant_index_to_keep, participant_ground_truths, participant_predictions, participant_third_array in\
                zip(index_without_transition, ground_truths, predictions, third_array):
            participant_ground_truth_without_transition = []
            participant_predictions_without_transition = []
            participant_third_array_without_transition = []
            for session_index_to_keep, session_ground_truths, session_predictions, session_third_array in zip(
                    participant_index_to_keep, participant_ground_truths, participant_predictions,
                    participant_third_array):
                participant_ground_truth_without_transition.append(np.array(session_ground_truths)[
                                                                       session_index_to_keep])
                participant_predictions_without_transition.append(np.array(session_predictions)[session_index_to_keep])
                participant_third_array_without_transition.append(np.array(session_third_array)[session_index_to_keep])

            ground_truth_without_transition.append(participant_ground_truth_without_transition)
            predictions_without_transition.append(participant_predictions_without_transition)
            third_array_without_transition.append(participant_third_array_without_transition)
        return ground_truth_without_transition, predictions_without_transition, third_array_without_transition


# Skipping 20 examples is equivalent in skipping the first 1 seconds as the armband is candenced at 1000Hz
def remove_transition_period_training(ground_truths, predictions, third_array=None,
                                      number_of_examples_to_skip_after_change=30):
    number_of_examples_skipped = 0
    stop_adding = False
    index_without_transition = []
    for participant_ground_truth in ground_truths:
        participant_index_without_transition = []
        last_label = participant_ground_truth[0]

        print(participant_ground_truth)
        for index_ground_truth, ground_truth in enumerate(participant_ground_truth):
            if stop_adding is False:
                if ground_truth == last_label:
                    participant_index_without_transition.append(index_ground_truth)
                    number_of_examples_skipped += 1
                else:
                    stop_adding = True
                    last_label = ground_truth
                    print(number_of_examples_skipped)
                    number_of_examples_skipped = 0
            else:
                number_of_examples_skipped += 1
                if number_of_examples_skipped >= number_of_examples_to_skip_after_change:
                    stop_adding = False
                    last_label = ground_truth

        stop_adding = False
        index_without_transition.append(participant_index_without_transition)
    if third_array is None:
        ground_truth_without_transition = []
        predictions_without_transition = []
        for participant_index_to_keep, participant_ground_truths, participant_predictions in zip(
                index_without_transition, ground_truths, predictions):
            ground_truth_without_transition.append(np.array(participant_ground_truths)[participant_index_to_keep])
            predictions_without_transition.append(np.array(participant_predictions)[participant_index_to_keep])
        return ground_truth_without_transition, predictions_without_transition
    else:
        ground_truth_without_transition = []
        predictions_without_transition = []
        third_array_without_transition = []
        for participant_index_to_keep, participant_ground_truths, participant_predictions, participant_third_array in\
                zip(index_without_transition, ground_truths, predictions, third_array):
            ground_truth_without_transition.append(np.array(participant_ground_truths)[participant_index_to_keep])
            predictions_without_transition.append(np.array(participant_predictions)[participant_index_to_keep])
            third_array_without_transition.append(np.array(participant_third_array)[participant_index_to_keep])
        return ground_truth_without_transition, predictions_without_transition, third_array_without_transition


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def create_confusion_matrix(ground_truth, predictions, class_names, fontsize=24,
                            normalize=True, fig=None, axs=None, title=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """

    print(np.shape(predictions[0]))
    # Calculate the confusion matrix across all participants
    predictions = [x for y in predictions for x in y]
    ground_truth = [x for y in ground_truth for x in y]
    predictions = [x for y in predictions for x in y]
    ground_truth = [x for y in ground_truth for x in y]
    print(np.shape(ground_truth))
    confusion_matrix_calculated = confusion_matrix_function(np.ravel(np.array(ground_truth)),
                                                            np.ravel(np.array(predictions)))

    if normalize:
        confusion_matrix_calculated = confusion_matrix_calculated.astype('float') /\
                                      confusion_matrix_calculated.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        print("Normalized confusion matrix")
    else:
        fmt = 'd'
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix_calculated, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cbar=False, annot_kws={"size": 28})
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.set(
        # ... and label them with the respective list entries
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=fontsize)
    heatmap.xaxis.label.set_size(fontsize + 4)
    heatmap.yaxis.label.set_size(fontsize + 4)
    heatmap.title.set_size(fontsize + 6)
    return fig, axs


def get_accuracy_in_respect_to_angles(predictions, ground_truth, angles, pitch_bins=6, yaw_bins=12,
                                      minimum_number_of_examples=1000):
    bins_pitch_per_participants = []
    bins_yaw_per_participants = []
    accuracies_per_participants = np.zeros((pitch_bins, yaw_bins, 0)).tolist()
    participants = []

    for index_participant, (participant_ground_truths, participant_predictions, participants_angles) in \
            enumerate(zip(predictions, ground_truth, angles)):
        prediction_and_ground_truth_align = []
        pitch_angles = []
        yaw_angles = []
        for ground_truths_seance, predictions_seance, angles_seances in \
                zip(participant_ground_truths, participant_predictions, participants_angles):
            prediction_and_ground_truth_align.extend(np.array(ground_truths_seance) == np.array(predictions_seance))
            for example_angles in angles_seances:
                pitch_angles.append(example_angles['pitch'])
                yaw_angles.append(example_angles['yaw'])

        if index_participant == 0:
            min_pitch, max_pitch = np.nanmin(pitch_angles), np.nanmax(pitch_angles)
            print(min_pitch, "   ", max_pitch)

        good_prediction_per_bin_pitch_and_yaw = np.zeros((pitch_bins, yaw_bins, 0)).tolist()
        bins_pitch = np.linspace(min_pitch, max_pitch, pitch_bins + 1)
        bins_pitch = bins_pitch[0:pitch_bins]
        bins_yaw = np.linspace(-70., 70., yaw_bins + 1)
        bins_yaw = bins_yaw[0:yaw_bins]

        for angle_pitch, angle_yaw, prediction_alignement in zip(pitch_angles, yaw_angles,
                                                                 prediction_and_ground_truth_align):
            right_bin_found = False
            for index_pitch, bin_for_histogram_pitch in reversed(list(enumerate(bins_pitch))):
                for index_yaw, bin_for_histogram_yaw in reversed(list(enumerate(bins_yaw))):
                    if angle_yaw > bin_for_histogram_yaw:
                        if angle_pitch > bin_for_histogram_pitch:
                            good_prediction_per_bin_pitch_and_yaw[index_pitch][index_yaw] = np.append(
                                good_prediction_per_bin_pitch_and_yaw[index_pitch][index_yaw], prediction_alignement)
                            right_bin_found = True
                    if right_bin_found:
                        break
                if right_bin_found:
                    break

        # Get accuracy per bin
        for x in range(len(good_prediction_per_bin_pitch_and_yaw)):
            for y in range(len(good_prediction_per_bin_pitch_and_yaw[x])):
                print(np.shape(accuracies_per_participants[x][y]), "   ",
                      np.shape(good_prediction_per_bin_pitch_and_yaw[x][y]))
                accuracies_per_participants[x][y] = np.concatenate((accuracies_per_participants[x][y],
                                                                    good_prediction_per_bin_pitch_and_yaw[x][y]))

    print(np.shape(accuracies_per_participants))
    for x in range(len(accuracies_per_participants)):
        for y in range(len(accuracies_per_participants[x])):
            if len(accuracies_per_participants[x][y]) < minimum_number_of_examples:
                accuracies_per_participants[x][y] = float('nan')
            else:
                accuracies_per_participants[x][y] = np.mean(accuracies_per_participants[x][y])
    return accuracies_per_participants, bins_pitch[1], max_pitch


def create_angles_classification_graph(predictions, ground_truth, angles, pitch_bins=6, yaw_bins=12):
    print(ground_truth)
    predictions, ground_truth, angles = remove_transition_period_evaluation(predictions=predictions,
                                                                            ground_truths=ground_truth,
                                                                            third_array=angles)
    accuracies_per_participants, min_pitch, max_pitch = get_accuracy_in_respect_to_angles(predictions, ground_truth,
                                                                                          angles, pitch_bins=pitch_bins,
                                                                                          yaw_bins=yaw_bins)
    print(np.swapaxes(accuracies_per_participants, 1, 0))
    accuracies_per_participants = accuracies_per_participants[1:]
    sns.set()

    # Compute areas and colors
    sns.set()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax = plt.subplot(111, polar=True)

    rad = np.linspace(min_pitch, max_pitch, pitch_bins-1)
    a = np.linspace((-70.*np.pi)/180., (70.*np.pi)/180., yaw_bins)
    r, th = np.meshgrid(rad, a)
    print(np.shape(r), "   ", np.shape(th), "   ", np.shape(accuracies_per_participants))

    new_cmap = truncate_colormap(cmap=plt.get_cmap("magma"), minval=np.nanmin(accuracies_per_participants),
                                 maxval=np.nanmax(accuracies_per_participants))

    ax.pcolormesh(th, r, np.swapaxes(accuracies_per_participants, 1, 0), cmap=new_cmap)
    ax.plot(a, r, ls='none', color='k')
    ax.grid()

    ax.set_rorigin(-125.)
    ax.set_theta_zero_location('W', offset=270)
    ax.set_thetamin(-70)
    ax.set_thetamax(70)

    plt.show()


def create_activation_classification_graph(predictions, ground_truth, activations_examples, number_of_bins=10):
    predictions, ground_truth, activations_examples = remove_transition_period_evaluation(predictions=predictions,
                                                                                          ground_truths=ground_truth,
                                                                                          third_array=
                                                                                          activations_examples)
    bins_per_participants = []
    accuracies_per_participants = []
    participants = []
    for index_participant, (participant_ground_truths, participant_predictions, participant_activation_examples) in \
            enumerate(zip(ground_truth, predictions, activations_examples)):
        prediction_and_ground_truth_align = []
        activation_percentage = []
        for ground_truths_seance, predictions_seance, activation_seances in zip(
                participant_ground_truths, participant_predictions, participant_activation_examples):
            # Remove the neutral gesture as it does not make sense for contraction intensity
            index_without_neutral = np.squeeze(np.argwhere(ground_truths_seance))
            prediction_and_ground_truth_align.extend(np.array(ground_truths_seance)[index_without_neutral] ==
                                                     np.array(predictions_seance)[index_without_neutral])
            # Add the activation percentage
            activation_percentage.extend(np.array(activation_seances)[index_without_neutral])

        good_prediction_per_bin = np.zeros((number_of_bins, 0)).tolist()
        bins = np.linspace(0, 1., number_of_bins+1)
        bins = bins[0:number_of_bins]
        for activation, prediction_alignement in zip(activation_percentage, prediction_and_ground_truth_align):
            for index, bin_for_histogram in reversed(list(enumerate(bins))):
                if activation > bin_for_histogram:
                    good_prediction_per_bin[index] = np.append(good_prediction_per_bin[index], prediction_alignement)
                    break

        # Get accuracy per bin
        accuracies = []
        for bin_prediction_and_ground_truth in good_prediction_per_bin:
            print(np.shape(bin_prediction_and_ground_truth))
            accuracies.append(np.mean(bin_prediction_and_ground_truth))
        bins_per_participants.extend(bins)
        accuracies_per_participants.extend(accuracies)
        participants.extend(index_participant*np.ones(number_of_bins))
    sns.set(font_scale=3.5)
    df = pd.DataFrame({"Accuracy": accuracies_per_participants, "Activation Percentage of Max": bins_per_participants,
                       "Participant": participants})
    print(df)
    sns.barplot(x="Activation Percentage of Max", y="Accuracy", data=df)

    plt.show()


def create_long_term_classification_graph(ground_truths_no_retraining,
                                          predictions_no_retraining,
                                          ground_truths_WITH_retraining,
                                          predictions_WITH_retraining, timestamps, number_of_seances_to_consider=2):
    print(timestamps)
    print(np.shape(timestamps[0]))
    time_distance_participants_in_hours = []
    for participant_times in timestamps:
        time_distance = []
        for i in range(0, len(participant_times)):
            time_distance.append((participant_times[i] - participant_times[0])/timedelta(days=1))
        time_distance_participants_in_hours.append(time_distance)
    print(time_distance_participants_in_hours)
    """
    print(np.shape(ground_truths))
    print(np.shape(ground_truths[0]))
    print(np.shape(ground_truths[0][0]))
    """
    accuracies = []
    recalibration_or_not = []
    time_hours = []
    participant_indexes = []
    participant_index = 0
    for participant_ground_truths, participant_predictions, participant_timestamps_in_hours in zip(
            ground_truths_no_retraining, predictions_no_retraining, time_distance_participants_in_hours):
        seance_index = 0
        for ground_truths_seance, predictions_seance, time_distance_seance in zip(
                participant_ground_truths, participant_predictions, participant_timestamps_in_hours):
            if seance_index > number_of_seances_to_consider:
                break
            print("TIME: ", time_distance_seance)
            print("ACCURACY: ", np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
            time_hours.append(time_distance_seance)
            accuracies.append(np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
            participant_indexes.append(participant_index)
            recalibration_or_not.append("No")
            seance_index += 1
        participant_index += 1

    participant_index = 0
    for participant_ground_truths, participant_predictions, participant_timestamps_in_hours in zip(
            ground_truths_WITH_retraining, predictions_WITH_retraining, time_distance_participants_in_hours):
        seance_index = 0
        for ground_truths_seance, predictions_seance, time_distance_seance in zip(
                participant_ground_truths, participant_predictions, participant_timestamps_in_hours):
            if seance_index > number_of_seances_to_consider:
                break
            print("TIME: ", time_distance_seance, " SEANCE: ", seance_index)
            print("ACCURACY: ", np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
            time_hours.append(time_distance_seance)
            accuracies.append(np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
            participant_indexes.append(participant_index)
            recalibration_or_not.append("Yes")
            seance_index += 1
        participant_index += 1

    df = pd.DataFrame({"Time (days)": time_hours, "Accuracy": accuracies, "Participant": participant_indexes,
                       "Recalibration": recalibration_or_not})

    accuracy_mean_first_session = np.mean(df['Accuracy'][df['Time (days)'] == 0.0])
    print("MEAN FIRST SESSION: ", accuracy_mean_first_session)
    #df['Accuracy'][df['Time (days)'] == 0.0] = accuracy_mean_first_session

    print(df)

    print(np.polyfit(time_hours, accuracies, deg=1))
    time_hours_no_retraining = df['Time (days)'][df['Recalibration'] == 'No']
    time_hours_retraining = df['Time (days)'][df['Recalibration'] == 'Yes']
    accuracies_no_retraining = df['Accuracy'][df['Recalibration'] == 'No']
    accuracies_hours_retraining = df['Accuracy'][df['Recalibration'] == 'Yes']
    import numpy.polynomial.polynomial as poly
    linear_regression_no_retraining = polyfit_with_fixed_points(n=1, x=np.array(time_hours_no_retraining),
                                                                y=np.array(accuracies_no_retraining),
                                                                xf=np.array([0.0]),
                                                                yf=np.array([accuracy_mean_first_session]))
    linear_regression_retraining = polyfit_with_fixed_points(n=1, x=np.array(time_hours_retraining),
                                                             y=np.array(accuracies_hours_retraining),
                                                             xf=np.array([0.0]),
                                                             yf=np.array([accuracy_mean_first_session]))

    x_linear_regression = np.linspace(0., np.max(df['Time (days)']), 10)
    y_regression_no_retraining = linear_regression_no_retraining[1]*x_linear_regression + \
                                 linear_regression_no_retraining[0]

    y_regression_retraining = linear_regression_retraining[1]*x_linear_regression + \
                                 linear_regression_retraining[0]
    sns.set(font_scale=3.5)
    sns.scatterplot(x="Time (days)", y="Accuracy", data=df, hue="Recalibration", style="Recalibration", legend=False,
                    s=100)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sns.lineplot(x=x_linear_regression, y=y_regression_no_retraining, linewidth=3.5, color=cycle[0])
    ax = sns.lineplot(x=x_linear_regression, y=y_regression_retraining, linewidth=3.5, color=cycle[1])

    plt.legend(handles=[plt.plot([], ls="-", linewidth=3.5, color=cycle[0])[0], plt.plot([], ls="-", linewidth=3.5, color=cycle[1])[0]],
               labels=["No retraining, y = %fx + %f" % (linear_regression_no_retraining[1],
                                                        linear_regression_no_retraining[0]),
                       "y = %fx + %f" % (linear_regression_retraining[1], linear_regression_retraining[0])],
               loc="best")
    plt.show()

def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]