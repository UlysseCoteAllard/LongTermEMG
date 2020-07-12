import numpy as np
import pandas as pd
import seaborn as sns
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from statistics import mean, stdev
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix as confusion_matrix_function


# Skipping 30 examples is equivalent in skipping the first 1.5 seconds as the armband is cadenced at 1000Hz
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
        for participant_index_to_keep, participant_ground_truths, participant_predictions, participant_third_array in \
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


# Skipping 30 examples is equivalent in skipping the first 1.5 seconds as the armband is candenced at 1000Hz
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
        for participant_index_to_keep, participant_ground_truths, participant_predictions, participant_third_array in \
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
        confusion_matrix_calculated = confusion_matrix_calculated.astype('float') / \
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


def create_angles_classification_graph(predictions, ground_truths, angles, pitch_bins=5, yaw_bins=12):
    print(ground_truths)
    predictions, ground_truths, angles = remove_transition_period_evaluation(predictions=predictions,
                                                                             ground_truths=ground_truths,
                                                                             third_array=angles,
                                                                             number_of_examples_to_skip_after_change=30)
    accuracies_per_participants, min_pitch, max_pitch = get_accuracy_in_respect_to_angles(predictions, ground_truths,
                                                                                          angles, pitch_bins=pitch_bins,
                                                                                          yaw_bins=yaw_bins,
                                                                                          minimum_number_of_examples=
                                                                                          600)
    print(np.swapaxes(accuracies_per_participants, 1, 0))
    accuracies_per_participants = accuracies_per_participants[1:]
    sns.set()

    # Compute areas and colors
    sns.set()

    plt.rcParams.update({'font.size': 24})
    fig = plt.figure()
    ax = Axes3D(fig)
    ax = plt.subplot(111, polar=True)

    rad = np.linspace(min_pitch, max_pitch, pitch_bins - 1)
    a = np.linspace((-70. * np.pi) / 180., (70. * np.pi) / 180., yaw_bins)
    r, th = np.meshgrid(rad, a)
    print(np.shape(r), "   ", np.shape(th), "   ", np.shape(accuracies_per_participants))

    new_cmap = truncate_colormap(cmap=plt.get_cmap("magma"), minval=.2,
                                 maxval=0.8)

    # new_cmap = truncate_colormap(cmap=plt.get_cmap("magma"), minval=np.nanmin(accuracies_per_participants),
    #                             maxval=np.nanmax(accuracies_per_participants))

    p = ax.pcolormesh(th, r, np.swapaxes(accuracies_per_participants, 1, 0), cmap=new_cmap)
    show_values(p)
    ax.plot(a, r, ls='none', color='k')
    ax.grid()

    ax.set_rorigin(-125.)
    ax.set_theta_zero_location('W', offset=270)
    ax.set_thetamin(-70)
    ax.set_thetamax(70)

    plt.savefig("angles_classification.svg", transparent=True)

    plt.show()
    plt.rcParams.update({'font.size': 48})
    # create dummy invisible image
    # (use the colormap you want to have on the colorbar)
    img = plt.imshow(np.array([[np.nanmin(accuracies_per_participants), np.nanmax(accuracies_per_participants)]]),
                     cmap=new_cmap)
    img.set_visible(False)

    plt.colorbar(orientation="vertical")
    plt.show()


def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        if value != float("nan"):
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def create_activation_classification_graph(predictions, ground_truths, activations_examples, number_of_bins=10):
    predictions, ground_truths, activations_examples = remove_transition_period_evaluation(predictions=predictions,
                                                                                           ground_truths=ground_truths,
                                                                                           third_array=
                                                                                           activations_examples)
    bins_per_participants = []
    accuracies_per_participants = []
    amount_per_participants_per_bins = []
    participants = []
    for index_participant, (participant_ground_truths, participant_predictions, participant_activation_examples) in \
            enumerate(zip(ground_truths, predictions, activations_examples)):
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
        bins = np.linspace(0, 1., number_of_bins + 1)
        bins = bins[0:number_of_bins]
        for activation, prediction_alignement in zip(activation_percentage, prediction_and_ground_truth_align):
            for index, bin_for_histogram in reversed(list(enumerate(bins))):
                if bin_for_histogram < activation:
                    good_prediction_per_bin[index] = np.append(good_prediction_per_bin[index], prediction_alignement)
                    break

        # Get accuracy per bin
        accuracies = []
        amount_per_bin = []
        for bin_prediction_and_ground_truth in good_prediction_per_bin:
            print(np.shape(bin_prediction_and_ground_truth))
            amount_per_bin.append(len(bin_prediction_and_ground_truth))
            accuracies.append(np.mean(bin_prediction_and_ground_truth))
        bins_per_participants.extend(bins)
        accuracies_per_participants.extend(accuracies)
        amount_per_participants_per_bins.append(amount_per_bin)
        print(accuracies)
        participants.extend(index_participant * np.ones(number_of_bins))

    print(np.shape(amount_per_participants_per_bins))
    print("NUMBER OF EXAMPLES PER BIN: ", np.sum(amount_per_participants_per_bins, axis=0))

    sns.set(font_scale=3.5, style='whitegrid')
    df = pd.DataFrame({"Accuracy": accuracies_per_participants,
                       "Percentage of Maximum Activation": bins_per_participants, "Participant": participants})

    g = sns.barplot(x="Percentage of Maximum Activation", y="Accuracy", data=df, errwidth=7)
    xlabels = []
    for i in range(len(g.get_xticklabels())):
        if i < len(g.get_xticklabels()) - 1:
            xlabels.append('[{:,.2f}, {:,.2f}['.format(float(g.get_xticklabels()[i].get_text()),
                                                       float(g.get_xticklabels()[i + 1].get_text())))
        else:
            xlabels.append('[{:,.2f}, 1.00]'.format(float(g.get_xticklabels()[i].get_text())))
    g.set_xticklabels(xlabels)
    sns.despine()
    plt.show()


def long_term_pointplot(ground_truths_in_array, predictions_in_array, text_for_legend_in_array, timestamps,
                        number_of_seances_to_consider, remove_transition_evaluation=False,
                        only_every_second_session=False):
    time_distance_participants_in_hours = []
    for participant_times in timestamps:
        time_distance = []
        for i in range(0, len(participant_times)):
            if only_every_second_session and i % 2 != 0:
                time_distance.append((participant_times[i] - participant_times[1]) / timedelta(days=1))
            elif only_every_second_session is False:
                time_distance.append((participant_times[i] - participant_times[0]) / timedelta(days=1))
        time_distance_participants_in_hours.append(time_distance)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    accuracies = []
    algorithm_index_array = []
    time_hours = []
    participant_indexes = []
    seance_index_array = []

    for index, (ground_truths, predictions) in enumerate(zip(ground_truths_in_array, predictions_in_array)):
        if remove_transition_evaluation:
            predictions, ground_truths = remove_transition_period_evaluation(predictions=predictions,
                                                                             ground_truths=ground_truths)
        participant_index = 0
        for participant_ground_truths, participant_predictions, participant_timestamps_in_hours in zip(
                ground_truths, predictions, time_distance_participants_in_hours):
            seance_index = 0
            for ground_truths_seance, predictions_seance, time_distance_seance in zip(
                    participant_ground_truths, participant_predictions, participant_timestamps_in_hours):
                if seance_index > number_of_seances_to_consider:
                    break

                print("TIME: ", time_distance_seance)
                print(ground_truths_seance)
                print(predictions_seance)
                print("ACCURACY: ", np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)

                time_hours.append(time_distance_seance)
                accuracies.append(np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
                participant_indexes.append(participant_index)
                algorithm_index_array.append(index)
                seance_index_array.append(seance_index)
                seance_index += 1
            participant_index += 1

    df = pd.DataFrame({"Time (days)": time_hours, "Accuracy": accuracies, "Participant": participant_indexes,
                       "Algorithm": algorithm_index_array, "Seance": seance_index_array})
    average_time_session = []
    for seance_j in range(0, number_of_seances_to_consider + 1):
        time_session_j = df['Time (days)'][(df['Algorithm'] == 0) & (df['Seance'] == seance_j)]
        average_time_session.append(np.mean(time_session_j))
    print("Average time seance: ", average_time_session)
    average_time_session = [0.0, 6.7, 13.2, 18.6]
    for seance_index in range(len(seance_index_array)):
        seance_index_array[seance_index] = np.round(average_time_session[seance_index_array[seance_index]], decimals=1)
    df = pd.DataFrame({"Time (days)": seance_index_array, "Accuracy": accuracies, "Participant": participant_indexes,
                       "Algorithm": algorithm_index_array})

    sns.set(font_scale=5, style="whitegrid")
    sns.set_context(font_scale=5, rc={"lines.linewidth": 4})
    print(df["Algorithm"])
    linestyles = ['-', '--', '-.', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1))]
    markers = ['o', 's', "^", "*", "h", "p", "X"]
    # g = sns.pointplot(x="Time (days)", y="Accuracy", data=df, hue="Algorithm", s=700, linestyles=linestyles,
    #                  palette=cycle[0:len(ground_truths_in_array)], markers=markers[0:len(ground_truths_in_array)],
    #                  dodge=True, capsize=.05, legend_out=True, ci="sd")
    '''
    g = sns.barplot(x="Time (days)", y="Accuracy", data=df, hue="Algorithm",
                      palette=cycle[0:len(ground_truths_in_array)],
                      dodge=True, capsize=.05, ci="sd")
    '''
    cycle = [cycle[i] for i in [0, 2, 3]]
    g = sns.barplot(x="Time (days)", y="Accuracy", data=df, hue="Algorithm",
                    palette=cycle,
                    dodge=True, capsize=.05, ci="sd")

    # Put the legend out of the figure
    # plt.legend(bbox_to_anchor=(.9, 1), loc=2, borderaxespad=0.)
    plt.setp(g.axes.get_legend().get_title(), fontsize='56')  # for legend title

    leg = g.axes.get_legend()
    text_legend_array = []
    for i, text_legend in enumerate(text_for_legend_in_array):
        text_legend_array.append(text_legend)
    for t, l in zip(leg.texts, text_legend_array):
        t.set_text(l)

    sns.despine(left=True)

    plt.show()


def accuracy_vs_score_lmplot(ground_truths_in_array, predictions_in_array, scores_in_array,
                             number_of_seances_to_consider):
    accuracies = []
    scores = []
    participant_index = 0
    for participant_ground_truths, participant_predictions, participant_scores in zip(
            ground_truths_in_array, predictions_in_array, scores_in_array):
        seance_index = 0
        for ground_truths_seance, predictions_seance, score_seance in zip(
                participant_ground_truths, participant_predictions, participant_scores):
            if seance_index > number_of_seances_to_consider:
                break
            scores.append(score_seance)
            accuracies.append(np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
            seance_index += 1
        participant_index += 1
    df = pd.DataFrame({"Score": scores, "Accuracy": accuracies})
    sns.set(font_scale=3, style="whitegrid")
    sns.lmplot(y="Score", x="Accuracy", data=df, x_ci="sd", line_kws={'color': 'black', "linewidth": 12},
               scatter_kws={"s": 700, 'color': 'black'})

    print(df['Score'].corr(df['Accuracy']))

    sns.despine(left=True)
    plt.show()


def long_term_classification_graph(ground_truths_in_array, predictions_in_array, text_for_legend_in_array, timestamps,
                                   number_of_seances_to_consider, remove_transition_evaluation=False,
                                   only_every_second_session=False):
    time_distance_participants_in_hours = []
    for participant_times in timestamps:
        time_distance = []
        for i in range(0, len(participant_times)):
            if only_every_second_session and i % 2 != 0:
                time_distance.append((participant_times[i] - participant_times[1]) / timedelta(days=1))
            elif only_every_second_session is False:
                time_distance.append((participant_times[i] - participant_times[0]) / timedelta(days=1))
        time_distance_participants_in_hours.append(time_distance)

    accuracies = []
    algorithm_index_array = []
    time_hours = []
    participant_indexes = []
    seance_index_array = []
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, (ground_truths, predictions) in enumerate(zip(ground_truths_in_array, predictions_in_array)):
        if remove_transition_evaluation:
            predictions, ground_truths = remove_transition_period_evaluation(predictions=predictions,
                                                                             ground_truths=ground_truths)
        participant_index = 0
        for participant_ground_truths, participant_predictions, participant_timestamps_in_hours in zip(
                ground_truths, predictions, time_distance_participants_in_hours):
            seance_index = 0
            for ground_truths_seance, predictions_seance, time_distance_seance in zip(
                    participant_ground_truths, participant_predictions, participant_timestamps_in_hours):
                if seance_index > number_of_seances_to_consider:
                    break
                '''
                print("TIME: ", time_distance_seance)
                print(ground_truths_seance)
                print(predictions_seance)
                print("ACCURACY: ", np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
                '''
                time_hours.append(time_distance_seance)
                accuracies.append(np.mean(np.array(ground_truths_seance) == np.array(predictions_seance)) * 100.)
                participant_indexes.append(participant_index)
                algorithm_index_array.append(index)
                seance_index_array.append(seance_index)
                seance_index += 1
            participant_index += 1

    df = pd.DataFrame({"Time (days)": time_hours, "Accuracy": accuracies, "Participant": participant_indexes,
                       "algorithm_index": algorithm_index_array, "Seance": seance_index_array})
    accuracy_mean_first_session = np.mean(df['Accuracy'][df['Time (days)'] == 0.0])
    print("MEAN FIRST SESSION: ", accuracy_mean_first_session)

    sns.set(font_scale=3, style="whitegrid")
    linestyles = ['-', '--', '-.', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1))]
    markers = ['o', 'd', "^", "*", "h", "p", "X", "P"]
    sns.scatterplot(x="Time (days)", y="Accuracy", data=df, hue="algorithm_index", style="algorithm_index",
                    legend=False, s=700, palette=cycle[0:len(ground_truths_in_array)],
                    markers=markers[0:len(ground_truths_in_array)])

    text_legend_array = []
    for i, text_legend in enumerate(text_for_legend_in_array):
        time_hours = df['Time (days)'][df['algorithm_index'] == i]
        accuracies = df['Accuracy'][df['algorithm_index'] == i]

        for seance_i in range(number_of_seances_to_consider + 1):
            get_statistics(df['Accuracy'][(df['algorithm_index'] == i) & (df['Seance'] == seance_i)], seance_i,
                           algorithm_name=text_legend)

        lin_regression = generate_linear_regression(X=time_hours, y=accuracies,
                                                    mean_accuracy_first_session=accuracy_mean_first_session,
                                                    color=cycle[i], linestyle=linestyles[i])
        text_legend_array.append(text_legend + ", Regression slope: %.4f" % lin_regression[1])
    '''
    for algorithm_1 in range(len(text_for_legend_in_array)):
        for algorithm_2 in range(algorithm_1+1, len(text_for_legend_in_array)):
            if algorithm_1 != algorithm_2:
                samples_d1 = []
                samples_d2 = []
                for seance_j in range(0, number_of_seances_to_consider + 1):
                    if seance_j % 2 != 0:
                        cohen_D = get_cohen_D(np.array(samples_d1), np.array(samples_d2))
                        print(text_for_legend_in_array[algorithm_1], " VS ",
                              text_for_legend_in_array[algorithm_2], "  COHEN D: ", cohen_D)
                    else:
                        samples_d1.extend(
                            df['Accuracy'][(df['algorithm_index'] == algorithm_1) & (df['Seance'] == seance_j)])
                        samples_d2.extend(
                            df['Accuracy'][(df['algorithm_index'] == algorithm_2) & (df['Seance'] == seance_j)])
    '''
    for algorithm_1 in range(len(text_for_legend_in_array)):
        for algorithm_2 in range(algorithm_1 + 1, len(text_for_legend_in_array)):
            if algorithm_1 != algorithm_2:
                for seance_j in range(0, number_of_seances_to_consider + 1):
                    samples_d1 = []
                    samples_d2 = []
                    samples_d1.extend(
                        df['Accuracy'][(df['algorithm_index'] == algorithm_1) & (df['Seance'] == seance_j)])
                    samples_d2.extend(
                        df['Accuracy'][(df['algorithm_index'] == algorithm_2) & (df['Seance'] == seance_j)])
                    cohen_D = get_cohen_Dz(np.array(samples_d1), np.array(samples_d2))
                    print("Seance: ", seance_j, ",  ", text_for_legend_in_array[algorithm_1], " VS ",
                          text_for_legend_in_array[algorithm_2], "  COHEN Dz: ", cohen_D)

    sns.despine(left=True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), labels=text_legend_array)
    plt.show()


def get_statistics(accuracies, seance, algorithm_name):
    print("ACCURACIES: ", accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print(algorithm_name + "  , seance: " + str(seance), "  Mean: " + str(mean), "  STD: " + str(std))


def get_cohen_Dz(samples_d1, samples_d2):
    # Pooled standard deviation
    difference = np.mean(samples_d1) - np.mean(samples_d2)
    cohen_Dz = difference / np.std(samples_d1 - samples_d2, ddof=1)
    return cohen_Dz


def generate_linear_regression(X, y, mean_accuracy_first_session, color, linestyle):
    linear_regression = polyfit_with_fixed_points(n=1, x=np.array(X),
                                                  y=np.array(y),
                                                  xf=np.array([0.0]),
                                                  yf=np.array([mean_accuracy_first_session]))
    x_linear_regression = np.linspace(0., np.max(X) + 5., len(X))
    y_regression = linear_regression[1] * x_linear_regression + linear_regression[0]

    lower, upper = bootstrap_confidence_interval(X=np.array(X),
                                                 y=np.array(y), first_value_x=np.array([0.0]),
                                                 first_value_y=np.array([mean_accuracy_first_session]))

    plt.fill_between(x_linear_regression, lower, upper, alpha=.25, color=color)

    plt.plot(x_linear_regression, y_regression, linewidth=12., color=color, linestyle=linestyle)

    return linear_regression


def bootstrap_confidence_interval(X, y, first_value_x, first_value_y, n_iter=100):
    res = 0
    intv_index = np.array(list(range(len(X))))
    y_b = np.zeros((n_iter, len(intv_index)), dtype=float)
    data_to_evaluate = np.linspace(np.min(X), np.max(X), len(X))
    for i in range(n_iter):
        # Bootstrap: resample data with replacement
        sample_index = np.random.choice(range(0, len(y)), int(len(y) / 2))
        X_samples = X[sample_index]
        y_samples = y[sample_index]

        lin_model = polyfit_with_fixed_points(n=1, x=X_samples, y=y_samples, xf=first_value_x, yf=first_value_y)
        y_hat = lin_model[1] * data_to_evaluate + lin_model[0]
        res += y - y_hat
        y_b[i, :] = y_hat[intv_index]
        # plt.plot(X, y_hat, color='r', alpha=0.2, zorder=1)
    y_conf_min = np.percentile(y_b, 2.5, axis=0)
    y_conf_max = np.percentile(y_b, 97.5, axis=0)
    return y_conf_min, y_conf_max


def polyfit_with_fixed_points(n, x, y, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x ** np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf ** np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]
