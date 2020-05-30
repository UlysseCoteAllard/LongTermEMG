import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta


def get_scores(path="../../../datasets/longterm_dataset_3DC"):
    scores = []
    for index_participant in range(22):
        # Those two participant did not complete the experiment
        if index_participant != 10 and index_participant != 11:
            folder_participant = "/Participant" + str(index_participant)
            sessions_directories = os.listdir(path + folder_participant)
            score_participant = []
            for session_directory in sessions_directories:
                if "Evaluation" in session_directory:
                    path_session = path + folder_participant + "/" + session_directory
                    if len(os.listdir(path_session)) > 0:
                        with open(path_session + "/score.txt") as score_file:
                            score = score_file.readline()
                            score_participant.append(float(score.strip()))
            scores.append(np.array(score_participant))
    return np.array(scores)


def get_statistics(accuracies, seance, ):
    print("ACCURACIES: ", accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print("seance: " + str(seance), "  Mean: " + str(mean), "  STD: " + str(std))


def graph_scores(scores, timestamps, number_of_seances_to_consider=8):
    time_distance_participants_in_hours = []
    for participant_times in timestamps:
        time_distance = []
        for i in range(0, len(participant_times)):
            time_distance.append((participant_times[i] - participant_times[0]) / timedelta(days=1))
        time_distance_participants_in_hours.append(time_distance)

    scores_for_pandas, time_hours, participant_index_array, seances_index_array = [], [], [], []
    for participant_index, (scores_participant, participant_timestamps_in_hours) in enumerate(
            zip(scores, time_distance_participants_in_hours)):
        seance_index = 0
        for score_session, time_distance_seance in zip(scores_participant, participant_timestamps_in_hours):
            if seance_index > number_of_seances_to_consider:
                break
            # If it's lower, there was a technical problem during the leap motion recording
            if score_session > 1500:
                print("TIME: ", time_distance_seance)
                time_hours.append(time_distance_seance)
                scores_for_pandas.append(score_session)
                seances_index_array.append(seance_index)
                participant_index_array.append(participant_index)
                seance_index += 1

    df = pd.DataFrame({"Time (days)": time_hours, "Score": scores_for_pandas, "Participant": participant_index_array,
                       "Seance": seances_index_array})
    sns.set(font_scale=3, style="whitegrid")
    sns.lmplot(x="Time (days)", y="Score", data=df, line_kws={'color': 'black', "linewidth": 12},
               scatter_kws={"s": 700, 'color': 'black'}, robust=True, fit_reg=True)

    for seance_i in range(number_of_seances_to_consider + 1):
        get_statistics(df['Score'][df['Seance'] == seance_i], seance_i)

    sns.despine(left=True)
    plt.show()


if __name__ == '__main__':
    print(os.listdir("../../../datasets/longterm_dataset_3DC"))
    scores = get_scores("../../../datasets/longterm_dataset_3DC")

    scores_from_first_evaluation_of_the_day = []
    scores_from_second_evaluation_of_the_day = []
    for i, scores_participant in enumerate(scores):
        print(scores_participant)
        print(np.mean(scores_participant))
        for j, score_session in enumerate(scores_participant):
            if j % 2 == 0:
                scores_from_first_evaluation_of_the_day.append(score_session)
            else:
                scores_from_second_evaluation_of_the_day.append(score_session)

    print(np.mean(scores_from_first_evaluation_of_the_day), "  VS ", np.mean(scores_from_second_evaluation_of_the_day))
    print(scores_from_first_evaluation_of_the_day)
    print(scores_from_second_evaluation_of_the_day)

    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']

    graph_scores(scores=scores, timestamps=evaluation_datetimes)
