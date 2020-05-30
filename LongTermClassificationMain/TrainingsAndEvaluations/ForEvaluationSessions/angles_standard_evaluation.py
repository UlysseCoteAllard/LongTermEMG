import os
import pickle
import numpy as np
import seaborn as sns
from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet as rawConvNet
from LongTermClassificationMain.TrainingsAndEvaluations.utils_training_and_evaluation import \
    create_angles_classification_graph
from LongTermClassificationMain.TrainingsAndEvaluations.ForEvaluationSessions.network_standard_evaluation import \
    test_network_convNet

if __name__ == '__main__':
    path_weights = "../weights_full_training"
    print(os.listdir("../../"))
    with open("../../Processed_datasets/LongTermDataset_evaluation_session.pickle", 'rb') as f:
        dataset_evaluation = pickle.load(file=f)

    examples_datasets_evaluation = dataset_evaluation['examples_evaluation']
    labels_datasets_evaluation = dataset_evaluation['labels_evaluation']
    evaluation_datetimes = dataset_evaluation['evaluation_datetimes']
    angles_arm = dataset_evaluation['angles_and_timestamps']

    num_kernels = [16, 32, 64]
    filter_size = (4, 10)

    test_network_convNet(examples_datasets=examples_datasets_evaluation, labels_datasets=labels_datasets_evaluation,
                         convNet=rawConvNet, type_of_calibration="ReCalibration", path_weights=path_weights,
                         num_kernels=num_kernels, filter_size=filter_size)

    # Graphs production
    ground_truths_WITH_retraining, predictions_WITH_retraining = np.load(
        "../../results/evaluation_sessions_WITH_retraining.npy")

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    font_size = 24
    sns.set(style='dark')
    create_angles_classification_graph(predictions=predictions_WITH_retraining,
                                       ground_truths=ground_truths_WITH_retraining, angles=angles_arm)
