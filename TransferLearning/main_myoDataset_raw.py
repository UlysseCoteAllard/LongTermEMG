import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from TransferLearning.Models.models_training import pre_train_model, train_model
from TransferLearning.Models.target_network_raw_emg_enhanced import SourceNetwork, TargetNetwork
from TransferLearning.PrepareAndLoadData import load_pre_training_dataset, load_evaluation_dataset

def run_pre_training_for_myo_dataset(examples, labels):
    list_train_dataloader = []
    list_val_dataloader = []

    for participant_index in range(len(labels)):
        X_training, Y_gestures = [], []
        X_validation, Y_gesture_validation = [], []
        for k in range(len(examples[participant_index])):
            if k < 21:
                X_training.extend(examples[participant_index][k])
                Y_gestures.extend(labels[participant_index][k])
            else:
                X_validation.extend(examples[participant_index][k])
                Y_gesture_validation.extend(labels[participant_index][k])

        train = TensorDataset(torch.from_numpy(np.array(X_training, dtype=np.float32)),
                              torch.from_numpy(np.array(Y_gestures, dtype=np.int64)))
        validation = TensorDataset(torch.from_numpy(np.array(X_validation, dtype=np.float32)),
                                   torch.from_numpy(np.array(Y_gesture_validation, dtype=np.int64)))

        trainLoader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
        validationLoader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)

        list_train_dataloader.append(trainLoader)
        list_val_dataloader.append(validationLoader)

        print("Shape training : ", np.shape(X_training))
        print("Shape valid : ", np.shape(X_validation))

    model = SourceNetwork(number_of_class=7, dropout_rate=0.35).cuda()
    criterion_class = nn.CrossEntropyLoss(reduction='sum')
    criterion_domain = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.002335721469090121)
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15,
                                                     verbose=True, eps=precision)

    best_weights = pre_train_model(model=model, criterion_class=criterion_class, criterion_domain=criterion_domain,
                                   optimizer=optimizer, scheduler=scheduler,
                                   dataloaders={"train": list_train_dataloader, "val": list_val_dataloader},
                                   precision=precision, lambda_value=0.1)

    # Save the best weights found to file
    torch.save(best_weights, "weights/best_pre_train_weights_target_raw_for_MyoDataset.pt")
    print("Done Pre-Training for Myo Armband Dataset")

def calculate_fitness(examples_training, labels_training, examples_test0, labels_test0, examples_test1, labels_test_1,
                      learning_rate=.1, training_cycle=4):
    accuracy_test0 = []
    accuracy_test1 = []
    for j in range(len(labels_training)):
        print("CURRENT DATASET : ", j)
        X_test_0, X_test_1, Y_test_0, Y_test_1, trainloader, validationloader = prepare_dataset_evaluation_for_training(
            examples_test0, examples_test1, examples_training, j, labels_test0, labels_test_1, labels_training,
            training_cycle)

        pre_trained_weights = torch.load('weights/best_pre_train_weights_target_raw_for_MyoDataset.pt')
        model = TargetNetwork(number_of_class=7, weights_pre_trained_convnet=pre_trained_weights, dropout=.5).cuda()

        criterion = nn.CrossEntropyLoss(size_average=False)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        precision = 1e-6
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)

        model = train_model(model, criterion, optimizer, scheduler, dataloaders={"train": trainloader,
                                                                                 "val": validationloader},
                            precision=precision)

        acc_test0, acc_test1 = evaluate_model(X_test_0, X_test_1, Y_test_0, Y_test_1, model)
        accuracy_test0.append(acc_test0)
        accuracy_test1.append(acc_test1)

    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_test0).mean())
    print("AVERAGE ACCURACY TEST 1 %.3f" % np.array(accuracy_test1).mean())
    return accuracy_test0, accuracy_test1


def prepare_dataset_evaluation_for_training(examples_test0, examples_test1, examples_training, j, labels_test0,
                                            labels_test_1, labels_training, training_cycle):
    examples_personne_training = []
    labels_gesture_personne_training = []
    for k in range(len(examples_training[j])):
        if k < training_cycle * 7:
            examples_personne_training.extend(examples_training[j][k])
            labels_gesture_personne_training.extend(labels_training[j][k])
    X_test_0, Y_test_0 = [], []
    for k in range(len(examples_test0)):
        X_test_0.extend(examples_test0[j][k])
        Y_test_0.extend(labels_test0[j][k])
    X_test_1, Y_test_1 = [], []
    for k in range(len(examples_test1)):
        X_test_1.extend(examples_test1[j][k])
        Y_test_1.extend(labels_test_1[j][k])
    print(np.shape(examples_personne_training))
    valid_examples = examples_personne_training[0:int(len(examples_personne_training) * 0.1)]
    labels_valid = labels_gesture_personne_training[0:int(len(labels_gesture_personne_training) * 0.1)]
    X_fine_tune = examples_personne_training[int(len(examples_personne_training) * 0.1):]
    Y_fine_tune = labels_gesture_personne_training[int(len(labels_gesture_personne_training) * 0.1):]
    train = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                          torch.from_numpy(np.array(Y_fine_tune, dtype=np.int64)))
    validation = TensorDataset(torch.from_numpy(np.array(valid_examples, dtype=np.float32)),
                               torch.from_numpy(np.array(labels_valid, dtype=np.int64)))
    trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
    validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)
    return X_test_0, X_test_1, Y_test_0, Y_test_1, trainloader, validationloader


def evaluate_model(X_test_0, X_test_1, Y_test_0, Y_test_1, model):
    model.eval()
    test_0 = TensorDataset(torch.from_numpy(np.array(X_test_0, dtype=np.float32)),
                           torch.from_numpy(np.array(Y_test_0, dtype=np.int64)))
    test_1 = TensorDataset(torch.from_numpy(np.array(X_test_1, dtype=np.float32)),
                           torch.from_numpy(np.array(Y_test_1, dtype=np.int64)))
    test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=256, shuffle=False)
    total_0 = 0
    correct_prediction_test_0 = 0
    with torch.no_grad():
        for k, data_test_0 in enumerate(test_0_loader, 0):
            # get the inputs
            inputs_test_0, ground_truth_test_0 = data_test_0
            inputs_test_0, ground_truth_test_0 = inputs_test_0.cuda(), ground_truth_test_0.cuda()

            outputs_test_0 = model(inputs_test_0)
            _, predicted = torch.max(outputs_test_0.data, 1)
            correct_prediction_test_0 += (predicted.cpu().numpy() == ground_truth_test_0.data.cpu().numpy()).sum()
            total_0 += ground_truth_test_0.size(0)
    print("ACCURACY TEST_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total_0)))
    test_1_loader = torch.utils.data.DataLoader(test_1, batch_size=256, shuffle=False)
    total_1 = 0
    correct_prediction_test_1 = 0
    with torch.no_grad():
        for k, data_test_1 in enumerate(test_1_loader, 0):
            # get the inputs
            inputs_test_1, ground_truth_test_1 = data_test_1
            inputs_test_1, ground_truth_test_1 = inputs_test_1.cuda(), ground_truth_test_1.cuda()

            outputs_test_1 = model(inputs_test_1)
            _, predicted = torch.max(outputs_test_1.data, 1)
            correct_prediction_test_1 += (predicted.cpu().numpy() == ground_truth_test_1.data.cpu().numpy()).sum()
            total_1 += ground_truth_test_1.size(0)
    print("ACCURACY TEST_1 FINAL : %.3f %%" % (100 * float(correct_prediction_test_1) / float(total_1)))
    return 100 * float(correct_prediction_test_0) / float(total_0), \
           100 * float(correct_prediction_test_1) / float(total_1)


if __name__ == '__main__':
    # Comment between here
    '''
    examples, labels = load_pre_training_dataset.read_data('../datasets/MyoDataset/PreTrainingDataset')
    datasets = [examples, labels]
    np.save("Processed_datasets/saved_pre_training_dataset_raw.npy", datasets)
    '''
    # And here if the pre-training dataset was already processed and saved

    # Comment between here

    datasets_pre_training = np.load("Processed_datasets/saved_pre_training_dataset_raw.npy", encoding="bytes")
    examples_pre_training, labels_pre_training = datasets_pre_training
    run_pre_training_for_myo_dataset(examples_pre_training, labels_pre_training)

    # Comment between here

    # Comment between here
    '''
    examples, labels = load_evaluation_dataset.read_data('../datasets/MyoDataset/EvaluationDataset', type="training0")
    datasets = [examples, labels]

    np.save("Processed_datasets/saved_evaluation_dataset_training.npy", datasets)

    examples, labels = load_evaluation_dataset.read_data('../datasets/MyoDataset/EvaluationDataset', type="Test0")
    datasets = [examples, labels]

    np.save("Processed_datasets/saved_evaluation_dataset_test0.npy", datasets)

    examples, labels = load_evaluation_dataset.read_data('../datasets/MyoDataset/EvaluationDataset', type="Test1")
    datasets = [examples, labels]

    np.save("Processed_datasets/saved_evaluation_dataset_test1.npy", datasets)
    '''
    # And here if the pre-training dataset was already processed and saved

    datasets_training = np.load("Processed_datasets/saved_evaluation_dataset_training.npy", encoding="bytes")
    examples_training, labels_training = datasets_training

    datasets_test0 = np.load("Processed_datasets/saved_evaluation_dataset_test0.npy", encoding="bytes")
    examples_test0, labels_test0 = datasets_test0

    datasets_test1 = np.load("Processed_datasets/saved_evaluation_dataset_test1.npy", encoding="bytes")
    examples_test1, labels_test1 = datasets_test1

    # And here if the pre-training of the network was already completed.
    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []
    # learning_rate=0.002335721469090121 (for network enhanced)

    with open("results/evaluation_dataset_TARGET_Adversarial_convnet_enhanced.txt", "a") as myfile:
        myfile.write("Test")
    for training_cycle in range(1, 5):
        test_0 = []
        test_1 = []
        for i in range(20):
            accuracy_test0, accuracy_test1 = calculate_fitness(examples_training, labels_training, examples_test0,
                                                               labels_test0, examples_test1, labels_test1,
                                                               learning_rate=0.002335721469090121,
                                                               training_cycle=training_cycle)

            test_0.append(accuracy_test0)
            test_1.append(accuracy_test1)
            print("TEST 0 SO FAR: ", test_0)
            print("TEST 1 SO FAR: ", test_1)
            print("CURRENT AVERAGE : ", (np.mean(test_0) + np.mean(test_1)) / 2.)

        print("ACCURACY FINAL TEST 0: ", test_0)
        print("ACCURACY FINAL TEST 0: ", np.mean(test_0))
        print("ACCURACY FINAL TEST 1: ", test_1)
        print("ACCURACY FINAL TEST 1: ", np.mean(test_1))
        print("ACCURACY FINAL: ", (np.mean(test_0) + np.mean(test_1)) / 2.)

        with open("results/evaluation_dataset_TARGET_Adversarial_convnet_enhanced.txt", "a") as myfile:
            myfile.write("ConvNet Training Cycle : " + str(training_cycle) + "\n\n")
            myfile.write("Test 0: \n")
            myfile.write(str(test_0) + '\n')
            myfile.write(str(np.mean(test_0, axis=0)) + '\n')
            myfile.write(str(np.mean(test_0)) + '\n')
            myfile.write("Test 1: \n")
            myfile.write(str(test_1) + '\n')
            myfile.write(str(np.mean(test_1, axis=0)) + '\n')
            myfile.write(str(np.mean(test_1)) + '\n')
            myfile.write("Test Mean: \n")
            myfile.write(str(np.mean(test_0, axis=0)) + '\n')
            myfile.write(str((np.mean(test_0) + np.mean(test_1)) / 2.) + '\n')
            myfile.write("\n\n\n")
