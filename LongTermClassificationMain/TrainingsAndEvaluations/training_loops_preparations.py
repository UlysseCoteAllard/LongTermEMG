import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet
from LongTermClassificationMain.Models.raw_TCN_Transfer_Learning import SourceNetwork, TargetNetwork
from LongTermClassificationMain.Models.model_training import train_model_standard, vadaTraining, DANN_Training,\
    MTDA_training, dirt_T_training, MTVADA_training, train_batch_norm, MTDANN_training
from LongTermClassificationMain.Models.model_training_Transfer_Learning import pre_train_model
from LongTermClassificationMain.PrepareAndLoadDataLongTerm.\
    load_dataset_in_dataloader import load_dataloaders_training_sessions
from LongTermClassificationMain.Models.model_utils import VATLoss, ConditionalEntropyLoss


def train_raw_convNet(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                      number_of_cycle_for_first_training=2, number_of_cycles_rest_of_training=2,
                      path_weight_to_save_to="../weights"):

    participants_train, participants_validation, _ = load_dataloaders_training_sessions(
    examples_datasets_train, labels_datasets_train, batch_size=512,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training)
    #participants_train, participants_validation, _, _, _, _ = load_dataloaders_training_sessions(
    #    examples_datasets_train, labels_datasets_train, batch_size=512, number_of_cycle_for_first_training=2)

    for participant_i in range(len(participants_train)):
        for session_j in range(len(participants_train[participant_i])):
            # Define Model
            #model = rawConvNet(number_of_class=11, number_of_blocks=3, dropout_rate=0.5, filter_size=filter_size,
            #                   number_of_features_output=[64, 64, 64]).cuda()
            model = TemporalConvNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()

            # Define Loss functions
            cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

            # Define Optimizer
            learning_rate = 0.0404709
            print(model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)

            best_weights = train_model_standard(model=model, criterion=cross_entropy_loss_classes, optimizer=optimizer,
                                                scheduler=scheduler,
                                                dataloaders={"train": participants_train[participant_i][session_j],
                                                             "val": participants_validation[participant_i][session_j]},
                                                precision=precision, patience=10, patience_increase=10)

            if not os.path.exists(path_weight_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weight_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_weights, f=path_weight_to_save_to +
                                       "/participant_%d/best_weights_participant_normal_training_%d.pt"
                                       % (participant_i, session_j))


def train_DA_convNet(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 16),
                     path_weights_to_load_from_for_dirtT='../weights_VADA', algo_name="DANN", path_weights_to_save_to="../weights_",
                     batch_size=512, patience_increment=10):
    participants_train, participants_validation, participants_test = load_dataloaders_training_sessions(
    examples_datasets_train, labels_datasets_train, batch_size=batch_size, number_of_cycle_for_first_training=2,
        get_validation_set=True, number_of_cycles_rest_of_training=2)

    for participant_i in range(len(participants_train)):
        print("SHAPE SESSIONS: ", np.shape(participants_train[participant_i]))

        # Skip the first session as it will be identical to normal training
        for session_j in range(1, len(participants_train[participant_i])):
            print(np.shape(participants_train[participant_i][session_j]))

            # Classifier and discriminator
            gesture_classification = TemporalConvNet(number_of_class=11, num_kernels=num_kernels,
                                                     kernel_size=filter_size, dropout=0.5).cuda()
            # Get the weights found during the first training
            '''
            best_weights = torch.load(
                path_weights_to_load_from + "/participant_%d/best_weights_participant_normal_training_%d.pt" %
                (participant_i, 0))
            gesture_classification.load_state_dict(best_weights, strict=False)
            '''

            # loss functions
            crossEntropyLoss = nn.CrossEntropyLoss().cuda()
            # optimizer
            precision = 1e-8
            learning_rate = 0.0404709
            optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_classifier, mode='min', factor=.2,
                                                             patience=5, verbose=True, eps=precision)
            if algo_name == "DANN":
                best_weights = DANN_Training(gesture_classifier=gesture_classification, scheduler=scheduler,
                                             optimizer_classifier=optimizer_classifier,
                                             train_dataset_source=participants_train[participant_i][0],
                                             train_dataset_target=participants_train[participant_i][session_j],
                                             validation_dataset_source=participants_validation[participant_i][0],
                                             crossEntropyLoss=crossEntropyLoss, patience_increment=patience_increment)
            elif algo_name == "VADA":
                # VADA need Conditional Entropy loss and Virtual Adversarial Training loss too
                conditionalEntropy = ConditionalEntropyLoss().cuda()
                vatLoss = VATLoss(gesture_classification).cuda()

                best_weights = vadaTraining(gesture_classifier=gesture_classification,
                                            conditionalEntropyLoss=conditionalEntropy,
                                            crossEntropyLoss=crossEntropyLoss, vatLoss=vatLoss, scheduler=scheduler,
                                            optimizer_classifier=optimizer_classifier,
                                            train_dataset_source=participants_train[participant_i][0],
                                            train_dataset_target=participants_train[participant_i][session_j],
                                            validation_dataset_source=participants_validation[participant_i][0],
                                            patience_increment=patience_increment)
            elif "Dirt_T" in algo_name:
                learning_rate = 0.0404709*0.2
                optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate,
                                                  betas=(0.5, 0.999))
                # Dirt T need Conditional Entropy loss and Virtual Adversarial Training loss too
                conditionalEntropy = ConditionalEntropyLoss().cuda()
                vatLoss = VATLoss(gesture_classification).cuda()

                best_weights_loaded = torch.load(
                    path_weights_to_load_from_for_dirtT + "/participant_%d/best_weights_participant_normal_training_%d.pt" %
                    (participant_i, session_j))

                gesture_classification.load_state_dict(best_weights_loaded, strict=False)

                best_weights = dirt_T_training(gesture_classifier=gesture_classification,
                                               conditionalEntropyLoss=conditionalEntropy,
                                               crossEntropyLoss=crossEntropyLoss, vatLoss=vatLoss, scheduler=scheduler,
                                               optimizer_classifier=optimizer_classifier,
                                               train_dataset_source=participants_train[participant_i][session_j],
                                               validation_dataset_source=participants_validation[participant_i][0],
                                               patience_increment=patience_increment, batch_size=batch_size)
            elif algo_name == "MTVADA":
                # VADA need Conditional Entropy loss and Virtual Adversarial Training loss too
                conditionalEntropy = ConditionalEntropyLoss().cuda()
                vatLoss = VATLoss(gesture_classification).cuda()

                best_weights = MTVADA_training(gesture_classifier=gesture_classification,
                                            conditionalEntropyLoss=conditionalEntropy,
                                            crossEntropyLoss=crossEntropyLoss, vatLoss=vatLoss, scheduler=scheduler,
                                            optimizer_classifier=optimizer_classifier,
                                            train_dataset_source=participants_train[participant_i][0],
                                            train_dataset_target=participants_train[participant_i][session_j],
                                            validation_dataset_source=participants_validation[participant_i][0],
                                            patience_increment=patience_increment)
            elif algo_name == "MTDANN":
                # VADA need Conditional Entropy loss and Virtual Adversarial Training loss too

                best_weights = MTDANN_training(gesture_classifier=gesture_classification,
                                            crossEntropyLoss=crossEntropyLoss,scheduler=scheduler,
                                            optimizer_classifier=optimizer_classifier,
                                            train_dataset_source=participants_train[participant_i][0],
                                            train_dataset_target=participants_train[participant_i][session_j],
                                            validation_dataset_source=participants_validation[participant_i][0],
                                            patience_increment=patience_increment)
            elif algo_name == "BatchNorm":
                best_weights_DANN = torch.load("../weights/participant_%d/best_weights_participant_normal_training_%d.pt" %
                    (participant_i, 0))

                gesture_classification.load_state_dict(best_weights_DANN, strict=False)
                gesture_classification.stop_gradient_except_for_batch_norm()
                best_weights = train_batch_norm(model=gesture_classification,
                                                dataloader=participants_train[participant_i][session_j])

            else:  # MTDA
                best_weights = MTDA_training(gesture_classifier=gesture_classification, scheduler=scheduler,
                                             optimizer_classifier=optimizer_classifier,
                                             train_dataset_source=participants_train[participant_i][0],
                                             train_dataset_target=participants_train[participant_i][session_j],
                                             validation_dataset_source=participants_validation[participant_i][0],
                                             crossEntropyLoss=crossEntropyLoss, patience_increment=patience_increment)

            if not os.path.exists(path_weights_to_save_to + algo_name + "/participant_%d" % participant_i):
                os.makedirs(path_weights_to_save_to + algo_name + "/participant_%d" % participant_i)
            torch.save(best_weights, f=path_weights_to_save_to + algo_name +
                                       "/participant_%d/best_weights_participant_normal_training_%d.pt"  %
                                       (participant_i, session_j))


def train_TL_convNet(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                     number_of_cycle_for_first_training=2, number_of_cycles_rest_of_training=2,
                     path_weight_to_save_to="../weights_TL"):
    participants_train_for_source, participants_validation_for_source, _ = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=512,
        number_of_cycle_for_first_training=4,
        number_of_cycles_rest_of_training=4)
    participants_train_for_target, participants_validation_for_target, _ = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=512,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training)

    for participant_i in range(len(participants_train_for_target)):
        # Skip the first session as it will be identical to normal training
        for session_j in range(1, len(participants_train_for_target[participant_i])):
            '''
            model_source = SourceNetwork(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()
            # Define Loss functions
            cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
            # Define Optimizer
            learning_rate = 0.0404709
            print(model_source.parameters())
            optimizer = optim.Adam(model_source.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=10,
                                                             verbose=True, eps=precision)
            list_train_dataloader = []
            list_validation_dataloader = []
            # Get all sessions before the current one and pre-train on these
            for k in range(0, session_j):
                list_train_dataloader.append(participants_train_for_source[participant_i][k])
                list_validation_dataloader.append(participants_validation_for_source[participant_i][k])
            best_weights_pre_training = pre_train_model(model=model_source, cross_entropy_loss=cross_entropy_loss,
                                                        optimizer_class=optimizer, scheduler=scheduler,
                                                        dataloaders={"train": list_train_dataloader,
                                                                     "val": list_validation_dataloader},
                                                        patience=20, patience_increase=20)

            if not os.path.exists(path_weight_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weight_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_weights_pre_training, f=path_weight_to_save_to +
                                                    "/participant_%d/best_weights_participant_pre_training_%d.pt"
                                                    % (participant_i, session_j-1))
            '''

            '''Train the source network with the current session'''
            weights_pre_training = torch.load(path_weight_to_save_to +
                                              "/participant_%d/best_weights_participant_pre_training_%d.pt" %
                                              (participant_i, session_j-1))
            model_target = TargetNetwork(weight_pre_trained_convNet=weights_pre_training, num_kernels=num_kernels,
                                         kernel_size=filter_size).cuda()
            # Define Loss functions
            cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
            # Define Optimizer
            learning_rate = 0.0404709
            print(model_target.parameters())
            optimizer = optim.Adam(model_target.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)
            best_weights = train_model_standard(model=model_target, criterion=cross_entropy_loss, optimizer=optimizer,
                                                scheduler=scheduler,
                                                dataloaders={"train":
                                                                 participants_train_for_target[participant_i][
                                                                     session_j],
                                                             "val":
                                                                 participants_validation_for_target[participant_i][
                                                                     session_j]},
                                                precision=precision, patience=10, patience_increase=10)
            if not os.path.exists(path_weight_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weight_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_weights, f=path_weight_to_save_to + "/participant_%d/"
                                                                "best_weights_participant_normal_training_%d.pt" %
                                       (participant_i, session_j))
