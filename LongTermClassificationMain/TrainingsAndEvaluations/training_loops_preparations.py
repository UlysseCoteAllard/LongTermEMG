import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from LongTermClassificationMain.Models.spectrogram_ConvNet import SpectrogramConvNet
from LongTermClassificationMain.Models.raw_TCN import TemporalConvNet
from LongTermClassificationMain.Models.raw_TCN_Transfer_Learning import SourceNetwork, TargetNetwork
from LongTermClassificationMain.Models.model_training import train_model_standard, dirt_T_training, DANN_BN_Training, \
    vada_BN_Training, AdaBN_adaptation
from LongTermClassificationMain.Models.model_training_Transfer_Learning import pre_train_model
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_spectrogram_in_dataloader import \
    load_dataloaders_training_sessions as load_dataloaders_training_sessions_spectrogram
from LongTermClassificationMain.PrepareAndLoadDataLongTerm. \
    load_dataset_in_dataloader import load_dataloaders_training_sessions, load_dataloaders_test_sessions
from LongTermClassificationMain.Models.model_utils import VATLoss, ConditionalEntropyLoss


def load_checkpoint(model, filename, optimizer=None, scheduler=None, strict=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        if optimizer is not None:
            print("Loading Optimizer")
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch


def train_Spectrogram_fine_tuning(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                                  number_of_cycle_for_first_training=2, number_of_cycles_rest_of_training=2,
                                  path_weight_to_save_to=
                                  "../weights_SPECTROGRAMS_TWO_CYCLES_normal_training_fine_tuning",
                                  gestures_to_remove=None, number_of_classes=11, batch_size=512):
    participants_train, participants_validation, _ = load_dataloaders_training_sessions_spectrogram(
        examples_datasets_train, labels_datasets_train, batch_size=batch_size,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, gestures_to_remove=gestures_to_remove,
        ignore_first=True)

    for participant_i in range(len(participants_train)):
        print("Participant: ", participant_i)
        for session_j in range(0, len(participants_train[participant_i])):
            print("Session: ", session_j)
            # Define Model
            model = SpectrogramConvNet(number_of_class=number_of_classes, num_kernels=num_kernels,
                                       kernel_size=filter_size).cuda()

            # Define Loss functions
            cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

            # Define Optimizer
            learning_rate = 0.001316
            print(model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)

            if session_j > 0:
                # Fine-tune from the previous training
                model, _, _, start_epoch = load_checkpoint(
                    model=model, optimizer=None, scheduler=None,
                    filename=path_weight_to_save_to +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, session_j - 1))

            best_state = train_model_standard(model=model, criterion=cross_entropy_loss_classes, optimizer=optimizer,
                                              scheduler=scheduler,
                                              dataloaders={"train": participants_train[participant_i][session_j],
                                                           "val": participants_validation[participant_i][session_j]},
                                              precision=precision, patience=10, patience_increase=10)

            if not os.path.exists(path_weight_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weight_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_state, f=path_weight_to_save_to +
                                     "/participant_%d/best_state_%d.pt"
                                     % (participant_i, session_j))


def train_DA_spectrograms_evaluation(examples_datasets_evaluations, labels_datasets_evaluation,
                                     examples_datasets_train, labels_datasets_train, algo_name,
                                     num_kernels, filter_size, path_weights_to_load_from, path_weights_DA,
                                     batch_size=512, patience_increment=10, use_recalibration_data=False,
                                     number_of_cycle_for_first_training=4, number_of_cycles_rest_of_training=4):
    # Get the data to use as the SOURCE from the training sessions
    participants_train, participants_validation, _ = load_dataloaders_training_sessions_spectrogram(
        examples_datasets_train, labels_datasets_train, batch_size=batch_size,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training, get_validation_set=True,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, gestures_to_remove=None,
        ignore_first=True, shuffle=True)

    # Get the data to use as the TARGET from the evaluation sessions
    participants_evaluation_dataloader = load_dataloaders_test_sessions(
        examples_datasets_evaluation=examples_datasets_evaluations,
        labels_datasets_evaluation=labels_datasets_evaluation, batch_size=batch_size, shuffle=True, drop_last=True)

    for participant_i in range(len(participants_evaluation_dataloader)):
        print("SHAPE SESSIONS: ", np.shape(participants_evaluation_dataloader[participant_i]))
        for session_j in range(0, len(participants_evaluation_dataloader[participant_i])):
            # There is two evaluation session for every training session. We train on the first one
            if session_j % 2 == 0:
                # Get the weights trained
                corresponding_training_session_index = 0 if use_recalibration_data is False else int(session_j / 2)

                # Classifier and discriminator
                gesture_classification = SpectrogramConvNet(number_of_class=11, num_kernels=num_kernels,
                                                            kernel_size=filter_size, dropout=0.5).cuda()
                # loss functions
                crossEntropyLoss = nn.CrossEntropyLoss().cuda()
                # optimizer
                precision = 1e-8
                learning_rate = 0.001316
                optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate,
                                                  betas=(0.5, 0.999))
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_classifier, mode='min', factor=.2,
                                                                 patience=5, verbose=True, eps=precision)

                gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint(
                    model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,
                    filename=path_weights_to_load_from +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, corresponding_training_session_index))

                best_state = None
                if "DANN" in algo_name:
                    best_state = DANN_BN_Training(gesture_classifier=gesture_classification, scheduler=scheduler,
                                                  optimizer_classifier=optimizer_classifier,
                                                  train_dataset_source=participants_train[participant_i][
                                                      corresponding_training_session_index],
                                                  train_dataset_target=participants_evaluation_dataloader[
                                                      participant_i][session_j],
                                                  validation_dataset_source=participants_validation[participant_i][
                                                      corresponding_training_session_index],
                                                  crossEntropyLoss=crossEntropyLoss,
                                                  patience_increment=patience_increment,
                                                  domain_loss_weight=1e-1)
                elif "VADA" in algo_name:
                    # VADA need Conditional Entropy loss and Virtual Adversarial Training loss too
                    conditionalEntropy = ConditionalEntropyLoss().cuda()
                    vatLoss = VATLoss(gesture_classification).cuda()

                    best_state = vada_BN_Training(gesture_classifier=gesture_classification,
                                                  conditionalEntropyLoss=conditionalEntropy,
                                                  crossEntropyLoss=crossEntropyLoss, vatLoss=vatLoss,
                                                  scheduler=scheduler,
                                                  optimizer_classifier=optimizer_classifier,
                                                  train_dataset_source=participants_train[participant_i][0],
                                                  train_dataset_target=
                                                  participants_evaluation_dataloader[participant_i][
                                                      session_j],
                                                  validation_dataset_source=participants_validation[participant_i][0],
                                                  patience_increment=patience_increment)
                elif "DirtT" in algo_name:
                    learning_rate = 0.001316
                    # Dirt T need Conditional Entropy loss and Virtual Adversarial Training loss too
                    conditionalEntropy = ConditionalEntropyLoss().cuda()
                    vatLoss = VATLoss(gesture_classification).cuda()

                    # Classifier and discriminator
                    gesture_classification = SpectrogramConvNet(number_of_class=11, num_kernels=num_kernels,
                                                                kernel_size=filter_size, dropout=0.5).cuda()
                    # loss functions
                    crossEntropyLoss = nn.CrossEntropyLoss().cuda()
                    # optimizer
                    precision = 1e-8
                    learning_rate = 0.001316
                    optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate,
                                                      betas=(0.5, 0.999))
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_classifier, mode='min',
                                                                     factor=.2,
                                                                     patience=5, verbose=True, eps=precision)
                    if use_recalibration_data:
                        gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint(
                            model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,
                            filename=path_weights_to_load_from +
                                     "/participant_%d/best_state_WITH_recalibration%d.pt" %
                                     (participant_i, session_j))
                    else:
                        gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint(
                            model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,
                            filename=path_weights_to_load_from +
                                     "/participant_%d/best_state_NO_recalibration%d.pt" %
                                     (participant_i, session_j))

                    best_state = dirt_T_training(gesture_classifier=gesture_classification,
                                                 conditionalEntropyLoss=conditionalEntropy,
                                                 crossEntropyLoss=crossEntropyLoss, vatLoss=vatLoss,
                                                 scheduler=scheduler,
                                                 optimizer_classifier=optimizer_classifier,
                                                 train_dataset_source=participants_evaluation_dataloader[participant_i][
                                                     session_j],
                                                 patience_increment=patience_increment, batch_size=batch_size)

                if use_recalibration_data:
                    if not os.path.exists(path_weights_DA + algo_name + "/participant_%d" % participant_i):
                        os.makedirs(path_weights_DA + algo_name + "/participant_%d" % participant_i)
                    torch.save(best_state, f=path_weights_DA + algo_name +
                                             "/participant_%d/best_state_WITH_recalibration%d.pt" %
                                             (participant_i, session_j))
                else:
                    if not os.path.exists(path_weights_DA + algo_name + "/participant_%d" % participant_i):
                        os.makedirs(path_weights_DA + algo_name + "/participant_%d" % participant_i)
                    print(os.listdir(path_weights_DA + algo_name))
                    torch.save(best_state, f=path_weights_DA + algo_name +
                                             "/participant_%d/best_state_NO_recalibration%d.pt" % (
                                                 participant_i, session_j))


def train_AdaBN_spectrograms(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                             algo_name="AdaBN", path_weights_to_save_to="../Weights/weights_", batch_size=512,
                             patience_increment=10,
                             path_weights_fine_tuning="../weights_TWO_CYCLES_normal_training_fine_tuning",
                             number_of_cycle_for_first_training=3, number_of_cycles_rest_of_training=3,
                             gestures_to_remove=None, number_of_classes=11):
    participants_train, participants_validation, participants_test = load_dataloaders_training_sessions_spectrogram(
        examples_datasets_train, labels_datasets_train, batch_size=batch_size,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training, get_validation_set=True,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, gestures_to_remove=gestures_to_remove,
        ignore_first=True)

    for participant_i in range(len(participants_train)):
        print("SHAPE SESSIONS: ", np.shape(participants_train[participant_i]))
        # Skip the first session as it will be identical to normal training
        for session_j in range(1, len(participants_train[participant_i])):
            print(np.shape(participants_train[participant_i][session_j]))

            # Classifier and discriminator
            gesture_classification = SpectrogramConvNet(number_of_class=number_of_classes, num_kernels=num_kernels,
                                                        kernel_size=filter_size, dropout=0.5).cuda()

            # optimizer
            precision = 1e-8
            learning_rate = 0.001316
            optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_classifier, mode='min', factor=.2,
                                                             patience=5, verbose=True, eps=precision)

            gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint(
                model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,
                filename=path_weights_fine_tuning +
                         "/participant_%d/best_state_%d.pt" %
                         (participant_i, 0))
            # Freeze all the weights except those associated with the BN statistics
            gesture_classification.freeze_all_except_BN()

            best_weights = AdaBN_adaptation(model=gesture_classification, scheduler=scheduler,
                                            optimizer_classifier=optimizer_classifier,
                                            dataloader=participants_train[participant_i][session_j])

            if not os.path.exists(path_weights_to_save_to + algo_name + "/participant_%d" % participant_i):
                os.makedirs(path_weights_to_save_to + algo_name + "/participant_%d" % participant_i)
            torch.save(best_weights, f=path_weights_to_save_to + algo_name +
                                       "/participant_%d/best_state_%d.pt" %
                                       (participant_i, session_j))

def train_DA_spectrograms(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                          path_weights_to_load_from_for_dirtT='../weights_VADA_TWO_Cycles', algo_name="DANN",
                          path_weights_to_save_to="../Weights/weights_", batch_size=512, patience_increment=10,
                          path_weights_fine_tuning="../weights_TWO_CYCLES_normal_training_fine_tuning",
                          number_of_cycle_for_first_training=3, number_of_cycles_rest_of_training=3,
                          gestures_to_remove=None, number_of_classes=11):
    participants_train, participants_validation, participants_test = load_dataloaders_training_sessions_spectrogram(
        examples_datasets_train, labels_datasets_train, batch_size=batch_size,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training, get_validation_set=True,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, gestures_to_remove=gestures_to_remove,
        ignore_first=True)

    for participant_i in range(len(participants_train)):
        print("SHAPE SESSIONS: ", np.shape(participants_train[participant_i]))

        # Skip the first session as it will be identical to normal training
        for session_j in range(1, len(participants_train[participant_i])):
            print(np.shape(participants_train[participant_i][session_j]))

            # Classifier and discriminator
            gesture_classification = SpectrogramConvNet(number_of_class=number_of_classes, num_kernels=num_kernels,
                                                        kernel_size=filter_size, dropout=0.5).cuda()

            # loss functions
            crossEntropyLoss = nn.CrossEntropyLoss().cuda()
            # optimizer
            precision = 1e-8
            learning_rate = 0.001316
            optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_classifier, mode='min', factor=.2,
                                                             patience=5, verbose=True, eps=precision)
            # Fine-tune from the previous training
            '''
            if session_j == 0:
                gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint(
                    model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,
                    filename=path_weights_fine_tuning +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, 0))
            else:
                gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint(
                    model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,
                    filename="../weights_Reduced_Spectrograms_SLADANN" +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, session_j))
                '''
            gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint(
                model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,
                filename=path_weights_fine_tuning +
                         "/participant_%d/best_state_%d.pt" %
                         (participant_i, 0))

            if "DANN" in algo_name:
                best_weights = DANN_BN_Training(gesture_classifier=gesture_classification, scheduler=scheduler,
                                                optimizer_classifier=optimizer_classifier,
                                                train_dataset_source=participants_train[participant_i][0],
                                                train_dataset_target=participants_train[participant_i][session_j],
                                                validation_dataset_source=participants_validation[participant_i][0],
                                                crossEntropyLoss=crossEntropyLoss,
                                                patience_increment=patience_increment,
                                                domain_loss_weight=1e-1)
            elif "VADA" in algo_name:
                # VADA need Conditional Entropy loss and Virtual Adversarial Training loss too
                conditionalEntropy = ConditionalEntropyLoss().cuda()
                vatLoss = VATLoss(gesture_classification).cuda()

                best_weights = vada_BN_Training(gesture_classifier=gesture_classification,
                                                conditionalEntropyLoss=conditionalEntropy,
                                                crossEntropyLoss=crossEntropyLoss, vatLoss=vatLoss, scheduler=scheduler,
                                                optimizer_classifier=optimizer_classifier,
                                                train_dataset_source=participants_train[participant_i][0],
                                                train_dataset_target=participants_train[participant_i][session_j],
                                                validation_dataset_source=participants_validation[participant_i][0],
                                                patience_increment=patience_increment)
            elif "Dirt_T" in algo_name:
                learning_rate = 0.001316
                optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate,
                                                  betas=(0.5, 0.999))
                # Dirt T need Conditional Entropy loss and Virtual Adversarial Training loss too
                conditionalEntropy = ConditionalEntropyLoss().cuda()
                vatLoss = VATLoss(gesture_classification).cuda()

                gesture_classification, optimizer_classifier, _, start_epoch = load_checkpoint(
                    model=gesture_classification, optimizer=optimizer_classifier, scheduler=None,
                    filename=path_weights_to_load_from_for_dirtT +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, session_j))

                best_weights = dirt_T_training(gesture_classifier=gesture_classification,
                                               conditionalEntropyLoss=conditionalEntropy,
                                               crossEntropyLoss=crossEntropyLoss, vatLoss=vatLoss, scheduler=scheduler,
                                               optimizer_classifier=optimizer_classifier,
                                               train_dataset_source=participants_train[participant_i][session_j],
                                               patience_increment=patience_increment, batch_size=batch_size)

            if not os.path.exists(path_weights_to_save_to + algo_name + "/participant_%d" % participant_i):
                os.makedirs(path_weights_to_save_to + algo_name + "/participant_%d" % participant_i)
            torch.save(best_weights, f=path_weights_to_save_to + algo_name +
                                       "/participant_%d/best_state_%d.pt" %
                                       (participant_i, session_j))


def train_raw_TCN_fine_tuning(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                              number_of_cycle_for_first_training=2, number_of_cycles_rest_of_training=2,
                              path_weight_to_save_to="../weights_single_cycle_normal_training"):
    participants_train, participants_validation, _ = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=512,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, ignore_first=True)

    for participant_i in range(len(participants_train)):
        for session_j in range(0, len(participants_train[participant_i])):
            # Define Model
            model = TemporalConvNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()

            # Define Loss functions
            cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

            # Define Optimizer
            learning_rate = 0.001316
            print(model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)

            if session_j > 0:
                # Fine-tune from the previous training
                model, optimizer, _, start_epoch = load_checkpoint(
                    model=model, optimizer=optimizer, scheduler=None,
                    filename=path_weight_to_save_to + "/participant_%d/best_state_%d.pt" %
                             (participant_i, session_j - 1))

            best_state = train_model_standard(model=model, criterion=cross_entropy_loss_classes, optimizer=optimizer,
                                              scheduler=scheduler,
                                              dataloaders={"train": participants_train[participant_i][session_j],
                                                           "val": participants_validation[participant_i][session_j]},
                                              precision=precision, patience=10, patience_increase=10)

            if not os.path.exists(path_weight_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weight_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_state, f=path_weight_to_save_to +
                                     "/participant_%d/best_state_%d.pt"
                                     % (participant_i, session_j))


def train_raw_convNet(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                      number_of_cycle_for_first_training=2, number_of_cycles_rest_of_training=2,
                      path_weight_to_save_to="../weights"):
    participants_train, participants_validation, _ = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=512,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training)
    # participants_train, participants_validation, _, _, _, _ = load_dataloaders_training_sessions(
    #    examples_datasets_train, labels_datasets_train, batch_size=512, number_of_cycle_for_first_training=2)

    for participant_i in range(len(participants_train)):
        for session_j in range(len(participants_train[participant_i])):
            # Define Model
            # model = rawConvNet(number_of_class=11, number_of_blocks=3, dropout_rate=0.5, filter_size=filter_size,
            #                   number_of_features_output=[64, 64, 64]).cuda()
            model = TemporalConvNet(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()

            # Define Loss functions
            cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

            # Define Optimizer
            learning_rate = 0.001316
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


def train_TL_convNet(examples_datasets_train, labels_datasets_train, num_kernels, filter_size=(4, 10),
                     number_of_cycle_for_first_training=2, number_of_cycles_rest_of_training=2,
                     path_weight_to_save_to="../weights_TL", path_weights_normal_training="../weights_TL"):
    participants_train_for_source, participants_validation_for_source, _ = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=512,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, drop_last=False, ignore_first=True)
    participants_train_for_target, participants_validation_for_target, _ = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=512,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, drop_last=False, ignore_first=True)

    for participant_i in range(0, len(participants_train_for_target)):
        for session_j in range(1, len(participants_train_for_target[participant_i])):

            model_source = SourceNetwork(number_of_class=11, num_kernels=num_kernels, kernel_size=filter_size).cuda()
            # Define Loss functions
            cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
            # Define Optimizer
            learning_rate = 0.001316
            print(model_source.parameters())
            optimizer = optim.Adam(model_source.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=10,
                                                             verbose=True, eps=precision)
            if session_j == 1:
                # Fine-tune from the previous training
                model_source, optimizer, _, start_epoch = load_checkpoint(
                    model=model_source, optimizer=optimizer, scheduler=None,
                    filename=path_weights_normal_training +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, 0))
            else:
                model_source, optimizer, _, start_epoch = load_checkpoint(
                    model=model_source, optimizer=optimizer, scheduler=None,
                    filename=path_weight_to_save_to +
                             "/participant_%d/best_state_participant_pre_training_%d.pt" %
                             (participant_i, session_j - 1))

            list_train_dataloader = []
            list_validation_dataloader = []
            # Get all sessions before the current one and pre-train on these
            for k in range(0, session_j + 1):
                list_train_dataloader.append(participants_train_for_source[participant_i][k])
                list_validation_dataloader.append(participants_validation_for_source[participant_i][k])
            best_state_pre_training = pre_train_model(model=model_source, cross_entropy_loss=cross_entropy_loss,
                                                      optimizer_class=optimizer, scheduler=scheduler,
                                                      dataloaders={"train": list_train_dataloader,
                                                                   "val": list_validation_dataloader},
                                                      patience=20, patience_increase=20)

            if not os.path.exists(path_weight_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weight_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_state_pre_training, f=path_weight_to_save_to +
                                                  "/participant_%d/best_state_participant_pre_training_%d.pt"
                                                  % (participant_i, session_j))

            '''Train the source network with the current session'''
            state_pre_training = torch.load(path_weight_to_save_to +
                                            "/participant_%d/best_state_participant_pre_training_%d.pt" %
                                            (participant_i, session_j))
            weights_pre_training = state_pre_training['state_dict']
            model_target = TargetNetwork(weight_pre_trained_convNet=weights_pre_training, num_kernels=num_kernels,
                                         kernel_size=filter_size).cuda()

            # Define Loss functions
            cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
            # Define Optimizer
            learning_rate = 0.001316
            print(model_target.parameters())
            optimizer = optim.Adam(model_target.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)

            if session_j > 1:
                # Fine-tune from the previous training
                model_target, optimizer_classifier, _, start_epoch = load_checkpoint(
                    model=model_target, optimizer=optimizer, scheduler=None,
                    filename=path_weight_to_save_to +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, session_j - 1))
            else:
                # Fine-tune from the previous training
                model_target, _, _, start_epoch = load_checkpoint(
                    model=model_target, optimizer=None, scheduler=None,
                    filename=path_weights_normal_training +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, 0), strict=False)

            best_state = train_model_standard(model=model_target, criterion=cross_entropy_loss, optimizer=optimizer,
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
            torch.save(best_state, f=path_weight_to_save_to + "/participant_%d/"
                                                              "best_state_%d.pt" %
                                     (participant_i, session_j))
