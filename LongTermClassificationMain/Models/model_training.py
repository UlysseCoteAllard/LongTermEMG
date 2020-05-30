import time
import copy
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset

from LongTermClassificationMain.Models.model_utils import ExponentialMovingAverage


def train_model_standard(model, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8,
                         patience=10, patience_increase=10):
    since = time.time()

    best_loss = float('inf')

    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data

                inputs, labels = inputs.cuda(), labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    model.train()
                    # forward
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()

                else:
                    model.eval()
                    with torch.no_grad():

                        # forward
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs.data, 1)

                        loss = criterion(outputs, labels)
                        loss = loss.item()

                # statistics
                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total
            print('{} Loss: {:.8f} Acc: {:.8}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_state = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state


# See: https://discuss.pytorch.org/t/how-to-run-the-model-to-only-update-the-batch-normalization-statistics/20626) for
# an explanation of why this update the batch norms statistics.
def AdaBN_adaptation(model, dataloader, optimizer_classifier, scheduler, epochs=5):
    model.train()

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            # get the inputs
            inputs, labels = data

            inputs, labels = inputs.cuda(), labels.cuda()
            # The running stats are updated during the forward pass if the model is in training mode.
            # This is the only adaptation performed by AdaBN while working with the unlabeled data.
            model.train()
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

    # Note that we do not need in principle optimizer_classifier, scheduler for the AdaBN adaptation
    # We save them (an not modify them) to be consistent with the interface of the other methods.
    best_state = {'epoch': epochs, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
    return best_state


def generate_new_dataset_for_dirt(dataset, classifier, batch_size=512):
    examples_pseudo = []
    labels_pseudo = []
    classifier.eval()
    for batch in dataset:
        with torch.no_grad():
            inputs, _ = batch
            inputs = inputs.cuda()
            pred_source = classifier(inputs)
            _, predictions = torch.max(pred_source.cpu().data, 1)
            labels_pseudo.extend(predictions.numpy())
            examples_pseudo.extend(inputs.cpu().numpy())
    train_replay = TensorDataset(torch.from_numpy(np.array(examples_pseudo, dtype=np.float32)),
                                 torch.from_numpy(np.array(labels_pseudo, dtype=np.int64)))
    pseudo_dataloader = torch.utils.data.DataLoader(train_replay, batch_size=batch_size, shuffle=True, drop_last=False)
    classifier.train()
    return pseudo_dataloader


def dirt_T_training(gesture_classifier, conditionalEntropyLoss, crossEntropyLoss, vatLoss, scheduler,
                    optimizer_classifier, train_dataset_source, patience_increment=10,
                    max_epochs=500, gesture_classification_loss_weight=1e-2, source_VAT_weight=1e-2,
                    loss_cluster_assumption_weight=1e-2, batch_size=512, rate_pseudolabels_update=5):
    since = time.time()
    patience = 0 + patience_increment

    ''' Exponential moving average (simulating teacher model) '''
    ema = ExponentialMovingAverage(0.998)
    ema.register(gesture_classifier)

    best_loss = float("inf")
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}

    train_dataset = generate_new_dataset_for_dirt(train_dataset_source, gesture_classifier, batch_size=batch_size)
    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        loss_main_sum, n_total = 0, 0
        loss_src_class_sum, loss_src_vat_sum, loss_cluster_assumption_sum = 0, 0, 0
        running_corrects, total_for_accuracy = 0, 0
        'TRAINING'
        gesture_classifier.train()
        for batch in train_dataset:
            inputs, pseudolabels = batch
            inputs, pseudolabels = inputs.cuda(), pseudolabels.cuda()

            preds = gesture_classifier(inputs)
            _, values_predictions = torch.max(preds.data, 1)

            'Classifier losses setup.'
            # Supervised source classification on the pseudolabels
            loss_source_class = crossEntropyLoss(preds, pseudolabels)

            # Conditional Entropy Loss. Following the cluster assumption (trying to not have decision frontier go
            # through clusters
            loss_cluster_assumption = conditionalEntropyLoss(preds)

            # Virtual Adversarial Loss. Trying to make the classifier not change rapidly its predictions for a small
            # change in the input
            loss_source_vat = vatLoss(inputs, preds)

            # Combine all the loss of the classifier
            loss_main = (
                    gesture_classification_loss_weight * loss_source_class +
                    source_VAT_weight * loss_source_vat +
                    loss_cluster_assumption_weight * loss_cluster_assumption
            )

            ' Update networks '
            # Update classifier.
            optimizer_classifier.zero_grad()
            loss_main.backward()
            optimizer_classifier.step()

            loss_cluster_assumption_sum += loss_cluster_assumption.item()
            loss_src_class_sum += loss_source_class.item()
            loss_src_vat_sum += loss_source_vat.item()
            loss_main_sum += loss_main.item()
            n_total += 1

            _, gestures_predictions_source = torch.max(preds.data, 1)
            running_corrects += torch.sum(gestures_predictions_source == pseudolabels.data)
            total_for_accuracy += pseudolabels.size(0)

            # Polyak averaging
            ema(gesture_classifier)

        print(' main loss classifier %4f,'
              ' source classification loss %4f,'
              ' source VAT %4f,' %
              (loss_main_sum / n_total,
               loss_src_class_sum / n_total,
               loss_src_vat_sum / n_total))

        print('Accuracy source %4f,'
              ' main loss classifier %4f,'
              ' source classification loss %4f,'
              ' source VAT %4f,'
              ' Target Cross Entropy %4f,' %
              (running_corrects.item() / total_for_accuracy,
               loss_main_sum / n_total,
               loss_src_class_sum / n_total,
               loss_src_vat_sum / n_total,
               loss_cluster_assumption_sum / n_total))

        if loss_cluster_assumption_sum / n_total < best_loss:
            print("New best loss without discrimination: ", loss_cluster_assumption_sum / n_total)
            best_loss = loss_cluster_assumption_sum / n_total
            best_state = {'epoch': epoch, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                          'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
            patience = epoch + patience_increment
        scheduler.step(loss_cluster_assumption_sum / n_total)

        if patience < epoch:
            break

        if epoch % rate_pseudolabels_update == 0:
            train_dataset = generate_new_dataset_for_dirt(train_dataset_source, gesture_classifier,
                                                          batch_size=batch_size)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_state


def vada_BN_Training(gesture_classifier, conditionalEntropyLoss, crossEntropyLoss, vatLoss, scheduler,
                     optimizer_classifier, train_dataset_source, train_dataset_target, validation_dataset_source,
                     patience_increment=10, max_epochs=500, gesture_classification_loss_weight=1.,
                     domain_loss_weight=1e-2,
                     source_VAT_weight=1., target_VAT_weight=1e-2, loss_cluster_assumption_weight=1e-2):
    since = time.time()
    patience = 0 + patience_increment

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(2):
        state_dict = gesture_classifier.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batchNorm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    gesture_classifier.train()

    ''' Exponential moving average (simulating teacher model) '''
    ema = ExponentialMovingAverage(0.998)
    ema.register(gesture_classifier)

    best_loss = float("inf")
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}

    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        loss_main_sum, n_total = 0, 0
        loss_domain_sum, loss_src_class_sum, loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
        running_corrects, running_correct_domain, total_for_accuracy, total_for_domain_accuracy = 0, 0, 0, 0

        'TRAINING'
        gesture_classifier.train()
        for source_batch, target_batch in zip(train_dataset_source, train_dataset_target):
            input_source, labels_source = source_batch
            input_source, labels_source = input_source.cuda(), labels_source.cuda()
            input_target, _ = target_batch
            input_target = input_target.cuda()

            # Feed the inputs to the classifier network
            # Retrieves the BN weights calculated so far for the source dataset
            BN_weights = list_dictionaries_BN_weights[0]
            gesture_classifier.load_state_dict(BN_weights, strict=False)

            # pass inputs through the classifier network.
            pred_gesture_source, pred_domain_source = gesture_classifier(input_source, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised source classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Virtual Adversarial Loss. Trying to make the classifier not change rapidly its predictions for a small
            # change in the input
            loss_source_vat = vatLoss(input_source, pred_gesture_source)

            # Try to be bad at the domain discrimination for the full network
            label_source_domain = torch.zeros(len(pred_domain_source), device='cuda', dtype=torch.long)
            loss_domain_source = 0.5 * crossEntropyLoss(pred_domain_source, label_source_domain)

            # Combine all the loss of the classifier
            loss_source = (
                    domain_loss_weight * loss_domain_source +
                    gesture_classification_loss_weight * loss_source_class +
                    source_VAT_weight * loss_source_vat
            )

            ' Update networks '
            # Update classifiers.
            optimizer_classifier.zero_grad()
            # loss_source.backward(retain_graph=True)
            loss_source.backward()
            optimizer_classifier.step()
            # Save the BN stats for the source
            state_dict = gesture_classifier.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[0] = copy.deepcopy(batch_norm_dict)
            # Load the BN statistics for the target
            BN_weights = copy.deepcopy(list_dictionaries_BN_weights[1])
            gesture_classifier.load_state_dict(BN_weights, strict=False)
            gesture_classifier.train()

            pred_gesture_target, pred_domain_target = gesture_classifier(input_target, get_all_tasks_output=True)
            # Conditional Entropy Loss. Following the cluster assumption (trying to not have decision frontier go
            # through clusters
            loss_cluster_assumption = conditionalEntropyLoss(pred_gesture_target)
            loss_target_vat = vatLoss(input_target, pred_gesture_target)
            label_target_domain = torch.ones(len(pred_domain_target), device='cuda', dtype=torch.long)
            loss_domain_target = 0.5 * crossEntropyLoss(pred_domain_target, label_target_domain)

            loss_target = (
                    domain_loss_weight * loss_domain_target +
                    target_VAT_weight * loss_target_vat +
                    loss_cluster_assumption_weight * loss_cluster_assumption
            )

            # Save the BN stats for the target
            state_dict = gesture_classifier.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[1] = copy.deepcopy(batch_norm_dict)
            BN_weights = list_dictionaries_BN_weights[0]
            gesture_classifier.load_state_dict(BN_weights, strict=False)

            # Polyak averaging
            ema(gesture_classifier)
            loss_main = loss_source + loss_target
            loss_domain = loss_domain_source + loss_domain_target
            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_source_class.item()
            loss_src_vat_sum += loss_source_vat.item()
            loss_trg_cent_sum += loss_cluster_assumption.item()
            loss_trg_vat_sum += loss_target_vat.item()
            loss_main_sum += loss_main.item()
            n_total += 1

            _, gestures_predictions_source = torch.max(pred_gesture_source.data, 1)
            running_corrects += torch.sum(gestures_predictions_source == labels_source.data)
            total_for_accuracy += labels_source.size(0)

            _, gestures_predictions_domain_source = torch.max(pred_domain_source.data, 1)
            _, gestures_predictions_domain_target = torch.max(pred_domain_target.data, 1)
            running_correct_domain += torch.sum(gestures_predictions_domain_source == label_source_domain.data)
            running_correct_domain += torch.sum(gestures_predictions_domain_target == label_target_domain.data)
            total_for_domain_accuracy += label_source_domain.size(0)
            total_for_domain_accuracy += label_target_domain.size(0)

        print('Accuracy source %4f,'
              ' accuracy domain distinction %4f'
              ' main loss classifier %4f,'
              ' loss domain distinction %4f,'
              ' source classification loss %4f,'
              ' source VAT %4f,'
              ' Target Conditional Entropy %4f,'
              ' target VAT %4f,' %
              (running_corrects.item() / total_for_accuracy,
               running_correct_domain.item() / total_for_domain_accuracy,
               loss_main_sum / n_total,
               loss_domain_sum / n_total,
               loss_src_class_sum / n_total,
               loss_src_vat_sum / n_total,
               loss_trg_cent_sum / n_total,
               loss_trg_vat_sum / n_total))

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_validation = 0
        BN_weights = copy.deepcopy(list_dictionaries_BN_weights[0])
        gesture_classifier.load_state_dict(BN_weights, strict=False)
        gesture_classifier.eval()
        for validation_batch in validation_dataset_source:
            # get the inputs
            inputs, labels = validation_batch

            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer_classifier.zero_grad()

            with torch.no_grad():
                # forward
                outputs = gesture_classifier(inputs)
                _, predictions = torch.max(outputs.data, 1)

                loss = crossEntropyLoss(outputs, labels)
                loss = loss.item()

                # statistics
                running_loss_validation += loss
                running_corrects_validation += torch.sum(predictions == labels.data)
                total_validation += labels.size(0)
                n_total_validation += 1

        epoch_loss = running_loss_validation / total_validation
        epoch_acc = running_corrects_validation.item() / total_validation
        print('{} Loss: {:.8f} Acc: {:.8}'.format("VALIDATION", epoch_loss, epoch_acc))

        scheduler.step(running_loss_validation / n_total_validation)
        if running_loss_validation / n_total_validation < best_loss:
            print("New best validation loss: ", running_loss_validation / n_total_validation)
            best_loss = running_loss_validation / n_total_validation
            best_state = {'epoch': epoch, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                          'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_state


def DANN_BN_Training(gesture_classifier, crossEntropyLoss, optimizer_classifier, train_dataset_source, scheduler,
                     train_dataset_target, validation_dataset_source, patience_increment=10, max_epochs=500,
                     gesture_classification_loss_weight=1., domain_loss_weight=1e-1):
    since = time.time()
    patience = 0 + patience_increment

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(2):
        state_dict = gesture_classifier.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batchNorm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    best_loss = float("inf")
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}

    print("STARTING TRAINING")
    for epoch in range(1, max_epochs):
        epoch_start = time.time()

        loss_main_sum, n_total = 0, 0
        loss_domain_sum, loss_src_class_sum, loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
        running_corrects, running_correct_domain, total_for_accuracy, total_for_domain_accuracy = 0, 0, 0, 0

        'TRAINING'
        gesture_classifier.train()
        for source_batch, target_batch in zip(train_dataset_source, train_dataset_target):

            input_source, labels_source = source_batch
            input_source, labels_source = input_source.cuda(), labels_source.cuda()
            input_target, _ = target_batch
            input_target = input_target.cuda()

            # Feed the inputs to the classifier network
            # Retrieves the BN weights calculated so far for the source dataset
            BN_weights = list_dictionaries_BN_weights[0]
            gesture_classifier.load_state_dict(BN_weights, strict=False)
            pred_gesture_source, pred_domain_source = gesture_classifier(input_source, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised/self-supervised gesture classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Try to be bad at the domain discrimination for the full network

            label_source_domain = torch.zeros(len(pred_domain_source), device='cuda', dtype=torch.long)
            loss_domain_source = crossEntropyLoss(pred_domain_source, label_source_domain)
            # Combine all the loss of the classifier
            loss_main_source = (0.5 * loss_source_class + domain_loss_weight * loss_domain_source)

            ' Update networks '
            # Update classifiers.
            # Zero the gradients
            optimizer_classifier.zero_grad()
            # loss_main_source.backward(retain_graph=True)
            loss_main_source.backward()
            optimizer_classifier.step()
            # Save the BN stats for the source
            state_dict = gesture_classifier.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[0] = copy.deepcopy(batch_norm_dict)

            _, pred_domain_target = gesture_classifier(input_target, get_all_tasks_output=True)
            label_target_domain = torch.ones(len(pred_domain_target), device='cuda', dtype=torch.long)
            loss_domain_target = 0.5 * (crossEntropyLoss(pred_domain_target, label_target_domain))
            # Combine all the loss of the classifier
            loss_domain_target = 0.5 * domain_loss_weight * loss_domain_target
            # Update classifiers.
            # Zero the gradients
            loss_domain_target.backward()
            optimizer_classifier.step()

            # Save the BN stats for the target
            state_dict = gesture_classifier.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[1] = copy.deepcopy(batch_norm_dict)

            loss_main = loss_main_source + loss_domain_target
            loss_domain = loss_domain_source + loss_domain_target

            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_source_class.item()
            loss_main_sum += loss_main.item()
            n_total += 1

            _, gestures_predictions_source = torch.max(pred_gesture_source.data, 1)
            running_corrects += torch.sum(gestures_predictions_source == labels_source.data)
            total_for_accuracy += labels_source.size(0)

            _, gestures_predictions_domain_source = torch.max(pred_domain_source.data, 1)
            _, gestures_predictions_domain_target = torch.max(pred_domain_target.data, 1)
            running_correct_domain += torch.sum(gestures_predictions_domain_source == label_source_domain.data)
            running_correct_domain += torch.sum(gestures_predictions_domain_target == label_target_domain.data)
            total_for_domain_accuracy += label_source_domain.size(0)
            total_for_domain_accuracy += label_target_domain.size(0)

        print('Accuracy source %4f,'
              ' main loss classifier %4f,'
              ' source classification loss %4f,'
              ' loss domain distinction %4f,'
              ' accuracy domain distinction %4f'
              %
              (running_corrects.item() / total_for_accuracy,
               loss_main_sum / n_total,
               loss_src_class_sum / n_total,
               loss_domain_sum / n_total,
               running_correct_domain.item() / total_for_domain_accuracy
               ))

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_val = 0

        # BN_weights = copy.deepcopy(list_dictionaries_BN_weights[0])
        # gesture_classifier.load_state_dict(BN_weights, strict=False)
        gesture_classifier.eval()
        for validation_batch in validation_dataset_source:
            # get the inputs
            inputs, labels = validation_batch

            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer_classifier.zero_grad()

            with torch.no_grad():
                # forward
                outputs = gesture_classifier(inputs)
                _, predictions = torch.max(outputs.data, 1)

                loss = crossEntropyLoss(outputs, labels)
                loss = loss.item()

                # statistics
                running_loss_validation += loss
                running_corrects_validation += torch.sum(predictions == labels.data)
                total_validation += labels.size(0)
                n_total_val += 1

        epoch_loss = running_loss_validation / n_total_val
        epoch_acc = running_corrects_validation.item() / total_validation
        print('{} Loss: {:.8f} Acc: {:.8}'.format("VALIDATION", epoch_loss, epoch_acc))

        scheduler.step(running_loss_validation / n_total_val)
        if running_loss_validation / n_total_val < best_loss:
            print("New best validation loss: ", running_loss_validation / n_total_val)
            best_loss = running_loss_validation / n_total_val
            BN_weights = copy.deepcopy(list_dictionaries_BN_weights[1])
            gesture_classifier.load_state_dict(BN_weights, strict=False)
            best_state = {'epoch': epoch, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                          'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_state
