import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

from LongTermClassificationMain.Models.model_utils import ExponentialMovingAverage, generate_task_examples

def train_model_standard(model, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8,
                         patience=10, patience_increase=10):
    since = time.time()

    best_loss = float('inf')

    best_weights = copy.deepcopy(model.state_dict())
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
                    best_weights = copy.deepcopy(model.state_dict())
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
    return best_weights


def train_batch_norm(model, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            # get the inputs
            inputs, labels = data

            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            model.train()
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)
    return copy.deepcopy(model.state_dict())

class DatasetWithPseudoLabels(data.Dataset):
    def __init__(self, dataset):
        list_images, list_labels, list_pseudo_labels = [], [], []
        for images, labels in dataset:
            list_images.extend(images)
            list_labels.extend(labels)
            list_pseudo_labels.extend(torch.ones(labels.size(0)))
        self._datalist = [{
            'image': list_images[ij],
            'label': list_labels[ij],
            'pseudolabel': list_pseudo_labels[ij]
        } for ij in range(len(list_labels))]


    def __len__(self):
        return len(self._datalist)

    def shuffledata(self):
        self._datalist = [self._datalist[ij] for ij in torch.randperm(len(self._datalist))]

    def __getitem__(self, index):
        return self._datalist[index]['image'], self._datalist[index]['label'], \
               self._datalist[index]['pseudolabel']

    def update_pseudolabels(self, pseudolabels):
        self._datalist = [{
            'image': self._datalist[ij]['image'],
            'label': self._datalist[ij]['label'],
            'pseudolabel': pseudolabels[ij]
        } for ij in range(len(self._datalist))]


def GenerateIteratorTarget(dataset, batch_size=512):
    params = {
        'pin_memory': True,
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 4,
        'drop_last': False,
    }

    return data.DataLoader(DatasetWithPseudoLabels(dataset), **params)


def generate_pseudolabels(classifier, train_dataset):
    classifier.eval()
    with torch.no_grad():
        'Pseudo labels de target example with the classifier'
        new_pseudolabels = []
        for images_source, _, _ in train_dataset:
            images_source = images_source.cuda()
            pred_source = classifier(images_source)
            _, predictions = torch.max(pred_source.cpu().data, 1)
            new_pseudolabels.extend(predictions)
        train_dataset.dataset.update_pseudolabels(new_pseudolabels)

    classifier.train()
    return train_dataset


def dirt_T_training(gesture_classifier, conditionalEntropyLoss, crossEntropyLoss, vatLoss, scheduler,
                    optimizer_classifier, train_dataset_source, validation_dataset_source, patience_increment=10,
                    max_epochs=500, gesture_classification_loss_weight=1e-2, source_VAT_weight=1e-2,
                    loss_cluster_assumption_weight=1e-2, batch_size=512, rate_pseudolabels_update=5):
    since = time.time()
    patience = 0 + patience_increment

    ''' Exponential moving average (simulating teacher model) '''
    ema = ExponentialMovingAverage(0.998)
    ema.register(gesture_classifier)

    best_loss = float("inf")
    best_weights = copy.deepcopy(gesture_classifier.state_dict())

    train_dataset = GenerateIteratorTarget(train_dataset_source, batch_size)
    train_dataset = generate_pseudolabels(gesture_classifier, train_dataset)
    for epoch in range(1, max_epochs):
        train_dataset.dataset.shuffledata()
        epoch_start = time.time()
        loss_main_sum, n_total = 0, 0
        loss_src_class_sum, loss_src_vat_sum, loss_cluster_assumption_sum = 0, 0, 0
        running_corrects, total_for_accuracy = 0, 0
        'TRAINING'
        gesture_classifier.train()
        for batch in train_dataset:
            inputs, _, pseudolabels = batch
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

        if loss_cluster_assumption / n_total < best_loss:
            print("New best loss without discrimination: ", loss_cluster_assumption / n_total)
            best_loss = loss_cluster_assumption / n_total
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment
        scheduler.step(loss_cluster_assumption / n_total)
        '''

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
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

        epoch_loss_validation = running_loss_validation / total_validation
        epoch_acc_validation = running_corrects_validation.item() / total_validation
        print('{} Loss: {:.8f} Acc: {:.8}'.format("VALIDATION", epoch_loss_validation, epoch_acc_validation))
        if epoch_loss_validation < best_loss:
            print("New best loss without discrimination: ", epoch_loss_validation)
            best_loss = epoch_loss_validation
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment
        scheduler.step(epoch_loss_validation)
        '''

        if patience < epoch:
            break

        if epoch % rate_pseudolabels_update == 0:
            train_dataset = generate_pseudolabels(classifier=gesture_classifier,  train_dataset=train_dataset)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_weights


def vadaTraining(gesture_classifier, conditionalEntropyLoss, crossEntropyLoss, vatLoss, scheduler,
                 optimizer_classifier, train_dataset_source, train_dataset_target, validation_dataset_source,
                 patience_increment=10, max_epochs=500, gesture_classification_loss_weight=1., domain_loss_weight=1e-2,
                 source_VAT_weight=1., target_VAT_weight=1e-2, loss_cluster_assumption_weight=1e-2):
    since = time.time()

    patience = 0 + patience_increment

    gesture_classifier.train()

    ''' Exponential moving average (simulating teacher model) '''
    ema = ExponentialMovingAverage(0.998)
    ema.register(gesture_classifier)

    best_loss = float("inf")
    best_weights = copy.deepcopy(gesture_classifier.state_dict())

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

            # pass inputs through the classifier network.
            pred_gesture_source, pred_domain_source, _, _ = gesture_classifier(input_source, get_all_tasks_output=True)
            pred_gesture_target, pred_domain_target, _, _ = gesture_classifier(input_target, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised source classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Conditional Entropy Loss. Following the cluster assumption (trying to not have decision frontier go
            # through clusters
            loss_cluster_assumption = conditionalEntropyLoss(pred_gesture_target)

            # Virtual Adversarial Loss. Trying to make the classifier not change rapidly its predictions for a small
            # change in the input
            loss_source_vat = vatLoss(input_source, pred_gesture_source)
            loss_target_vat = vatLoss(input_target, pred_gesture_target)

            # Try to be bad at the domain discrimination for the full network
            label_source_domain = torch.zeros(len(pred_domain_source), device='cuda', dtype=torch.long)
            label_target_domain = torch.ones(len(pred_domain_target), device='cuda', dtype=torch.long)
            loss_domain = 0.5 * (crossEntropyLoss(pred_domain_source, label_source_domain)
                                 + crossEntropyLoss(pred_domain_target, label_target_domain))

            # Combine all the loss of the classifier
            loss_main = (
                domain_loss_weight * loss_domain +
                gesture_classification_loss_weight * loss_source_class +
                source_VAT_weight * loss_source_vat +
                target_VAT_weight * loss_target_vat +
                loss_cluster_assumption_weight * loss_cluster_assumption
            )

            ' Update networks '
            # Update classifiers.
            optimizer_classifier.zero_grad()
            loss_main.backward()
            optimizer_classifier.step()

            # Polyak averaging
            ema(gesture_classifier)

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

        '''
        loss_without_discrimination = (
                gesture_classification_loss_weight * loss_src_class_sum +
                source_VAT_weight * loss_src_vat_sum +
                target_VAT_weight * loss_trg_vat_sum +
                loss_cluster_assumption_weight * loss_trg_cent_sum
            ) / n_total
        
        if loss_without_discrimination < best_loss:
            print("New best loss without discrimination: ", loss_without_discrimination)
            best_loss = loss_without_discrimination
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment
        '''



        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_validation = 0
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
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_weights


def DANN_Training(gesture_classifier, crossEntropyLoss, optimizer_classifier, train_dataset_source, scheduler,
                  train_dataset_target, validation_dataset_source, patience_increment=10, max_epochs=500,
                  gesture_classification_loss_weight=1., domain_loss_weight=1e-1):
    since = time.time()

    patience = 0 + patience_increment

    gesture_classifier.train()

    ''' Exponential moving average (simulating teacher model) '''
    #ema = ExponentialMovingAverage(0.998)
    #ema.register(gesture_classifier)

    best_loss = float("inf")
    best_weights = copy.deepcopy(gesture_classifier.state_dict())

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

            # pass inputs through the classifier network.
            pred_gesture_source, pred_domain_source, _, _ = gesture_classifier(input_source, get_all_tasks_output=True)
            _, pred_domain_target, _, _ = gesture_classifier(input_target, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised source classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Try to be bad at the domain discrimination for the full network
            label_source_domain = torch.zeros(len(pred_domain_source), device='cuda', dtype=torch.long)
            label_target_domain = torch.ones(len(pred_domain_target), device='cuda', dtype=torch.long)
            loss_domain = 0.5 * (crossEntropyLoss(pred_domain_source, label_source_domain)
                                 + crossEntropyLoss(pred_domain_target, label_target_domain))

            # Combine all the loss of the classifier
            loss_main = (
                domain_loss_weight * loss_domain +
                gesture_classification_loss_weight * loss_source_class
            )

            ' Update networks '
            # Update classifiers.
            optimizer_classifier.zero_grad()
            loss_main.backward()
            optimizer_classifier.step()

            # Polyak averaging
            #ema(gesture_classifier)

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
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_weights


def MTDANN_training(gesture_classifier, crossEntropyLoss, scheduler, optimizer_classifier, train_dataset_source,
                    train_dataset_target, validation_dataset_source, patience_increment=10, max_epochs=500,
                    gesture_classification_loss_weight=1., domain_loss_weight=1e-1,
                    shuffle_loss_weight=1., time_swap_loss_weight=1.):
    since = time.time()

    patience = 0 + patience_increment

    gesture_classifier.train()

    ''' Exponential moving average (simulating teacher model) '''
    ema = ExponentialMovingAverage(0.998)
    ema.register(gesture_classifier)

    best_loss = float("inf")
    best_weights = copy.deepcopy(gesture_classifier.state_dict())

    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        loss_main_sum, n_total = 0, 0
        loss_domain_sum, loss_src_class_sum = 0, 0
        running_corrects, running_correct_domain, total_for_accuracy, total_for_domain_accuracy = 0, 0, 0, 0

        loss_shuffling_sum, loss_time_swapped_sum = 0, 0
        running_correct_for_shuffling, total_for_shuffling = 0, 0
        running_correct_for_time_swapped, total_for_time_swapped = 0, 0

        'TRAINING'
        gesture_classifier.train()
        for source_batch, target_batch in zip(train_dataset_source, train_dataset_target):
            input_source, labels_source = source_batch
            input_source, labels_source = input_source.cuda(), labels_source.cuda()
            input_target, _ = target_batch
            input_target = input_target.cuda()

            # As we are combining the batch, get only half of each batch
            input_shuffled, label_shuffled, input_time_modified, label_time_modified = generate_task_examples(
                input_source[0:int(input_source.size(0) / 2)], input_target[0:int(input_target.size(0) / 2)])

            # pass inputs through the classifier network.
            pred_gesture_source, pred_domain_source, _, _ = gesture_classifier(input_source, get_all_tasks_output=True)
            pred_gesture_target, pred_domain_target, _, _ = gesture_classifier(input_target, get_all_tasks_output=True)

            # pass shuffled inputs through the classifier network
            #_, _, pred_shuffled, _ = gesture_classifier(input_shuffled, get_all_tasks_output=True)
            _, _, _, pred_time_swapped = gesture_classifier(input_time_modified, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised source classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Try to be bad at the domain discrimination for the full network
            label_source_domain = torch.zeros(len(pred_domain_source), device='cuda', dtype=torch.long)
            label_target_domain = torch.ones(len(pred_domain_target), device='cuda', dtype=torch.long)
            loss_domain = 0.5 * (crossEntropyLoss(pred_domain_source, label_source_domain)
                                 + crossEntropyLoss(pred_domain_target, label_target_domain))

            # Try to be good at detecting shuffling
            #loss_shuffling = crossEntropyLoss(pred_shuffled, label_shuffled)
            loss_time_swapped = crossEntropyLoss(pred_time_swapped, label_time_modified)

            # Combine all the loss of the classifier
            loss_main = (
                domain_loss_weight * loss_domain +
                gesture_classification_loss_weight * loss_source_class +
                #shuffle_loss_weight * loss_shuffling +
                time_swap_loss_weight * loss_time_swapped
            )

            ' Update networks '
            # Update classifiers.
            optimizer_classifier.zero_grad()
            loss_main.backward()
            optimizer_classifier.step()

            # Polyak averaging
            ema(gesture_classifier)

            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_source_class.item()
            #loss_shuffling_sum += loss_shuffling.item()
            loss_time_swapped_sum += loss_time_swapped.item()
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

            #_, shuffle_prediction = torch.max(pred_shuffled.data, 1)
            #running_correct_for_shuffling += torch.sum(shuffle_prediction == label_shuffled.data)
            #total_for_shuffling += label_shuffled.size(0)

            _, time_prediction = torch.max(pred_time_swapped.data, 1)
            running_correct_for_time_swapped += torch.sum(time_prediction == label_time_modified.data)
            total_for_time_swapped += label_time_modified.size(0)

        print('Accuracy source %4f,'
              ' accuracy domain distinction %4f'
              ' main loss classifier %4f,'
              ' loss domain distinction %4f,'
              ' source classification loss %4f,'
              #' accuracy shuffling %4f,'
              #' loss shuffling %4f,'
              ' accuracy Time Swapped %4f,'
              ' loss Time Swapped %4f,' %
              (running_corrects.item() / total_for_accuracy,
               running_correct_domain.item() / total_for_domain_accuracy,
               loss_main_sum / n_total,
               loss_domain_sum / n_total,
               loss_src_class_sum / n_total,
               #running_correct_for_shuffling.item() / total_for_shuffling,
               #loss_shuffling_sum / n_total,
               running_correct_for_time_swapped.item() / total_for_time_swapped,
               loss_time_swapped_sum / n_total
               ))

        '''
        loss_without_discrimination = (
                gesture_classification_loss_weight * loss_src_class_sum +
                source_VAT_weight * loss_src_vat_sum +
                target_VAT_weight * loss_trg_vat_sum +
                loss_cluster_assumption_weight * loss_trg_cent_sum +
                shuffle_loss_weight * loss_shuffling_sum +
                time_swap_loss_weight * loss_time_swapped_sum
            ) / n_total
        if loss_without_discrimination < best_loss:
            print("New best loss without discrimination: ", loss_without_discrimination)
            best_loss = loss_without_discrimination
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment
        '''

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_validation = 0
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
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_weights

def MTVADA_training(gesture_classifier, conditionalEntropyLoss, crossEntropyLoss, vatLoss, scheduler,
                    optimizer_classifier, train_dataset_source, train_dataset_target, validation_dataset_source,
                    patience_increment=10, max_epochs=500, gesture_classification_loss_weight=1., domain_loss_weight=1.,
                    source_VAT_weight=1., target_VAT_weight=1e-2, loss_cluster_assumption_weight=1e-2,
                    shuffle_loss_weight=1., time_swap_loss_weight=1.):
    since = time.time()

    patience = 0 + patience_increment

    gesture_classifier.train()

    ''' Exponential moving average (simulating teacher model) '''
    ema = ExponentialMovingAverage(0.998)
    ema.register(gesture_classifier)

    best_loss = float("inf")
    best_weights = copy.deepcopy(gesture_classifier.state_dict())

    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        loss_main_sum, n_total = 0, 0
        loss_domain_sum, loss_src_class_sum, loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
        running_corrects, running_correct_domain, total_for_accuracy, total_for_domain_accuracy = 0, 0, 0, 0

        loss_shuffling_sum, loss_time_swapped_sum = 0, 0
        running_correct_for_shuffling, total_for_shuffling = 0, 0
        running_correct_for_time_swapped, total_for_time_swapped = 0, 0

        'TRAINING'
        gesture_classifier.train()
        for source_batch, target_batch in zip(train_dataset_source, train_dataset_target):
            input_source, labels_source = source_batch
            input_source, labels_source = input_source.cuda(), labels_source.cuda()
            input_target, _ = target_batch
            input_target = input_target.cuda()

            # As we are combining the batch, get only half of each batch
            input_shuffled, label_shuffled, input_time_modified, label_time_modified = generate_task_examples(
                input_source[0:int(input_source.size(0) / 2)], input_target[0:int(input_target.size(0) / 2)])

            # pass inputs through the classifier network.
            pred_gesture_source, pred_domain_source, _, _ = gesture_classifier(input_source, get_all_tasks_output=True)
            pred_gesture_target, pred_domain_target, _, _ = gesture_classifier(input_target, get_all_tasks_output=True)

            # pass shuffled inputs through the classifier network
            _, _, pred_shuffled, _ = gesture_classifier(input_shuffled, get_all_tasks_output=True)
            _, _, _, pred_time_swapped = gesture_classifier(input_time_modified, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised source classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Conditional Entropy Loss. Following the cluster assumption (trying to not have decision frontier go
            # through clusters
            loss_cluster_assumption = conditionalEntropyLoss(pred_gesture_target)

            # Virtual Adversarial Loss. Trying to make the classifier not change rapidly its predictions for a small
            # change in the input
            loss_source_vat = vatLoss(input_source, pred_gesture_source)
            loss_target_vat = vatLoss(input_target, pred_gesture_target)

            # Try to be bad at the domain discrimination for the full network
            label_source_domain = torch.zeros(len(pred_domain_source), device='cuda', dtype=torch.long)
            label_target_domain = torch.ones(len(pred_domain_target), device='cuda', dtype=torch.long)
            loss_domain = 0.5 * (crossEntropyLoss(pred_domain_source, label_source_domain)
                                 + crossEntropyLoss(pred_domain_target, label_target_domain))

            # Try to be good at detecting shuffling
            loss_shuffling = crossEntropyLoss(pred_shuffled, label_shuffled)
            loss_time_swapped = crossEntropyLoss(pred_time_swapped, label_time_modified)

            # Combine all the loss of the classifier
            loss_main = (
                domain_loss_weight * loss_domain +
                gesture_classification_loss_weight * loss_source_class +
                source_VAT_weight * loss_source_vat +
                target_VAT_weight * loss_target_vat +
                loss_cluster_assumption_weight * loss_cluster_assumption +
                shuffle_loss_weight * loss_shuffling +
                time_swap_loss_weight * loss_time_swapped
            )

            ' Update networks '
            # Update classifiers.
            optimizer_classifier.zero_grad()
            loss_main.backward()
            optimizer_classifier.step()

            # Polyak averaging
            ema(gesture_classifier)

            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_source_class.item()
            loss_src_vat_sum += loss_source_vat.item()
            loss_trg_cent_sum += loss_cluster_assumption.item()
            loss_trg_vat_sum += loss_target_vat.item()
            loss_shuffling_sum += loss_shuffling.item()
            loss_time_swapped_sum += loss_time_swapped.item()
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

            _, shuffle_prediction = torch.max(pred_shuffled.data, 1)
            running_correct_for_shuffling += torch.sum(shuffle_prediction == label_shuffled.data)
            total_for_shuffling += label_shuffled.size(0)

            _, time_prediction = torch.max(pred_time_swapped.data, 1)
            running_correct_for_time_swapped += torch.sum(time_prediction == label_time_modified.data)
            total_for_time_swapped += label_time_modified.size(0)

        print('Accuracy source %4f,'
              ' accuracy domain distinction %4f'
              ' main loss classifier %4f,'
              ' loss domain distinction %4f,'
              ' source classification loss %4f,'
              ' source VAT %4f,'
              ' Target Cross Entropy %4f,'
              ' target VAT %4f,'
              ' accuracy shuffling %4f,'
              ' loss shuffling %4f,'
              ' accuracy Time Swapped %4f,'
              ' loss Time Swapped %4f,' %
              (running_corrects.item() / total_for_accuracy,
               running_correct_domain.item() / total_for_domain_accuracy,
               loss_main_sum / n_total,
               loss_domain_sum / n_total,
               loss_src_class_sum / n_total,
               loss_src_vat_sum / n_total,
               loss_trg_cent_sum / n_total,
               loss_trg_vat_sum / n_total,
               running_correct_for_shuffling.item() / total_for_shuffling,
               loss_shuffling_sum / n_total,
               running_correct_for_time_swapped.item() / total_for_time_swapped,
               loss_time_swapped_sum / n_total
               ))

        '''
        loss_without_discrimination = (
                gesture_classification_loss_weight * loss_src_class_sum +
                source_VAT_weight * loss_src_vat_sum +
                target_VAT_weight * loss_trg_vat_sum +
                loss_cluster_assumption_weight * loss_trg_cent_sum +
                shuffle_loss_weight * loss_shuffling_sum +
                time_swap_loss_weight * loss_time_swapped_sum
            ) / n_total
        if loss_without_discrimination < best_loss:
            print("New best loss without discrimination: ", loss_without_discrimination)
            best_loss = loss_without_discrimination
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment
        '''

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_validation = 0
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
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_weights


def MTDA_training(gesture_classifier, crossEntropyLoss, optimizer_classifier, train_dataset_source, scheduler,
                  train_dataset_target, validation_dataset_source, patience_increment=10, max_epochs=500,
                  gesture_classification_loss_weight=1., domain_loss_weight=1e-2, shuffle_loss_weight=1.,
                  time_swap_loss_weight=1.):
    since = time.time()

    patience = 0 + patience_increment

    gesture_classifier.train()

    ''' Exponential moving average (simulating teacher model) '''
    ema = ExponentialMovingAverage(0.998)
    ema.register(gesture_classifier)

    best_loss = float("inf")
    best_weights = copy.deepcopy(gesture_classifier.state_dict())

    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        loss_main_sum, n_total = 0, 0
        loss_domain_sum, loss_src_class_sum, loss_shuffling_sum, loss_time_swapped_sum = 0, 0, 0, 0
        running_corrects, running_correct_domain, total_for_accuracy, total_for_domain_accuracy = 0, 0, 0, 0
        running_correct_for_shuffling, total_for_shuffling = 0, 0
        running_correct_for_time_swapped, total_for_time_swapped = 0, 0

        'TRAINING'
        gesture_classifier.train()
        for source_batch, target_batch in zip(train_dataset_source, train_dataset_target):
            input_source, labels_source = source_batch
            input_source, labels_source = input_source.cuda(), labels_source.cuda()
            input_target, _ = target_batch
            input_target = input_target.cuda()

            # As we are combining the batch, get only half of each batch
            input_shuffled, label_shuffled, input_time_modified, label_time_modified = generate_task_examples(
                input_source[0:int(input_source.size(0)/2)], input_target[0:int(input_target.size(0)/2)])

            # pass inputs through the classifier network.
            pred_gesture_source, pred_domain_source, _, _ = gesture_classifier(input_source, get_all_tasks_output=True)
            _, pred_domain_target, _, _ = gesture_classifier(input_target, get_all_tasks_output=True)

            # pass shuffled inputs through the classifier network
            #_, _, pred_shuffled, _ = gesture_classifier(input_shuffled, get_all_tasks_output=True)
            _, _, _, pred_time_swapped = gesture_classifier(input_time_modified, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised source classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Try to be good at detecting shuffling
            #loss_shuffling = crossEntropyLoss(pred_shuffled, label_shuffled)
            loss_time_swapped = crossEntropyLoss(pred_time_swapped, label_time_modified)

            # Combine all the loss of the classifier
            loss_main = (
                gesture_classification_loss_weight * loss_source_class +
                #shuffle_loss_weight * loss_shuffling +
                time_swap_loss_weight * loss_time_swapped
            )

            ' Update networks '
            # Update classifiers.
            optimizer_classifier.zero_grad()
            loss_main.backward()
            optimizer_classifier.step()

            # Polyak averaging
            ema(gesture_classifier)

            loss_src_class_sum += loss_source_class.item()
            #loss_shuffling_sum += loss_shuffling.item()
            loss_time_swapped_sum += loss_time_swapped.item()
            loss_main_sum += loss_main.item()
            n_total += 1

            _, gestures_predictions_source = torch.max(pred_gesture_source.data, 1)
            running_corrects += torch.sum(gestures_predictions_source == labels_source.data)
            total_for_accuracy += labels_source.size(0)
            '''
            _, gestures_predictions_domain_source = torch.max(pred_domain_source.data, 1)
            _, gestures_predictions_domain_target = torch.max(pred_domain_target.data, 1)
            running_correct_domain += torch.sum(gestures_predictions_domain_source == label_source_domain.data)
            running_correct_domain += torch.sum(gestures_predictions_domain_target == label_target_domain.data)
            total_for_domain_accuracy += label_source_domain.size(0)
            total_for_domain_accuracy += label_target_domain.size(0)
            '''

            #_, shuffle_prediction = torch.max(pred_shuffled.data, 1)
            #running_correct_for_shuffling += torch.sum(shuffle_prediction == label_shuffled.data)
            #total_for_shuffling += label_shuffled.size(0)

            _, time_prediction = torch.max(pred_time_swapped.data, 1)
            running_correct_for_time_swapped += torch.sum(time_prediction == label_time_modified.data)
            total_for_time_swapped += label_time_modified.size(0)

        print('Accuracy source %4f,'
              ' main loss classifier %4f,'
              ' source classification loss %4f,'
              #' accuracy shuffling %4f,'
              #' loss shuffling %4f,'
              ' accuracy Time Swapped %4f,'
              ' loss Time Swapped %4f,'
              %
              (running_corrects.item() / total_for_accuracy,
               loss_main_sum / n_total,
               loss_src_class_sum / n_total,
               #running_correct_for_shuffling.item() / total_for_shuffling,
               #loss_shuffling_sum / n_total,
               running_correct_for_time_swapped.item() / total_for_time_swapped,
               loss_time_swapped_sum / n_total
               ))

        '''
        loss_main_without_descrimination = (
                gesture_classification_loss_weight * loss_src_class_sum +
                shuffle_loss_weight * loss_shuffling_sum +
                time_swap_loss_weight * loss_time_swapped_sum
            ) / n_total

        if loss_main_without_descrimination < best_loss:
            print("New best validation loss: ", loss_main_without_descrimination)
            best_loss = loss_main_without_descrimination
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment
        '''

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_validation = 0
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

        epoch_loss = running_loss_validation / n_total_validation
        epoch_acc = running_corrects_validation.item() / total_validation
        print('{} Loss: {:.8f} Acc: {:.8}'.format("VALIDATION", epoch_loss, epoch_acc))

        scheduler.step(running_loss_validation / n_total_validation)
        if running_loss_validation / n_total_validation < best_loss:
            print("New best validation loss: ", running_loss_validation / n_total_validation)
            best_loss = running_loss_validation / n_total_validation
            best_weights = copy.deepcopy(gesture_classifier.state_dict())
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_weights
