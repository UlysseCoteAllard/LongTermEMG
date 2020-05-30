import time
import copy

import torch


def SCADANN_BN_training(replay_dataset_train, target_validation_dataset, target_dataset, model, crossEntropyLoss,
                        optimizer_classifier, scheduler, patience_increment=10, max_epochs=500,
                        domain_loss_weight=1e-1):
    since = time.time()

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(2):
        state_dict = model.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batchNorm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    patience = 0 + patience_increment

    best_loss = float("inf")
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(),
                  'scheduler': scheduler.state_dict()}

    print("STARTING TRAINING")
    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        n_total = 0
        loss_main_sum, loss_domain_sum, loss_src_class_sum, loss_target_class_sum = 0, 0, 0, 0
        running_corrects_source, running_corrects_target, running_correct_domain = 0, 0, 0
        total_for_accuracy_source, total_for_accuracy_target, total_for_domain_accuracy = 0, 0, 0

        'TRAINING'
        model.train()
        # alpha = 1e-1*(1/epoch)+1e-1
        alpha = 0.
        for source_batch, target_batch in zip(replay_dataset_train, target_dataset):
            input_source, labels_source = source_batch
            input_source, labels_source = input_source.cuda(), labels_source.cuda()
            input_target, labels_target = target_batch
            input_target, labels_target = input_target.cuda(), labels_target.cuda()

            # Feed the inputs to the classifier network
            # Retrieves the BN weights calculated so far for the source dataset
            BN_weights = list_dictionaries_BN_weights[0]
            model.load_state_dict(BN_weights, strict=False)
            pred_source, pred_domain_source = model(input_source, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised/self-supervised gesture classification
            loss_source_class = crossEntropyLoss(pred_source, labels_source)

            # Try to be bad at the domain discrimination for the full network

            label_source_domain = torch.zeros(len(pred_domain_source), device='cuda', dtype=torch.long)
            loss_domain_source = ((1 - alpha) * crossEntropyLoss(pred_domain_source, label_source_domain))
            # Combine all the loss of the classifier
            loss_main_source = (0.5 * loss_source_class + domain_loss_weight * loss_domain_source)

            ' Update networks '
            # Update classifiers.
            # Zero the gradients
            optimizer_classifier.zero_grad()
            #loss_main_source.backward(retain_graph=True)
            loss_main_source.backward()
            optimizer_classifier.step()
            # Save the BN stats for the source
            state_dict = model.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[0] = copy.deepcopy(batch_norm_dict)
            # Load the BN statistics for the target
            BN_weights = copy.deepcopy(list_dictionaries_BN_weights[1])
            model.load_state_dict(BN_weights, strict=False)
            model.train()

            pred_target, pred_domain_target = model(input_target, get_all_tasks_output=True)
            loss_target_class = crossEntropyLoss(pred_target, labels_target)
            label_target_domain = torch.ones(len(pred_domain_target), device='cuda', dtype=torch.long)
            loss_domain_target = 0.5 * (crossEntropyLoss(pred_domain_target, label_target_domain))
            # Combine all the loss of the classifier
            loss_main_target = (0.5 * loss_target_class + domain_loss_weight * loss_domain_target)
            # Update classifiers.
            # Zero the gradients
            loss_main_target.backward()
            optimizer_classifier.step()

            # Save the BN stats for the target
            state_dict = model.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[1] = copy.deepcopy(batch_norm_dict)

            loss_main = loss_main_source + loss_main_target
            loss_domain = loss_domain_source + loss_domain_target

            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_source_class.item()
            loss_target_class_sum += loss_target_class.item()
            loss_target_class += loss_target_class.item()
            loss_main_sum += loss_main.item()
            n_total += 1

            _, gestures_predictions_source = torch.max(pred_source.data, 1)
            running_corrects_source += torch.sum(gestures_predictions_source == labels_source.data)
            total_for_accuracy_source += labels_source.size(0)

            _, gestures_predictions_target = torch.max(pred_target.data, 1)
            running_corrects_target += torch.sum(gestures_predictions_target == labels_target.data)
            total_for_accuracy_target += labels_target.size(0)

            _, gestures_predictions_domain_source = torch.max(pred_domain_source.data, 1)
            _, gestures_predictions_domain_target = torch.max(pred_domain_target.data, 1)
            running_correct_domain += torch.sum(gestures_predictions_domain_source == label_source_domain.data)
            running_correct_domain += torch.sum(gestures_predictions_domain_target == label_target_domain.data)
            total_for_domain_accuracy += label_source_domain.size(0)
            total_for_domain_accuracy += label_target_domain.size(0)

        accuracy_total = (running_corrects_source.item() + running_corrects_target.item()) / \
                         (total_for_accuracy_source + total_for_accuracy_target)
        print('Accuracy total %4f,'
              ' main loss classifier %4f,'
              ' source accuracy %4f'
              ' source classification loss %4f,'
              ' target accuracy %4f'
              ' target loss %4f'
              ' accuracy domain distinction %4f'
              ' loss domain distinction %4f,'
              %
              (accuracy_total,
               loss_main_sum / n_total,
               running_corrects_source.item() / total_for_accuracy_source,
               loss_src_class_sum / n_total,
               running_corrects_target.item() / total_for_accuracy_target,
               loss_target_class_sum / n_total,
               running_correct_domain.item() / total_for_domain_accuracy,
               loss_domain_sum / n_total
               ))

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_val = 0
        model.eval()
        for validation_batch in target_validation_dataset:
            # get the inputs
            inputs, labels = validation_batch
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer_classifier.zero_grad()

            with torch.no_grad():
                # forward
                outputs = model(inputs)
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

        loss_val = running_loss_validation / n_total_val
        scheduler.step(loss_val)
        if loss_val < best_loss:
            print("New best validation loss: ", loss_val)
            best_loss = loss_val
            # Load the BN statistics for the target
            BN_weights = copy.deepcopy(list_dictionaries_BN_weights[1])
            model.load_state_dict(BN_weights, strict=False)
            best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                          'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
            patience = epoch + patience_increment

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

        if patience < epoch:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_state