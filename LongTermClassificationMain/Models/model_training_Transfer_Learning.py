import copy
import time
import numpy as np

import torch


def get_inputs_domain(dataloaders, current_domain_index):
    random_vector_index_domain = np.arange(len(dataloaders))
    np.random.shuffle(random_vector_index_domain)
    for i in random_vector_index_domain:
        if i != current_domain_index:
            for _, data in enumerate(dataloaders[i]):
                inputs, labels = data
                return inputs, i


def pre_train_model(model, cross_entropy_loss, optimizer_class, scheduler,
                    dataloaders, num_epochs=500, precision=1e-8, weight_domain_loss=1e-1, patience=20,
                    patience_increase=20):
    since = time.time()
    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(len(dataloaders['train'])):
        state_dict = model.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batch_norm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    best_loss = float('inf')
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer_class.state_dict(), 'scheduler': scheduler.state_dict()}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        'Training'
        loss_main_sum, n_total = 0, 0
        loss_domain_sum, loss_source_class_sum = 0, 0
        accuracy_source_class_sum, accuracy_source_domain, accuracy_target_domain = 0., 0., 0.

        # Get a random order for the training dataset
        random_vec = np.arange(len(dataloaders['train']))
        np.random.shuffle(random_vec)

        lambda_value = 1.
        for dataset_index in random_vec:
            # Retrieves the BN weights calculated so far for this dataset
            BN_weights = list_dictionaries_BN_weights[dataset_index]
            model.load_state_dict(BN_weights, strict=False)
            model.train()
            for i, data in enumerate(dataloaders['train'][dataset_index], 0):
                # get the source inputs
                inputs_source, labels_source = data
                inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

                # pass the source input through the model
                output_class_source, output_domain_source = model(inputs_source, lambda_value)
                loss_classification_source = cross_entropy_loss(output_class_source, labels_source)
                optimizer_class.zero_grad()
                if len(dataloaders['train']) > 1:
                    loss_domain_discrimination_source = 0.5 * cross_entropy_loss(output_domain_source, torch.zeros(len(
                        output_domain_source),
                        device='cuda',
                        dtype=torch.long))

                    loss_source = loss_classification_source + weight_domain_loss * loss_domain_discrimination_source
                else:
                    loss_source = loss_classification_source
                if len(dataloaders['train']) <= 1:
                    loss_source.backward()
                    optimizer_class.step()

                # Save the BN statistics for this dataset
                state_dict = model.state_dict()
                batch_norm_dict = {}
                for key in state_dict:
                    if "batch_norm" in key:
                        batch_norm_dict.update({key: state_dict[key]})
                list_dictionaries_BN_weights[dataset_index] = copy.deepcopy(batch_norm_dict)
                'Handle the target data (if there is more than one session data being use' \
                '(i.e. the data from the other training session than the one currently being used)'
                if len(dataloaders['train']) > 1:
                    # get the target input
                    inputs_from_adversarial_domain, adversarial_domain_index = get_inputs_domain(dataloaders['train'],
                                                                                                 current_domain_index=
                                                                                                 dataset_index)
                    # Load the BN statistics for the domain
                    BN_weights = copy.deepcopy(list_dictionaries_BN_weights[adversarial_domain_index])
                    model.load_state_dict(BN_weights, strict=False)
                    model.train()

                    # pass the target input through the model
                    _, output_domain_target = model(inputs_from_adversarial_domain.cuda(), lambda_value)

                    loss_domain_discrimination_target = 0.5 * (
                        cross_entropy_loss(output_domain_target, torch.ones(len(output_domain_target), device='cuda',
                                                                            dtype=torch.long)))

                    loss_target = weight_domain_loss * loss_domain_discrimination_target

                    loss_main = loss_source + loss_target

                    # Get back to the right BN statistics for the current domain
                    BN_weights = list_dictionaries_BN_weights[dataset_index]
                    model.load_state_dict(BN_weights, strict=False)
                    loss_main.backward()
                    optimizer_class.step()

                    # Track the losses
                    loss_main_sum += loss_source.item() + loss_target.item()
                    loss_source_class_sum += loss_classification_source.item()
                    loss_domain_sum += (loss_domain_discrimination_source.item() +
                                        loss_domain_discrimination_target.item())
                    # Track the accuracies
                    _, predictions_class = torch.max(output_class_source.data, 1)
                    accuracy_source_class_sum += torch.sum(
                        predictions_class == labels_source).item() / labels_source.size(0)
                    _, predictions_domain_source = torch.max(output_domain_source.data, 1)
                    accuracy_source_domain += torch.sum(predictions_domain_source ==
                                                        torch.zeros_like(predictions_domain_source, device='cuda',
                                                                         dtype=torch.long)
                                                        ).item() / predictions_domain_source.size(0)
                    _, prediction_domain_target = torch.max(output_domain_target.data, 1)
                    accuracy_target_domain += torch.sum(prediction_domain_target == torch.ones_like(
                        prediction_domain_target, device='cuda', dtype=torch.long)
                                                        ).item() / prediction_domain_target.size(0)
                    n_total += 1
                else:
                    # Track the losses
                    loss_source_class_sum += loss_classification_source.item()
                    # Track the accuracies
                    _, predictions_class = torch.max(output_class_source.data, 1)
                    accuracy_source_class_sum += torch.sum(
                        predictions_class == labels_source).item() / labels_source.size(0)
                    _, predictions_domain_source = torch.max(output_domain_source.data, 1)
                    accuracy_source_domain += torch.sum(predictions_domain_source ==
                                                        torch.zeros_like(predictions_domain_source, device='cuda',
                                                                         dtype=torch.long)
                                                        ).item() / predictions_domain_source.size(0)

                    n_total += 1

        if len(dataloaders['train']) > 1:
            print("Training phase: Main Loss: {:.5f}, Accuracy class: {:.5f}, Loss class {:.5f},"
                  " Accuracy domain source: {:.5f}, Accuracy domain target: {:.5f}, Loss domain: {:.5f}"
                  "".format(loss_main_sum / n_total, accuracy_source_class_sum / n_total,
                            loss_source_class_sum / n_total,
                            accuracy_source_domain / n_total, accuracy_target_domain / n_total,
                            loss_domain_sum / n_total))
        else:
            print("Training phase: Accuracy class: {:.5f}, Loss class {:.5f},"
                  "".format(accuracy_source_class_sum / n_total, loss_source_class_sum / n_total))
        'Validation step'
        if epoch % 1 == 0:
            with torch.no_grad():
                prediction_accuracy = 0.
                validation_loss, n_total = 0, 0
                for dataset_index, dataset in enumerate(dataloaders['val']):
                    BN_weights = list_dictionaries_BN_weights[dataset_index]
                    model.load_state_dict(BN_weights, strict=False)
                    model.eval()
                    for i, data in enumerate(dataset):
                        inputs, labels = data
                        inputs, labels = inputs.cuda(), labels.cuda()
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)
                        prediction_accuracy += torch.sum(predictions ==
                                                         labels).item() / labels.size(0)
                        validation_loss += cross_entropy_loss(outputs, labels)
                        n_total += 1

                print("\nValidation Accuracy: {:.5f}, Loss Accuracy {:.5f}\n".format(
                    prediction_accuracy / n_total, validation_loss.item() / n_total
                ))
                epoch_validation_loss = validation_loss.item() / n_total

                # deep copy the model
                scheduler.step(epoch_validation_loss)
                if epoch_validation_loss + precision < best_loss:
                    print("New best validation loss:", epoch_validation_loss)
                    best_loss = epoch_validation_loss
                    best_state = {'epoch': epoch, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer_class.state_dict(), 'scheduler': scheduler.state_dict()}
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
