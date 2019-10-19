import time
import copy
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


def get_domain_index_for_batch(batch_size, datasets_segregated_by_domain, domain_that_can_be_selected_for_training,
                               list_current_index_for_each_domain):
    domain_index_to_create_batch_from = np.random.choice(domain_that_can_be_selected_for_training)
    while True:
        index_after_batch = list_current_index_for_each_domain[domain_index_to_create_batch_from] + \
                            batch_size
        if index_after_batch >= len(datasets_segregated_by_domain[domain_index_to_create_batch_from]):
            if len(domain_that_can_be_selected_for_training) > 1:
                domain_that_can_be_selected_for_training = domain_that_can_be_selected_for_training[
                    domain_that_can_be_selected_for_training != domain_index_to_create_batch_from]
                domain_index_to_create_batch_from = np.random.choice(
                    domain_that_can_be_selected_for_training)
            else:
                return None
        return domain_index_to_create_batch_from


def create_source_and_target_batch(batch_size, datasets_segregated_by_domain, domain_index_to_create_batch_from,
                                   labels_segregated_by_domain, list_current_index_for_each_domain,
                                   list_index_shuffle_segregated_by_domains):
    # Create the batch for the source domain
    examples_source, labels_source = [], []
    for i in range(list_current_index_for_each_domain[domain_index_to_create_batch_from],
                   list_current_index_for_each_domain[domain_index_to_create_batch_from] +
                   batch_size):
        if i >= len(list_index_shuffle_segregated_by_domains[domain_index_to_create_batch_from]):
            break
        index_to_add = list_index_shuffle_segregated_by_domains[domain_index_to_create_batch_from][i]
        examples_source.append(datasets_segregated_by_domain[domain_index_to_create_batch_from][index_to_add])
        labels_source.append(labels_segregated_by_domain[domain_index_to_create_batch_from][index_to_add])
    list_current_index_for_each_domain[domain_index_to_create_batch_from] += batch_size
    # Create the batch for the target domain
    examples_target = []
    domain_to_make_target_from = np.arange(len(datasets_segregated_by_domain))
    domain_to_make_target_from = domain_to_make_target_from[domain_to_make_target_from !=
                                                            domain_index_to_create_batch_from]
    number_of_examples_per_domain_to_take = math.ceil(batch_size / len(datasets_segregated_by_domain))
    while len(examples_target) < batch_size:
        domain_to_take_example_from_for_target = np.random.choice(domain_to_make_target_from)
        # Remove that domain from the choice to not have a repeat
        domain_to_make_target_from = domain_to_make_target_from[domain_to_make_target_from !=
                                                                domain_index_to_create_batch_from]
        index_examples_to_choose_from = np.arange(len(datasets_segregated_by_domain[
                                                          domain_to_take_example_from_for_target]))
        examples_index_to_add = np.random.choice(index_examples_to_choose_from,
                                                 number_of_examples_per_domain_to_take,
                                                 replace=False)
        for j in examples_index_to_add:
            examples_target.append(datasets_segregated_by_domain[
                                       domain_to_take_example_from_for_target][j])
            if len(examples_target) >= batch_size:
                break

    return examples_source, labels_source, examples_target

"""
The batch size must be bigger than the number of domains, and optimally should be divisible by the number of domains
"""
"""
def pre_train_model(convNet, criterion_class, criterion_domain, optimizer, scheduler, examples_segregated_by_domain,
                    labels_segregated_by_domain, examples_valid_segregated_by_domain,
                    labels_valid_segregated_by_domain, batch_size, num_epochs=100, precision=1e-8):
    for examples in examples_segregated_by_domain:
        print(np.shape(examples))
        assert len(examples) > batch_size
    since = time.time()

    best_loss = float('inf')

    best_weights = copy.deepcopy(convNet.state_dict())

    patience = 30
    patience_increase = 30
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        if epoch < 30:
            alpha = 0.
        else:
            p = float(epoch-30) / (num_epochs + 30)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

        lambda_value = alpha
        print("Alpha : ", alpha)
        for phase in ['train', 'valid']:
            if "train" in phase:
                running_loss_class = 0.
                running_loss_domain_source = 0.
                running_loss_domain_target = 0.
                running_corrects_class = 0.
                running_corrects_source_domain = 0
                running_corrects_target_domain = 0
                total = 0


                convNet.train(True)  # Set model to training mode

                # For each domain,
                domain_that_can_be_selected_for_training = np.arange(len(examples_segregated_by_domain))
                list_current_index_for_each_domain = np.zeros(len(examples_segregated_by_domain), dtype=np.int32)
                list_examples_shuffle_domain = []
                for domain_dataset in examples_segregated_by_domain:
                    random_vec = np.arange(len(domain_dataset))
                    np.random.shuffle(random_vec)
                    list_examples_shuffle_domain.append(random_vec)
                while True:
                    domain_index_to_create_batch_from = get_domain_index_for_batch(batch_size,
                                                                                   examples_segregated_by_domain,
                                                                                   domain_that_can_be_selected_for_training,
                                                                                   list_current_index_for_each_domain)
                    if domain_index_to_create_batch_from is None:
                        break
                    else:
                        optimizer.zero_grad()

                        examples_source, labels_source, examples_target = \
                            create_source_and_target_batch(batch_size, examples_segregated_by_domain,
                                                           domain_index_to_create_batch_from,
                                                           labels_segregated_by_domain,
                                                           list_current_index_for_each_domain,
                                                           list_examples_shuffle_domain)

                        examples_source = Variable(torch.tensor(examples_source, dtype=torch.float32)).cuda()
                        labels_source = Variable(torch.tensor(labels_source, dtype=torch.long)).cuda()
                        examples_target = Variable(torch.tensor(examples_target, dtype=torch.float32)).cuda()

                        # Create the domain labels for the source
                        labels_domains_source = torch.zeros(len(examples_source), dtype=np.long)
                        labels_domains_source = Variable(labels_domains_source).cuda()
                        # Create the domain labels for the target
                        labels_domains_target = torch.ones(len(examples_target), dtype=np.long)
                        labels_domains_target = Variable(labels_domains_target).cuda()

                        # Training model using source data
                        if len(examples_source) < 1:
                            break
                        output_class_source, output_domain_source = convNet(examples_source,
                                                                            domain_index=
                                                                            domain_index_to_create_batch_from)
                        err_source_labels = criterion_class(output_class_source, labels_source)
                        err_source_domain = criterion_domain(output_domain_source, labels_domains_source)

                        # Training model using target data
                        _, output_domain_target = convNet(examples_target,
                                                          domain_index=domain_index_to_create_batch_from,
                                                          lambda_value=lambda_value)
                        err_target_domain = criterion_domain(output_domain_target, labels_domains_target)
                        error = err_source_labels + err_source_domain + err_target_domain
                        error.backward()
                        optimizer.step()

                        # Record the information to be displayed during training
                        _, predictions = torch.max(output_class_source.data, 1)
                        running_corrects_class += torch.sum(predictions == labels_source.data)
                        _, predictions = torch.max(output_domain_source.data, 1)
                        running_corrects_source_domain += torch.sum(predictions == labels_domains_source.data)
                        _, predictions = torch.max(output_domain_target.data, 1)
                        running_corrects_target_domain += torch.sum(predictions == labels_domains_target.data)

                        running_loss_class += err_source_labels.data.cpu().numpy()
                        running_loss_domain_source += err_source_domain.data.cpu().numpy()
                        running_loss_domain_target += err_target_domain.data.cpu().numpy()
                        total += labels_source.size(0)
                running_loss_class = running_loss_class.item() / total
                running_loss_domain_source = running_loss_domain_source.item() / total
                running_loss_domain_target = running_loss_domain_target.item() / total
                running_corrects_class = running_corrects_class.item() / total
                running_corrects_source_domain = running_corrects_source_domain.item() / total
                running_corrects_target_domain = running_corrects_target_domain.item() / total
                print('epoch: %d, [/patience: %d], Accuracy: %f,  err_s_label: %f, err_s_domain: %f, err_t_domain: %f, '
                      'accuracy source domain: %f, accuracy target domain: %f' \
                      % (epoch, patience, running_corrects_class, running_loss_class,
                         running_loss_domain_source, running_loss_domain_target, running_corrects_source_domain,
                         running_corrects_target_domain))
            else:
                running_loss_class = 0.
                running_corrects = 0
                total = 0


                convNet.train(False)  # Set model to evaluate mode
                # Go over all the domains
                index_domain = 0
                for domains_data, domains_labels in zip(examples_valid_segregated_by_domain,
                                                        labels_valid_segregated_by_domain):
                    # Make batch
                    for index_batch in range(0, len(domains_data), batch_size):
                        batch_size_to_use_now = batch_size
                        if index_batch + batch_size >= len(domains_data):
                            batch_size_to_use_now = len(domains_data) - index_batch
                        data_batch_valid = domains_data[index_batch:index_batch + batch_size_to_use_now]
                        labels_batch_valid = domains_labels[index_batch:index_batch + batch_size_to_use_now]
                        data_batch_valid = Variable(torch.tensor(data_batch_valid, dtype=torch.float32)).cuda()
                        labels_batch_valid = Variable(torch.tensor(labels_batch_valid, dtype=torch.long)).cuda()
                        # Training model using source data
                        output_class_source, _ = convNet(data_batch_valid, domain_index=index_domain)
                        _, predictions = torch.max(output_class_source.data, 1)
                        loss = criterion_class(output_class_source, labels_batch_valid)
                        loss = loss.item()

                        # statistics
                        running_loss_class += loss
                        running_corrects += torch.sum(predictions == labels_batch_valid.data)
                        total += labels_batch_valid.size(0)
                    index_domain += 1
                epoch_loss = running_loss_class / total
                epoch_acc = running_corrects.item() / total

                print('{} Loss: {} Acc: {}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(convNet.state_dict())
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    return best_weights
    # Save the best weights found to file
    # torch.save(best_weights, 'weights/best_pre_train_weights_target_raw.pt')
"""


def get_inputs_domain(dataloaders, current_domain_index):
    random_vector_index_domain = np.arange(len(dataloaders))
    np.random.shuffle(random_vector_index_domain)
    for i in random_vector_index_domain:
        if i != current_domain_index:
            for _, data in enumerate(dataloaders[i]):
                inputs, labels = data
                return inputs, i


def pre_train_model(model, criterion_class, criterion_domain, optimizer, scheduler, dataloaders, num_epochs=500,
                    precision=1e-8, lambda_value=1.):
    since = time.time()

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(len(dataloaders['val'])):
        state_dict = model.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batch_norm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    best_loss = float('inf')

    best_weights = copy.deepcopy(model.state_dict())

    patience = 30
    patience_increase = 30
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

            total = 0
            running_loss_class = 0.
            running_loss_domain_source = 0.
            running_loss_domain_target = 0.
            running_corrects_class = 0.
            running_corrects_source_domain = 0
            running_corrects_target_domain = 0

            # Get a random order for the training dataset
            random_vec = np.arange(len(dataloaders[phase]))
            np.random.shuffle(random_vec)

            for dataset_index in random_vec:
                # Retrieves the BN weights calculated so far for this dataset

                BN_weights = list_dictionaries_BN_weights[dataset_index]
                model.load_state_dict(BN_weights, strict=False)

                loss_over_datasets = 0.
                correct_over_datasets = 0.

                loss_over_domain_source = 0.
                loss_over_domain_target = 0.
                correct_over_domain_source = 0.
                correct_over_domain_target = 0.
                for i, data in enumerate(dataloaders[phase][dataset_index], 0):
                    # get the inputs
                    inputs, labels = data

                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    if phase == 'train':
                        model.train()
                        # forward
                        outputs, outputs_domain_source = model(inputs, lambda_value=lambda_value)
                        _, predictions = torch.max(outputs.data, 1)
                        # backward
                        loss_class = criterion_class(outputs, labels)
                        loss_class.backward(retain_graph=True)
                        optimizer.step()
                        loss_class = loss_class.item()

                        if lambda_value > 0:
                            # Save the BN statistics for this dataset
                            state_dict = model.state_dict()
                            batch_norm_dict = {}
                            for key in state_dict:
                                if "batch_norm" in key:
                                    batch_norm_dict.update({key: state_dict[key]})
                            list_dictionaries_BN_weights[dataset_index] = copy.deepcopy(batch_norm_dict)

                            # Create the domain labels for the source.
                            labels_domains_source = torch.zeros(len(inputs), dtype=torch.long)
                            labels_domains_source = Variable(labels_domains_source).cuda()
                            loss_domain_source = criterion_domain(outputs_domain_source, labels_domains_source)


                            # Create the domain labels for the target.
                            inputs_domains, domain_used_to_get_target_examples = get_inputs_domain(dataloaders[phase],
                                                                                                   current_domain_index=
                                                                                                   dataset_index)
                            BN_weights = list_dictionaries_BN_weights[domain_used_to_get_target_examples]
                            model.load_state_dict(BN_weights, strict=False)
                            inputs_domains = Variable(inputs_domains).cuda()

                            _, outputs_domain_target = model(inputs_domains, lambda_value=lambda_value)
                            labels_domains_target = torch.ones(len(inputs_domains), dtype=torch.long)
                            labels_domains_target = Variable(labels_domains_target).cuda()
                            loss_domain_target = criterion_domain(outputs_domain_target, labels_domains_target)

                            #print(loss_domain_target)

                            loss = loss_domain_source + loss_domain_target
                            loss.backward()
                            optimizer.step()

                            'Statistics specific for the training phase'
                            loss_over_domain_source += loss_domain_source.item()
                            loss_over_domain_target += loss_domain_target.item()
                            _, prediction_domain_source = torch.max(outputs_domain_source.data, 1)
                            _, prediction_domain_target = torch.max(outputs_domain_target.data, 1)
                            correct_over_domain_source += torch.sum(prediction_domain_source ==
                                                                    labels_domains_source.data)
                            correct_over_domain_target += torch.sum(prediction_domain_target ==
                                                                    labels_domains_target.data)


                            BN_weights = list_dictionaries_BN_weights[dataset_index]
                            model.load_state_dict(BN_weights, strict=False)

                    else:
                        model.eval()
                        # forward
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs.data, 1)
                        loss_class = criterion_class(outputs, labels)
                        loss_class = loss_class.item()
                    # Statistic for this dataset
                    loss_over_datasets += loss_class
                    correct_over_datasets += torch.sum(predictions == labels.data)
                    total += labels.size(0)
                # Statistic global
                running_loss_class += loss_over_datasets
                running_corrects_class += correct_over_datasets
                if phase == 'train':
                    running_loss_domain_source += loss_over_domain_source
                    running_loss_domain_target += loss_over_domain_target
                    running_corrects_source_domain += correct_over_domain_source
                    running_corrects_target_domain += correct_over_domain_target

                # Save the BN statistics for this dataset
                state_dict = model.state_dict()
                batch_norm_dict = {}
                for key in state_dict:
                    if "batch_norm" in key:
                        batch_norm_dict.update({key: state_dict[key]})
                list_dictionaries_BN_weights[dataset_index] = copy.deepcopy(batch_norm_dict)

            if phase == 'train' and lambda_value > 0:
                running_loss_class = running_loss_class / total
                running_corrects_class = running_corrects_class.item() / total

                running_loss_domain_source = running_loss_domain_source / total
                running_loss_domain_target = running_loss_domain_target / total
                running_corrects_source_domain = running_corrects_source_domain.item() / total
                running_corrects_target_domain = running_corrects_target_domain.item() / total
                print('epoch: %d, [/patience: %d], Accuracy: %f,  err_s_label: %f, err_s_domain: %f, err_t_domain: %f, '
                      'accuracy source domain: %f, accuracy target domain: %f' \
                      % (epoch, patience, running_corrects_class, running_loss_class,
                         running_loss_domain_source, running_loss_domain_target, running_corrects_source_domain,
                         running_corrects_target_domain))
            else:
                epoch_loss = running_loss_class / total
                epoch_acc = running_corrects_class.item() / total
                print('{} Loss: {:.8f} Acc: {:.8}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
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
'''

def train_model(network, criterion_class, criterion_domain, optimizer, scheduler, dataloaders, list_dataloaders_target,
                num_epochs=500, precision=1e-8):
    since = time.time()

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(len(list_dataloaders_target)+1):
        state_dict = network.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batch_norm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    best_loss = float('inf')

    best_weights = copy.deepcopy(network.state_dict())

    patience = 30
    patience_increase = 30
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                network.train(True)  # Set model to training mode
            else:
                network.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            running_corrects_domains = 4
            total_domains = 0

            # Get a random order for the training dataset
            random_vec = np.arange(len(dataloaders[phase]))
            np.random.shuffle(random_vec)

            # Retrieves the BN weights calculated so far for this dataset

            BN_weights = list_dictionaries_BN_weights[0]
            network.load_state_dict(BN_weights, strict=False)

            loss_over_datasets = 0.
            correct_over_datasets = 0.
            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data

                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    network.train()
                    # forward
                    if epoch < 20:
                        lambda_value = 0.
                    else:
                        lambda_value = 1.

                    outputs, outputs_domain_source = network(inputs, lambda_value=lambda_value)
                    _, predictions = torch.max(outputs.data, 1)
                    # backward
                    loss_class = criterion_class(outputs, labels)
                    loss_class.backward(retain_graph=True)
                    optimizer.step()
                    loss_class = loss_class.item()

                    if lambda_value != 0:
                        # Save the BN statistics for this dataset
                        state_dict = network.state_dict()
                        batch_norm_dict = {}
                        for key in state_dict:
                            if "batch_norm" in key:
                                batch_norm_dict.update({key: state_dict[key]})
                        list_dictionaries_BN_weights[0] = copy.deepcopy(batch_norm_dict)

                        # Create the domain labels for the source.
                        labels_domains_source = torch.zeros(len(inputs), dtype=np.long)
                        labels_domains_source = Variable(labels_domains_source).cuda()
                        loss_domain_source = criterion_domain(outputs_domain_source, labels_domains_source)

                        _, predictions_domain = torch.max(outputs_domain_source.data, 1)
                        running_corrects_domains += torch.sum(predictions_domain == labels_domains_source.data)
                        total_domains += len(labels_domains_source.data)


                        # Create the domain labels for the target.
                        inputs_domains, domain_used_to_get_target_examples = get_inputs_domain(list_dataloaders_target,
                                                                                               current_domain_index=
                                                                                               -1)
                        BN_weights = list_dictionaries_BN_weights[domain_used_to_get_target_examples+1]
                        network.load_state_dict(BN_weights, strict=False)
                        inputs_domains = Variable(inputs_domains).cuda()

                        outputs, outputs_domain_target = network(inputs_domains, lambda_value=lambda_value)
                        labels_domains_target = torch.ones(len(inputs_domains), dtype=np.long)
                        labels_domains_target = Variable(labels_domains_target).cuda()
                        loss_domain_target = criterion_domain(outputs_domain_target, labels_domains_target)
                        #print("SOURCE : ", predictions_domain)
                        _, predictions_domain = torch.max(outputs_domain_target.data, 1)
                        #print("TARGET : ", predictions_domain)
                        running_corrects_domains += torch.sum(predictions_domain == labels_domains_target.data)
                        total_domains += labels_domains_target.size(0)
                        #print(loss_domain_target)

                        loss = loss_domain_source + .25*loss_domain_target
                        loss.backward()
                        optimizer.step()

                        batch_norm_dict = {}
                        for key in state_dict:
                            if "batch_norm" in key:
                                batch_norm_dict.update({key: state_dict[key]})
                        list_dictionaries_BN_weights[domain_used_to_get_target_examples+1] = copy.deepcopy(batch_norm_dict)


                        BN_weights = list_dictionaries_BN_weights[0]
                        network.load_state_dict(BN_weights, strict=False)
                else:
                    network.eval()
                    # forward
                    outputs = network(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    loss_class = criterion_class(outputs, labels)
                    loss_class = loss_class.item()
                # Statistic for this dataset
                loss_over_datasets += loss_class
                correct_over_datasets += torch.sum(predictions == labels.data)
                total += labels.size(0)
            # Statistic global
            running_loss += loss_over_datasets
            running_corrects += correct_over_datasets


            # Save the BN statistics for this dataset
            state_dict = network.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batch_norm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[0] = copy.deepcopy(batch_norm_dict)


            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            if total_domains > 0:
                print(running_corrects_domains)
                epoch_accuracy_domain = running_corrects_domains.item() / total_domains
                print("Accuracy domain: %f" % epoch_accuracy_domain)

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(network.state_dict())
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
    network.load_state_dict(best_weights)
    network.eval()
    return network

'''
def train_model(convNet, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8):
    since = time.time()

    best_loss = float('inf')

    patience = 30
    patience_increase = 10

    best_weights = copy.deepcopy(convNet.state_dict())

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                convNet.train(True)  # Set model to training mode
            else:
                convNet.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    convNet.train()
                    # forward
                    outputs = convNet(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    # backward
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()

                else:
                    convNet.eval()
                    # forward
                    outputs = convNet(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss = loss.item()

                # statistics
                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total

            print('{} Loss: {} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(convNet.state_dict())
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
        #if epoch >= 25:
        #    break
    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    # load best model weights
    convNet.load_state_dict(copy.deepcopy(best_weights))
    convNet.eval()
    return convNet
