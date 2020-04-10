__author__ = 'marvinler'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import time
import numpy as np
import pprint
import datetime

from code import N_PROCESSES, get_logger
from code.data_processing.main import main as end_to_end_data_preprocessing
from code.data_processing.pytorch_dataset import Dataset
from code.data_processing.case_factory import split_svs_samples_casewise
from code.models.mil_wrapper import MaxMinMIL
from code.models.image_classifiers import instantiate_model


def define_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--preprocessed-data-folder', type=str, default='./data/preprocessed', metavar='PATH',
                        help='path of parent folder containing preprocessed slides data')

    parser.add_argument('--alpha', type=float, default=0.1, metavar='PERCENT',
                        help='assumed minimal % of tumor extent in tumor slides')
    parser.add_argument('--beta', type=float, default=0., metavar='PERCENT',
                        help='assumed minimal % of non-tumor extent in tumor slides')

    parser.add_argument('--underlying-model-type', type=str, default='resnet18', metavar='MODEL',
                        help='type of underlying model to use: this is the instance classifier architecture')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained underlying architecture as init point')
    parser.add_argument('--load-from', type=str, default=None, metavar='PT_PATH',
                        help='model pt path from which to initialize training')
    parser.add_argument('--no-save-model', action='store_true', default=False,
                        help='toggle model saving')
    parser.add_argument('--save-model-timesteps', type=int, default=5,
                        help='number of epochs for each model save')
    parser.add_argument('--save-model-folder', type=str, default='./saved_models/',
                        help='folder in which to save model')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--timestep-epoch', type=int, default=None, metavar='N',
                        help='number of step per epoch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--without-data-augmentation', action='store_true', default=False,
                        help='use data augmentation')
    parser.add_argument('--patience', type=int, default=10, metavar='N_EPOCHS',
                        help='number of epochs (patience) for early stopping callback')
    parser.add_argument('--no-tensorboard', action='store_true', default=False,
                        help='disables tensorboard losses logging')

    parser.add_argument('--dataset-max-size', type=int, default=None, metavar='SEED',
                        help='max number of slides per split set train/val/test')
    parser.add_argument('--max-bag-size', type=int, default=None, metavar='SEED',
                        help='max number of instances per bag (will randomly select if there are more)')
    parser.add_argument('--val-size', type=float, default=0.1, metavar='PROPORTION',
                        help='% of cases used for validation set')
    parser.add_argument('--test-size', type=float, default=0.15, metavar='PROPORTION',
                        help='% of cases used for test set')

    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='put logger console handler level to logging.DEBUG')
    parser.add_argument('--seed', type=int, default=123, metavar='SEED',
                        help='seed for datasets creation')

    args = parser.parse_args()

    hyper_parameters = {
        'preprocessed_data_folder': args.preprocessed_data_folder,
        'alpha': args.alpha,
        'beta': args.beta,

        # methods and underlying models
        'underlying_model_type': args.underlying_model_type.lower(),
        'underlying_model_pretrained': args.pretrained,
        'underlying_model_load_from': args.load_from,
        'save_model': not args.no_save_model,
        'save_model_timesteps': args.save_model_timesteps,

        # training control parameters
        'n_epochs': args.epochs,
        'n_timesteps_per_epoch': args.timestep_epoch,
        'learning_rate': args.lr,
        'weight_decay': args.reg,
        'early_stopping_patience': args.patience,
        'cuda': not args.no_cuda and torch.cuda.is_available(),
        'models_save_folder': args.save_model_folder,

        # dataset control parameters
        'max_bag_size': args.max_bag_size,
        'dataset_max_size': args.dataset_max_size,
        'with_data_augmentation': not args.without_data_augmentation,
        'with_tensorboard': not args.no_tensorboard,
        'seed': args.seed,
        'val_size': args.val_size,
        'test_size': args.test_size,

        'verbose': args.verbose,
    }

    return hyper_parameters


def to_dataloader(dataset, for_training):
    assert isinstance(dataset, Dataset) or isinstance(dataset, torch.utils.data.Subset)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=for_training, num_workers=N_PROCESSES)


def build_datasets(source_slides_folders, model_input_width, hyper_parameters, logger):
    normalization_channels_mean = (0.6387467, 0.51136744, 0.6061169)
    normalization_channels_std = (0.31200314, 0.3260718, 0.30386254)

    # First load all data into a single Dataset
    whole_dataset = Dataset(slides_folders=source_slides_folders, model_input_size=model_input_width,
                            is_training=False, max_bag_size=hyper_parameters['max_bag_size'],
                            logger=logger, max_dataset_size=hyper_parameters['dataset_max_size'],
                            with_data_augmentation=hyper_parameters['with_data_augmentation'],
                            seed=hyper_parameters['seed'],
                            normalization_mean=normalization_channels_mean,
                            normalization_std=normalization_channels_std)
    whole_cases_ids = whole_dataset.slides_cases
    whole_indexes = list(range(len(whole_dataset)))

    val_size = hyper_parameters['val_size']
    test_size = hyper_parameters['test_size']
    train_idx, val_idx, test_idx = split_svs_samples_casewise(whole_indexes, whole_cases_ids,
                                                              val_size=val_size, test_size=test_size,
                                                              seed=hyper_parameters['seed'])

    val_dataset = torch.utils.data.Subset(whole_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(whole_dataset, test_idx)
    train_dataset = torch.utils.data.Subset(whole_dataset, train_idx)
    train_dataset.dataset.is_training = True
    train_dataset.dataset.transform = train_dataset.dataset._define_data_transforms(normalization_channels_mean,
                                                                                    normalization_channels_std)

    return train_dataset, val_dataset, test_dataset


def early_stopping(val_losses, patience):
    """ Return (True, min achieved val loss) if no val losses is under the minimal achieved val loss for patience
        epochs, otherwise (False, None) """
    # Do not stop until enough epochs have been made
    if len(val_losses) < patience:
        return False, None

    best_val_loss = np.min(val_losses)
    if not np.any(val_losses[-patience:] <= best_val_loss):
        return True, best_val_loss
    return False, None


def perform_epoch(model, optimizer, epoch, dataloader, hyper_parameters, is_training, logger, set_name,
                  summary_writer=None):
    if is_training:
        model.train()
    else:
        model.eval()

    # Util to access mil wrapper self.loss method when using DataParallel
    model_get_attr = model.module if isinstance(model, nn.DataParallel) else model

    epoch_loss = []
    for batch_idx, (slide_instances, slide_label) in enumerate(dataloader):
        # reset gradients
        if is_training:
            optimizer.zero_grad()

        slide_instances = slide_instances.cuda()
        slide_label = slide_label.cuda()

        # Forward pass
        instances_predictions, computed_instances_labels, mask_instances_labels = model(slide_instances, slide_label)
        loss = model_get_attr.loss(instances_predictions, computed_instances_labels, mask_instances_labels)

        epoch_loss.append(loss.item() / slide_instances.shape[0])

        if is_training:
            loss.backward()
            optimizer.step()

        if hyper_parameters['n_timesteps_per_epoch'] is not None:
            if batch_idx + 1 >= hyper_parameters['n_timesteps_per_epoch']:
                break

    # STATS AND TENSORBOARD
    mean_epoch_loss = np.mean(epoch_loss)
    std_epoch_loss = np.std(epoch_loss)

    if hyper_parameters['with_tensorboard']:
        summary_writer.add_scalar('loss_mean/' + set_name, mean_epoch_loss, epoch)
        summary_writer.add_scalar('loss_std/' + set_name, std_epoch_loss, epoch)

    if is_training:
        log_msg = 'Epoch: {:2d}/{}  loss={:.4f}+/-{:.4f}'.format(epoch, hyper_parameters['n_epochs'],
                                                                 mean_epoch_loss, std_epoch_loss)
    else:
        log_msg = ' ' * 100 + set_name + '  loss={:.4f}+/-{:.4f}'.format(mean_epoch_loss, std_epoch_loss)
    logger.info(log_msg)

    # Save model
    if hyper_parameters['save_model'] and is_training and epoch % hyper_parameters['save_model_timesteps'] == 0:
        date_prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = os.path.join(hyper_parameters['models_save_folder'],
                                 '{}_epoch{}_loss{:.3f}.pt'.format(date_prefix, epoch, mean_epoch_loss))
        torch.save(model.state_dict(), save_path)
        logger.info('  saved model @%s' % save_path)
    else:
        save_path = None

    return mean_epoch_loss, save_path


def main(hyper_parameters):
    logger = get_logger(filename_handler='training.log', verbose=hyper_parameters['verbose'])
    logger.info('Hyper parameters')
    logger.info(pprint.pformat(hyper_parameters, indent=4))

    # Pre-processing should have been done beforehand, retrieve data by specifying data preprocessing output folder
    slides_folders = end_to_end_data_preprocessing(source_folder=None,
                                                   output_folder=hyper_parameters['preprocessed_data_folder'],
                                                   gdc_executable_path=None)

    logger.info('Initializing model... ')
    if not os.path.exists(hyper_parameters['models_save_folder']):
        os.makedirs(hyper_parameters['models_save_folder'])
    # Instantiate instance classifier model, then MIL wrapper
    instance_classifier, input_width = instantiate_model(model_type=hyper_parameters['underlying_model_type'],
                                                         pretrained=hyper_parameters['underlying_model_pretrained'],
                                                         n_classes=1)
    mil_model = MaxMinMIL(instance_classifier,
                          alpha=hyper_parameters['alpha'],
                          beta=hyper_parameters['beta'],
                          cuda=hyper_parameters['cuda'])
    logger.info('Instance model:')
    logger.info(instance_classifier)
    logger.info('MIL wrapper model:')
    logger.info(mil_model)

    if hyper_parameters['underlying_model_load_from'] is not None:
        logger.warning('Initializing model from %s' % hyper_parameters['underlying_model_load_from'])
        mil_model.load_state_dict(torch.load(hyper_parameters['underlying_model_load_from']))

    if hyper_parameters['cuda']:
        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            mil_model = nn.DataParallel(mil_model)
        mil_model.cuda()

    optimizer = optim.Adam(mil_model.parameters(), lr=hyper_parameters['learning_rate'],
                           weight_decay=hyper_parameters['weight_decay'])

    # Log model's and optimizer's state_dict
    logger.info("  Model's state_dict:")
    for param_tensor in mil_model.state_dict():
        logger.info(
            '   ' + param_tensor + " " * (40 - len(param_tensor)) + str(mil_model.state_dict()[param_tensor].size()))
    logger.info("  Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        logger.info('   ' + var_name + " " * (20 - len(var_name)) + str(optimizer.state_dict()[var_name]))

    # Load data and split case-wise into train, val and test sets
    logger.info('Pre-loading all data...')
    train_dataset, val_dataset, test_dataset = build_datasets(source_slides_folders=slides_folders,
                                                              model_input_width=input_width,
                                                              hyper_parameters=hyper_parameters,
                                                              logger=logger)
    logger.info('Train size %d' % len(train_dataset))
    logger.info('Val size %d' % len(val_dataset))
    logger.info('Test size %d' % len(test_dataset))

    train_dataloader = to_dataloader(train_dataset, True)
    val_dataloader = to_dataloader(val_dataset, False)
    test_dataloader = to_dataloader(test_dataset, False)

    # Instantiate summary writer if tensorboard activated
    if hyper_parameters['with_tensorboard']:
        summary_writer_filename = 'summary_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        summary_writer_folder_path = os.path.join('tensorboard', summary_writer_filename)
        summary_writer = SummaryWriter(log_dir=summary_writer_folder_path)
    else:
        summary_writer = None

    val_losses = []
    # Perform training loop: for each epoch, train and validate
    logger.info('Starting training...')
    start_training_time = time.time()
    for epoch in range(hyper_parameters['n_epochs']):
        # Train
        train_loss, train_savepath = perform_epoch(mil_model, optimizer, epoch, train_dataloader,
                                                   hyper_parameters=hyper_parameters, is_training=True,
                                                   logger=logger, set_name='training', summary_writer=summary_writer)

        # Validate
        with torch.no_grad():
            val_loss, _ = perform_epoch(mil_model, optimizer, epoch, val_dataloader,
                                        hyper_parameters=hyper_parameters, is_training=False,
                                        logger=logger, set_name='validation', summary_writer=summary_writer)

        # Early stopping
        val_losses.append(val_loss)
        do_stop, best_value = early_stopping(val_losses, patience=hyper_parameters['early_stopping_patience'])
        if do_stop:
            logger.warning('Early stopping triggered: stopping training after no improvement on val set for '
                           '%d epochs with value %.3f' % (hyper_parameters['early_stopping_patience'], best_value))
            break

    logger.warning('Total training time %s' % (time.time() - start_training_time))

    # Test
    logger.info('Starting testing...')
    with torch.no_grad():
        perform_epoch(mil_model, optimizer, -1, test_dataloader, hyper_parameters=hyper_parameters,
                      is_training=False, logger=logger, set_name='test', summary_writer=summary_writer)

    return


if __name__ == '__main__':
    main(define_args())
