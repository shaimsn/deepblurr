"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import pdb
import pytorch_ssim

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
from save_images import evaluate_save

parser = argparse.ArgumentParser()
parser.add_argument('--ssim', default='true', help='Whether or not to use ssim loss')
parser.add_argument('--data_dir', default='data/WF_final', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/testing_GAN', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--restore_fileD', default=None,
                    help="Optional, name of the file in --model_dir containing weights for discriminator to reload before \
                    training")

# TODO check change to inputs also need to check how to save/deal with second set of weights!
def train(model, modelD, optimizer, optimizerD, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()
    modelD.train()

    criterion = nn.BCELoss() #added for GAN
    real_label = 1
    fake_label = 0



    # check cuda
    # print(params.cuda)
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    reg_loss_avg = utils.RunningAverage()
    adv_loss_avg = utils.RunningAverage()

    ssim_loss = pytorch_ssim.SSIM()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        # pdb.set_trace()
        for i, (train_batch, labels_batch, _) in enumerate(dataloader):

            label = torch.FloatTensor(train_batch.size()[0])

            # move to GPU if available
            if params.cuda:
                # print("CUDA IS WORKING")
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
                criterion.cuda()
                label = label.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)




            # compute model output and loss
            # output_batch = model(train_batch)
            # loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            # optimizer.zero_grad()
            # loss.backward()

            # performs updates using calculated gradients
            # optimizer.step()

            # TODO evaluate: this is the new training method LOOK AT BATCH_SIZE
            # train D on ground truth
            modelD.zero_grad()
            # This should be a 1
            d_real_decision = modelD(labels_batch)
            # Loss of real decision
            d_real_error = criterion(torch.squeeze(d_real_decision), Variable(label.fill_(real_label)))
            d_real_error.backward()  # compute/store but dont change params

            # train D on generated images (fake)
            d_fake_data = model(train_batch).detach()  # detach to avoid training on these labels
            # This should be 0
            d_fake_decision = modelD(d_fake_data)
            # This is loss function
            d_fake_error = criterion(torch.squeeze(d_fake_decision), Variable(label.fill_(fake_label)))
            d_fake_error.backward()
            optimizerD.step()  # only optimizes D's parameters

            # train G on D's response but do not train D on these labels
            # TODO do I need to use the next batch??
            model.zero_grad()
            output_batch = model(train_batch)
            g_fake_decision = modelD(output_batch)

            adv_loss = criterion(torch.squeeze(g_fake_decision), Variable(label.fill_(real_label)))

            if args.ssim == 'true':
                # want to fool so set as true (uses L2 and Adv loss)

                reg_loss = -ssim_loss(output_batch, labels_batch)

            else:
                reg_loss = loss_fn(output_batch, labels_batch)

            loss = reg_loss + adv_loss

            # loss = loss_fn(output_batch, labels_batch) + criterion(torch.squeeze(g_fake_decision), Variable(label.fill_(real_label))) #want to fool so set as true (uses L2 and Adv loss)
            loss.backward()
            optimizer.step()  # only optimizes G's parameters

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])
            reg_loss_avg.update(reg_loss.data[0])
            adv_loss_avg.update(adv_loss.data[0])

            t.set_postfix(loss='reg: {:05.7f} + adv: {:05.7f}'.format(reg_loss_avg(), adv_loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.7f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, modelD, train_dataloader, val_dataloader, optimizer, optimizerD, loss_fn, metrics, params, model_dir,
                       restore_file=None, restore_fileD=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

        #TODO need to do this for Discriminator
    if restore_fileD is not None:
        restore_pathD = os.path.join(args.model_dir, args.restore_fileD + '.pth.tar')
        logging.info("Restoring Discriminator parameters from {}".format(restore_pathD))
        utils.load_checkpoint(restore_pathD, modelD, optimizerD)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, modelD, optimizer, optimizerD, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        if epoch % params.save_image_epochs == 0:
            logging.info('SAVING IMAGES')
            model_name = model_dir.split('/')[-1]
            evaluate_save(model, loss_fn, val_dataloader, metrics, params, iter_num=epoch, model_name=model_name)

        val_acc = val_metrics['psnr']
        is_best = val_acc >= best_val_acc
        
#        is_best=True
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)
        #TODO do the same for the Discriminator not sure what to do here
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': modelD.state_dict(),
                               'optim_dict': optimizerD.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir, gan=True)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # pdb.set_trace()
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    #TODO check these for discriminator
    modelD = net.NetD(params).cuda() if params.cuda else net.NetD(params)
    optimizerD = optim.Adam(modelD.parameters(), lr=params.learning_rate_NetD)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, modelD, train_dl, val_dl, optimizer, optimizerD, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
