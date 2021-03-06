"""Evaluates the model"""

import argparse
import logging
import os
import matplotlib.cm as cm
import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader
from PIL import Image
import pdb
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/WF_final', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/learning_rate', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate_save(model, loss_fn, dataloader, metrics, params, iter_num=0, model_name='default'):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch, out_names in dataloader:
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        # pdb.set_trace()
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # save the images
        num_images = len(out_names)
        for image in range(num_images):
            out_path = out_names[image].split('/')[:-1]
            out_path.append(model_name)  # separate into different models
            out_path.append('iter_{}'.format(iter_num))
            out_path = '/'.join(out_path)
            out_name = out_names[image].split('/')[-1]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out_path = out_path + '/' + out_name
            temp = output_batch[image].copy()
            temp = np.swapaxes(temp, 0, 2)
            temp = np.swapaxes(temp, 0, 1)
            # print('out_path = {}'.format(out_path))
            scipy.misc.imsave(out_path, temp)

    #     # compute all metrics on this batch
    #     summary_batch = {metric: metrics[metric](output_batch, labels_batch)
    #                      for metric in metrics}
    #     summary_batch['loss'] = loss.data[0]
    #     summ.append(summary_batch)
    #
    # # compute mean of all metrics in summary
    # metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Eval metrics : " + metrics_string)
    # return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    setname = 'train'
    dataloaders = data_loader.fetch_dataloader([setname], args.data_dir, params)
    test_dl = dataloaders[setname]

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate_save(model, loss_fn, test_dl, metrics, params, iter_num=0, model_name='learning_rate_'+setname)
    save_path = os.path.join(args.model_dir, "metrics_{}_{}.json".format(setname, args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
