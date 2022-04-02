from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from losses.triplet_loss import compute_loss
from model.advise_pytorch import ADVISE
from dataloaders.ads_dataset import AdsDataset
from torch.utils.data import DataLoader, dataloader, dataset

import os
import json
import time
import nltk
import numpy as np

import torch
from dataloaders.utils import load_action_reason_annots
from utils import eval_utils

split = 'test'
batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

action_reason_annot_path = 'data/test/QA_Combined_Action_Reason_test.json'
results_path = 'results/advise_results.json'

def _load_vocab(filename):
    """Loads vocabulary.

    Args:
        filename: path to the vocabulary file.

    Returns:
        a list mapping from id to word.
    """
    with open(filename, 'r') as fp:
        vocab = ['UNK'] + [x.strip('\n').split('\t')[0] for x in fp.readlines()]
    return vocab


def export_inference(results, groundtruths, filename):
    """Exports results to a specific file.

    Args:
        results: 
        groundtruths:
        filename: the path to the output json file.
    """
    final_results = {}
    for image_id, result in results.items():
        pred = np.array(result['distances']).argmin()
        final_results[image_id] = groundtruths[image_id]['all_examples'][pred]

    with open(filename, 'w') as fp:
        fp.write(json.dumps(final_results))

#Currently only for the non continuous eval mode
def evaluate_predictions(groundtruths, results):
    export_inference(results, groundtruths, results_path)

    metrics = eval_utils.evaluate(results, groundtruths)
    print("Evaluation results: {}".format(json.dumps(metrics, indent=2)))

    # Save results.
    return metrics['accuracy']

def main():
    with open('configs/advise_densecap_data.json') as fp:
        data_config = json.load(fp)

    groundtruths = load_action_reason_annots(action_reason_annot_path)

    dataset = AdsDataset(data_config, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with open('configs/advise_kb_training.json') as fp:
        train_config = json.load(fp)

    #Init model for training
    model = ADVISE(train_config, device,is_training=False)

    model.eval()

    # Loop through the evaluation dataset.
    results = {}

    # Iterate over data.
    for examples in dataloader:
        for key, value in examples.items():
            if torch.is_tensor(examples[key]):
                examples[key] = examples[key].to(device)
        #examples = examples.to(device)

        running_loss = 0.0

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(examples)
            loss_dict = compute_loss(outputs, train_config, is_training= False)
            
            loss = torch.tensor(0.0)
            for loss_name, loss_tensor in loss_dict.items():
                loss += loss_tensor

        for image_id, distances in zip(outputs['image_id'], outputs['distance']):
            results[image_id] = {
            'distances': map(lambda x: round(x, 5), distances.tolist()),
            }

        # statistics
        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / len(dataset)
    acc = evaluate_predictions(groundtruths, results)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format("Test", epoch_loss, acc))


if __name__ == '__main__':
    main()