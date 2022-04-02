import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, dataloader, dataset

import copy
import json
import time

from model.advise_pytorch import ADVISE
from dataloaders.ads_dataset import AdsDataset
from losses.triplet_loss_2 import compute_loss
from utils import eval_utils
from dataloaders.utils import load_action_reason_annots

num_epochs = 300
batch_size = 128
lr = 0.001
lr_decay = 1.0
lr_decay_epochs = 25

action_reason_annot_path = 'data/train/QA_Combined_Action_Reason_train.json'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def evaluate_predictions(groundtruths, results):
    metrics = eval_utils.evaluate(results, groundtruths)
    print("Evaluation results: {}".format(json.dumps(metrics, indent=2)))

    # Save results.
    return metrics['accuracy']


def main():
    with open('configs/advise_densecap_data.json') as fp:
        data_config = json.load(fp)

    print("Train Dataset")
    train_dataset = AdsDataset(data_config, split='train')
    print("Val Dataset")
    val_dataset = AdsDataset(data_config, split='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    print("Dataset:", dataset_sizes)

    # Load training config
    with open('configs/advise_kb_training.json') as fp:
        train_config = json.load(fp)

    # Init model for training
    model = ADVISE(train_config, device, is_training=True)
    model = model.to(device)

    print("Model Parameters:")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0

    groundtruths = load_action_reason_annots(action_reason_annot_path)

    # torch.autograd.set_detect_anomaly(True)
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                model.is_training = True
            else:
                model.eval()  # Set model to evaluate mode
                model.is_training = False

            # Loop through the evaluation dataset.
            results = {}

            running_loss = 0.0
            loss_img_stmt = 0.0
            loss_stmt_img = 0.0
            # running_corrects = 0

            # Iterate over data.
            for examples in dataloaders[phase]:
                for key, value in examples.items():
                    if torch.is_tensor(examples[key]):
                        examples[key] = examples[key].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(examples)
                    loss_dict = compute_loss(outputs, train_config, is_training=model.is_training)

                    loss = torch.tensor(0.0).to(device)
                    for loss_name, loss_tensor in loss_dict.items():
                        loss += loss_tensor

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_size

                loss_img_stmt += loss_dict['triplet_img_stmt'].item() * batch_size
                loss_stmt_img += loss_dict['triplet_stmt_img'].item() * batch_size

                # print('{} Total Loss: {:.4f} Img-Stmt Loss: {:.4f} Stmt-Img Loss: {:.4f} Dense-Img Loss: {:.4f} Dense-Stmt Loss: {:.4f} Symb-Img Loss: {:.4f} Symb-Stmt Loss: {:.4f}'.format(phase, loss.item(), loss_dict['triplet_img_stmt'].item(), loss_dict['triplet_stmt_img'].item(), loss_dict['triplet_dense_img'].item(), loss_dict['triplet_dense_stmt'].item(), loss_dict['triplet_symb_img'].item(), loss_dict['triplet_symb_stmt'].item()))

                if phase != 'test':
                    for image_id, distances in zip(outputs['image_id'], outputs['distance']):
                        results[image_id] = {
                            'distances': list(map(lambda x: round(x, 5), distances.tolist())),
                        }
            # if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            loss_img_stmt = loss_img_stmt / dataset_sizes[phase]
            loss_stmt_img = loss_stmt_img / dataset_sizes[phase]

            # print('\nEpoch done')
            # print('-'*10)
            print(
                '{} Total Loss: {:.4f} Img-Stmt Loss: {:.4f} Stmt-Img Loss: {:.4f} '.format(phase, epoch_loss, loss_img_stmt, loss_stmt_img))

            if phase != 'test':
                acc = evaluate_predictions(groundtruths, results)
                print('{} Acc: {:.4f}'.format(phase, acc))

            # deep copy the model
            if phase == 'val' and acc > best_val_acc:
                best_val_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_val_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    add = "trained_models/model_kb_vit_512_0005.pth"
    torch.save(model.state_dict(), add)
    print('Model saved at ' + add)


if __name__ == '__main__':
    main()