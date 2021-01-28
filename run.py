# run.py

import torch
import torch.utils.model_zoo
from tqdm import tqdm
import os
import numpy as np
import torchvision
import pickle as p

from cornet_irl import get_cornet_irl
from datasets import TDWSurfaceNormalsDataset

N_GPUS = 2
NUM_WORKERS = 0

NBP = dict()  # Parameters for training with everything but normals branch frozen
NBP['N_GPUS'] = N_GPUS
NBP['BATCH_SIZE'] = 8
NBP['N_EPOCHS'] = 3
NBP['LEARNING_RATE'] = 1e-2
NBP['MOMENTUM'] = 0.9
NBP['WEIGHT_DECAY'] = 0
NBP['CLASS_LOSS_WEIGHT'] = 100
NBP['NORMALS_LOSS_WEIGHT'] = 0.1

EBP = dict()  # Parameters for training with only normals branch frozen
EBP['N_GPUS'] = N_GPUS
EBP['BATCH_SIZE'] = 128
EBP['N_EPOCHS'] = 3
EBP['LEARNING_RATE'] = 1e-4
EBP['MOMENTUM'] = 0.9
EBP['WEIGHT_DECAY'] = 0
EBP['CLASS_LOSS_WEIGHT'] = 100
EBP['NORMALS_LOSS_WEIGHT'] = 0.1

FMP = dict()  # Parameters for training with nothing frozen
FMP['N_GPUS'] = N_GPUS
FMP['BATCH_SIZE'] = 128
FMP['N_EPOCHS'] = 3
FMP['LEARNING_RATE'] = 1e-4
FMP['MOMENTUM'] = 0.9
FMP['WEIGHT_DECAY'] = 0
FMP['CLASS_LOSS_WEIGHT'] = 100
FMP['NORMALS_LOSS_WEIGHT'] = 0.1

TRAIN_DATASET_LOCATION = os.path.join("reformatted_data", "train")
VAL_DATASET_LOCATION = os.path.join("reformatted_data", "val")
SAVED_LOCATION = "saved"

GENERATE_SAVED_DOWNSAMPLED_NORMALS = False


cornet_s = get_cornet_irl(pretrained='cornet_s', n_gpus=N_GPUS)
cornet_irl = get_cornet_irl(pretrained='cornet_s', n_gpus=N_GPUS)

# Instantiate Datasets
train_dataset = TDWSurfaceNormalsDataset(root_dir=TRAIN_DATASET_LOCATION)
val_dataset = TDWSurfaceNormalsDataset(root_dir=VAL_DATASET_LOCATION)

if GENERATE_SAVED_DOWNSAMPLED_NORMALS:
    train_dataset.generate_saved_downsampled_normals()
    val_dataset.generate_saved_downsampled_normals()


def forward_pass(criterion1, criterion2, cornet_s, cornet_irl, data, parameters):
    images = data['img']
    downsampled_normals = data['downsampled_normals']
    if parameters['N_GPUS'] > 0:
        downsampled_normals = downsampled_normals.cuda(non_blocking=True)

    _, s_class_preds = cornet_s(images)
    irl_normals_preds, irl_class_preds = cornet_irl(images)
    class_loss = criterion1(irl_class_preds, s_class_preds)
    normals_loss = criterion2(irl_normals_preds, downsampled_normals)

    class_loss *= parameters['CLASS_LOSS_WEIGHT']
    normals_loss *= parameters['NORMALS_LOSS_WEIGHT']

    return class_loss, normals_loss


def validate(criterion1, criterion2, cornet_s, cornet_irl, val_loader, parameters):
    cornet_irl.eval()
    cornet_s.eval()

    with torch.no_grad():
        steps_in_epoch = len(val_loader)

        class_losses = np.zeros(steps_in_epoch)
        normals_losses = np.zeros(steps_in_epoch)

        for step, data in enumerate(tqdm(val_loader, desc="Validating")):
            class_loss, normals_loss =\
                forward_pass(criterion1, criterion2, cornet_s,\
                             cornet_irl, data, parameters)

            class_losses[step] = class_loss
            normals_losses[step] = normals_loss

        avg_class_loss = class_losses.mean()
        avg_normals_loss = normals_losses.mean()

    cornet_irl.train()
    cornet_s.train()
    return avg_class_loss, avg_normals_loss


def train(train_loader, val_loader, cornet_s, cornet_irl, optimizer, parameters, prefix):
    VAL_PERIOD = int(16000 / parameters['BATCH_SIZE'])

    # criterion for matching standard CORnet S output
    criterion1 = torch.nn.MSELoss()

    # criterion for correctly predicting surface normals
    criterion2 = torch.nn.MSELoss()

    if parameters['N_GPUS'] > 0:
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()

    steps_in_epoch = len(train_loader)

    recent_training_class_losses = []
    recent_training_normals_losses = []
    recent_training_total_losses = []

    records = []

    for epoch in range(parameters['N_EPOCHS']):
        for step, data in enumerate(tqdm(train_loader, desc="Training")):
            global_step = epoch * steps_in_epoch + step

            if step % VAL_PERIOD == 0:
                val_class_loss, val_normals_loss =\
                    validate(criterion1, criterion2, cornet_s,
                             cornet_irl, val_loader, parameters)
                val_total_loss = val_class_loss + val_normals_loss

            training_class_loss, training_normals_loss =\
                forward_pass(criterion1, criterion2, cornet_s, cornet_irl,\
                             data, parameters)
            training_total_loss = training_class_loss + training_normals_loss

            recent_training_class_losses.append(training_class_loss)
            recent_training_normals_losses.append(training_normals_loss)
            recent_training_total_losses.append(training_total_loss)

            if step % VAL_PERIOD == 0:
                avg_recent_training_class_loss =\
                    sum(recent_training_class_losses)/len(recent_training_class_losses)
                avg_recent_training_normals_loss =\
                    sum(recent_training_normals_losses)/len(recent_training_normals_losses)
                avg_recent_training_total_loss =\
                    sum(recent_training_total_losses)/len(recent_training_total_losses)

                recent_training_class_losses = []
                recent_training_normals_losses = []
                recent_training_total_losses = []

                record = {'global_step': global_step,
                          'val_class_loss': val_class_loss,
                          'val_normals_loss': val_normals_loss,
                          'val_total_loss': val_total_loss,
                          'train_class_loss': avg_recent_training_class_loss,
                          'train_normals_loss': avg_recent_training_normals_loss,
                          'train_total_loss': avg_recent_training_total_loss}

                # Print to console
                print("Global Step [{}/{}]".format(global_step, parameters['N_EPOCHS'] * steps_in_epoch))

                print("Validation Class Loss: {:.4f}".format(val_class_loss))
                print("Validation Normals Loss: {:.4f}".format(val_normals_loss))
                print("Validation Total Loss: {:.4f}".format(val_total_loss))

                print("Training Class Loss: {:.4f}".format(avg_recent_training_class_loss))
                print("Training Normals Loss: {:.4f}".format(avg_recent_training_normals_loss))
                print("Training Total Loss: {:.4f}".format(avg_recent_training_total_loss))

                records.append(record)

                saved_records_location =\
                    os.path.join(SAVED_LOCATION, "{}_{}_records.p".format(prefix, global_step))
                saved_records_file = open(saved_records_location, 'wb')
                p.dump(records, saved_records_file, protocol=p.HIGHEST_PROTOCOL)

                saved_model_location =\
                    os.path.join(SAVED_LOCATION, "{}_{}_model.pth.tar".format(prefix, global_step))
                ckpt_data = dict()
                ckpt_data['parameters'] = parameters
                ckpt_data['state_dict'] = cornet_irl.state_dict()
                ckpt_data['optimizer'] = optimizer.state_dict()
                torch.save(ckpt_data, saved_model_location)

            # Backward pass and optimize
            optimizer.zero_grad()
            training_total_loss.backward()
            optimizer.step()

    return records


if __name__ == "__main__":
    # Train cornet_irl, freeze everything but normals branch
    nbp_train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=NBP['BATCH_SIZE'],
                                                   shuffle=True,
                                                   num_workers=NUM_WORKERS,
                                                   pin_memory=True,
                                                   drop_last=True)
    nbp_val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=NBP['BATCH_SIZE'],
                                                 shuffle=True,
                                                 num_workers=NUM_WORKERS,
                                                 pin_memory=True,
                                                 drop_last=True)

    nbp_optimizer = torch.optim.SGD(list(cornet_irl.module.normals_branch.parameters()),
                                    NBP['LEARNING_RATE'],
                                    NBP['MOMENTUM'],
                                    NBP['WEIGHT_DECAY'])

    print("Training With Everything But Normals Branch Frozen")
    train(nbp_train_loader, nbp_val_loader, cornet_s, cornet_irl, nbp_optimizer, NBP, 'NBP')

    # Train cornet_irl, freeze only normals branch
    ebp_train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=EBP['BATCH_SIZE'],
                                                   shuffle=True,
                                                   num_workers=NUM_WORKERS,
                                                   pin_memory=True,
                                                   drop_last=True)
    ebp_val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=EBP['BATCH_SIZE'],
                                                 shuffle=True,
                                                 num_workers=NUM_WORKERS,
                                                 pin_memory=True,
                                                 drop_last=True)

    normals_branch_params = set(cornet_irl.module.normals_branch.parameters())
    all_params = set(cornet_irl.module.parameters())
    wanted_params = list(all_params.difference(normals_branch_params))

    ebp_optimizer = torch.optim.SGD(wanted_params,
                                    EBP['LEARNING_RATE'],
                                    EBP['MOMENTUM'],
                                    EBP['WEIGHT_DECAY'])

    print("Training With Only Normals Branch Frozen")
    train(ebp_train_loader, ebp_val_loader, cornet_s, cornet_irl, ebp_optimizer, EBP, 'EBP')

    # Train cornet_irl, freeze nothing
    fmp_train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=FMP['BATCH_SIZE'],
                                                   shuffle=True,
                                                   num_workers=NUM_WORKERS,
                                                   pin_memory=True,
                                                   drop_last=True)
    fmp_val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=FMP['BATCH_SIZE'],
                                                 shuffle=True,
                                                 num_workers=NUM_WORKERS,
                                                 pin_memory=True,
                                                 drop_last=True)

    full_model_optimizer = torch.optim.SGD(cornet_irl.parameters(),
                                           FMP['LEARNING_RATE'],
                                           FMP['MOMENTUM'],
                                           FMP['WEIGHT_DECAY'])

    print("Training With Nothing Frozen")
    train(fmp_train_loader, fmp_val_loader, cornet_s, cornet_irl, full_model_optimizer, FMP, 'FMP')
