import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from stopper import EarlyStopping
from logger import TrainingLogger, MetricMeters
from .dataset import GraphDataset, collate_fn
from .model import Devign
from focal_loss import focal_loss
import yaml
from transformers import logging, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, f1_score, recall_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score, confusion_matrix
from balanced_loss import Loss


def train_teacher(ckpt_path, config):
    train_config = config['train']
    data_config = config['data']
    loss_config = config['loss']
    loss_fun = focal_loss(alpha=loss_config['focal_alpha'])
    ### Set random seed
    set_random_seed(train_config['random_seed'])

    ### Get data loader
    val_dset = GraphDataset(f'{data_config["ssl_data_path"]}/slice_{data_config["slice"]}/{data_config["label_rate"]}/val.txt')
    test_dset = GraphDataset(f'{data_config["ssl_data_path"]}/slice_{data_config["slice"]}/{data_config["label_rate"]}/test.txt')
    train_dset = GraphDataset(f'{data_config["ssl_data_path"]}/slice_{data_config["slice"]}/{data_config["label_rate"]}/train.txt', mode='train')

    num_classes = train_dset.num_classes

    train_loader = DataLoader(train_dset, batch_size=train_config['batch_size'], shuffle=False, pin_memory=True,
                              num_workers=1, collate_fn=collate_fn)
    num_vul, num_none = 0, 0
    for data in train_loader:
        num_vul += sum(data['label'] == 1)
        num_none += sum(data['label'] == 0)
    samples_per_class = [num_none, num_vul]
    print(samples_per_class)
    val_loader = DataLoader(val_dset, batch_size=train_config['eval_batch_size'], pin_memory=False, num_workers=1, collate_fn=collate_fn)
    test_loader = DataLoader(test_dset, batch_size=train_config['eval_batch_size'], pin_memory=False, num_workers=1, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Devign(input_channels=train_config['input_channels'], hidden_channels=train_config['hidden_channels'], num_layers=train_config['num_layers'])
    max_steps = train_config['epoch_stop_patience'] * len(train_loader) * 3
    warmup_steps = len(train_loader) * train_config['epoch_stop_patience'] // 5
    optimizer, scheduler = get_optimizer(model, train_config, max_steps, warmup_steps)

    logger = TrainingLogger(dest_path=os.path.join(ckpt_path, f'teacher_std_log.txt'))

    logger.print('Number of classes: {}'.format(num_classes))
    logger.print('Length of train/val/test datasets: {}, {}, {}'.format(
        len(train_dset), len(val_dset), len(test_dset)))
    logger.print('Length of train/val/test dataloaders: {}, {}, {}'.format(
        len(train_loader), len(val_loader), len(test_loader)))
    logger.print(f'Using {torch.cuda.device_count()} GPUs: '
                 + ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    logger.print('Config:\n' + str(config))
    logger.print('\n')

    # ------------- Start Training Teacher -------------#
    model = model.to(device)
    metric_meter = MetricMeters()
    train_steps = 0
    epoch_early_stopper = EarlyStopping(model, patience=train_config['epoch_stop_patience'], print_fn=logger.print,
                                        mode=train_config['stopper_mode'])

    while True:
        ### Training
        metric_meter.reset(mode='train')
        train(train_loader, model, optimizer, scheduler, metric_meter, device, loss_config, loss_fun, samples_per_class)
        train_steps += 1
        scores = metric_meter.epoch_get_scores()
        logger.print('Steps {:2d} | Train Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Train Acc: {:5f}%, prec: {:5f}%, '
                     'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%'.format(
            train_steps, scores['total_loss'], scores['class_loss'], scores['metric_loss'], scores['accuracy'] * 100.0,
            scores['precision'], scores['recall'], scores['f1'], scores['mcc']))
        if train_steps >= 3:
            metric_meter.reset(mode='val')
            val(val_loader, model, metric_meter, device, loss_config)
            scores = metric_meter.epoch_get_scores()
            logger.print(
                'Steps {:2d} | Val Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Val Acc: {:5f}%, prec: {:5f}%, '
                'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%, kappa: {:5f}%, auc: {:5f}%'.format(
                    train_steps, scores['total_loss'], scores['class_loss'], scores['metric_loss'],
                    scores['accuracy'] * 100.0,
                    scores['precision'], scores['recall'], scores['f1'], scores['mcc'], scores['kappa'], scores['auc']))
            save_ckpt = epoch_early_stopper(scores['mcc'], scores['class_loss'], scores['f1'], scores['accuracy'], scores['recall'])
            if save_ckpt:
                save_checkpoint(model, path=os.path.join(ckpt_path, f'best_teacher.pth'))
                if train_config['stopper_mode'] == 'loss':
                    logger.print(f'Best loss = {scores["class_loss"]:.5f} achieved.Checkpoint saved.')
                elif train_config['stopper_mode'] == 'mcc':
                    logger.print(f'Best mcc = {scores["mcc"]:.5f} achieved.Checkpoint saved.')
                elif train_config['stopper_mode'] == 'f1':
                    logger.print(f'Best f1 = {scores["f1"]:.5f} achieved.Checkpoint saved.')
                elif train_config['stopper_mode'] == 'acc':
                    logger.print(f'Best acc = {scores["accuracy"]:.5f} achieved.Checkpoint saved.')
                elif train_config['stopper_mode'] == 'f1acc':
                    logger.print(f'Best acc = {scores["accuracy"]:.5f} f1 = {scores["f1"]:.5f} achieved.Checkpoint saved.')
                elif train_config['stopper_mode'] == 'recallacc':
                    logger.print(f'Best acc = {scores["accuracy"]:.5f} recall = {scores["recall"]:.5f} achieved.Checkpoint saved.')
            if epoch_early_stopper.early_stop:
                break
            logger.print('\n')

    model_path = os.path.join(ckpt_path, 'best_teacher.pth')
    if not os.path.exists(model_path):
        return None
    load_checkpoint(model, model_path)

    metric_meter.reset(mode='test')
    confusion = val(test_loader, model, metric_meter, device, loss_config)
    scores = metric_meter.epoch_get_scores()
    logger.print(
        'Test Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Test Acc: {:5f}%, prec: {:5f}%, '
        'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%, kappa: {:5f}%, auc: {:5f}%, TN: {:d}, FP: {:d}, FN: {:d}, TP: {:d}'.format(
            scores['total_loss'], scores['class_loss'], scores['metric_loss'],
            scores['accuracy'] * 100.0,
            scores['precision'], scores['recall'], scores['f1'], scores['mcc'], scores['kappa'], scores['auc'],
            confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]))
    metric_meter.dump(path=os.path.join(ckpt_path, f'teacher_metric_log.json'))
    logger.print('\n' * 5)

    return scores


def set_random_seed(seed):
    # print('Random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, model, optimizer, scheduler, metric_meter, device, config, loss_fun, samples_per_class):
    model.train()
    for i, batch in enumerate(train_loader, start=1):

        input_ids = batch['input_ids'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_type = batch['edge_type'].to(device)

        label = batch['label'].to(device)

        output = model(input_ids, edge_index, edge_type)
        logits = output['logits']
        labels_np = label.cpu().detach().numpy()
        output_np = logits.cpu().argmax(1).detach().numpy()
        ### Calculate class loss
        if config['loss_type'] == 'FocalLoss':
            class_loss = loss_fun(logits, label, None)
        elif config['loss_type'] == 'BalancedLoss':
            focal_loss = Loss(
                loss_type="focal_loss",
                beta=0.999,  # class-balanced loss beta
                fl_gamma=2,  # focal loss gamma
                samples_per_class=samples_per_class,
                class_balanced=True
            )
            class_loss = focal_loss(logits, label)
        else:
            class_loss = F.cross_entropy(logits, label, reduction='none')
        class_loss = class_loss.mean()
        ### Calculate metric loss
        metric_loss = torch.tensor(0.0).to(device)
        protos = output['hidden_state']
        unique_label = torch.unique(label)
        if config['coef_teacher'] == 0.0:
            protos = protos.detach()

        for l in unique_label:
            target_protos = protos[label == l]  # (-1, hidden size)
            centroid = torch.mean(target_protos, axis=0)  # (hidden_size)
            distance = torch.sum(((target_protos - centroid) ** 2), axis=1)
            metric_loss += torch.mean(distance, axis=0)
        metric_loss = metric_loss / len(unique_label)  # / protos.size(-1)
        optimizer.zero_grad()
        loss = class_loss + config['coef_teacher'] * metric_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        metric_meter.update('total_loss', loss.item(), total=1)
        metric_meter.update('class_loss', class_loss.item(), total=1)
        metric_meter.update('metric_loss', metric_loss.item(), total=1)
        metric_meter.update('accuracy',
                            correct=(logits.argmax(1) == label).sum().item(), total=len(label))
        metric_meter.update('precision',
                            correct=precision_score(labels_np, output_np), total=1)
        metric_meter.update('recall',
                            correct=recall_score(labels_np, output_np), total=1)
        metric_meter.update('f1',
                            correct=f1_score(labels_np, output_np), total=1)
        metric_meter.update('mcc',
                            correct=matthews_corrcoef(labels_np, output_np), total=1)

def val(val_loader, model, metric_meter, device, config):
    model.eval()
    confusion = None
    logit_list = []
    target_list = []
    predict_list = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader, start=1):

            input_ids = batch['input_ids'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_type = batch['edge_type'].to(device)
            label = batch['label'].to(device)
            output = model(input_ids, edge_index, edge_type)
            logits = output['logits']
            labels_np = label.cpu().detach().numpy()
            output_np = logits.cpu().argmax(1).detach().numpy()
            logit_list.append(F.softmax(logits).cpu().detach().numpy())
            target_list.append(labels_np)
            predict_list.append(output_np)
            
            class_loss = F.cross_entropy(logits, label, reduction='mean')

            ### Calculate metric loss
            contrastive_loss = torch.tensor(0.0).to(device)
            protos = output['hidden_state']
            if config['coef_student'] == 0.0:
                protos = protos.detach()

            loss = class_loss + config['coef_student'] * contrastive_loss
            
            if confusion is None:
                confusion = confusion_matrix(labels_np, output_np)
            else:
                confusion += confusion_matrix(labels_np, output_np)

            metric_meter.update('total_loss', loss.item(), total=1)
            metric_meter.update('class_loss', class_loss.item(), total=1)
            metric_meter.update('metric_loss', contrastive_loss.item(), total=1)
            metric_meter.update('accuracy',
                                correct=(logits.argmax(1) == label).sum().item(), total=len(label))

    logit_list = np.concatenate(logit_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    predict_list = np.concatenate(predict_list, axis=0)
    confusion = confusion_matrix(target_list, predict_list)
    f1 = f1_score(target_list, predict_list)
    precision = precision_score(target_list, predict_list)
    recall = recall_score(target_list, predict_list)
    mcc = matthews_corrcoef(np.array([1 if x == 1 else -1 for x in target_list]), np.array([1 if x == 1 else -1 for x in predict_list]))
    kappa = cohen_kappa_score(target_list, predict_list)
    
    auc = roc_auc_score(target_list, logit_list[:, 1])

    metric_meter.precision = precision
    metric_meter.recall = recall
    metric_meter.f1 = f1
    metric_meter.mcc = mcc
    metric_meter.kappa = kappa
    metric_meter.auc = auc
    
    return confusion

def save_checkpoint(model, path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))


def get_optimizer(model, train_config, max_steps, warmup_steps):
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters()],
         'weight_decay': train_config['weight_decay'], 'lr': train_config['bert_lr']},], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    return optimizer, scheduler


if __name__ == '__main__':
    pass