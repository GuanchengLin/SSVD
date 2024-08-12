import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .dataset import GraphDataset, collate_fn
from .model import ReVeal
from .sampler import AugmentedDataSelector, IndicesSampler
from stopper import EarlyStopping
from logger import TrainingLogger, MetricMeters
from nrce import NRCE
from sklearn.metrics import precision_score, f1_score, recall_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score, confusion_matrix
from focal_loss import focal_loss
from MetricLoss import *
from balanced_loss import Loss
import pandas as pd


def train_HADES(ckpt_path, excel_path, teacher_path, config):

    train_config = config['train']
    data_config = config['data']
    loss_config = config['loss']
    sample_config = config['sampling']

    ### Set random seed
    set_random_seed(train_config['random_seed'])

    ### Set up logger
    loss_fun = focal_loss(alpha=loss_config['focal_alpha'])
    logger = TrainingLogger(dest_path=os.path.join(ckpt_path, f'student_std_log.txt'))

    ### Get data loader
    val_dset = GraphDataset(f'{data_config["ssl_data_path"]}/slice_{data_config["slice"]}/{data_config["label_rate"]}/val.txt')
    test_dset = GraphDataset(f'{data_config["ssl_data_path"]}/slice_{data_config["slice"]}/{data_config["label_rate"]}/test.txt')
    train_dset = GraphDataset(f'{data_config["ssl_data_path"]}/slice_{data_config["slice"]}/{data_config["label_rate"]}/train.txt', mode='train')
    unlabeled_dset = GraphDataset(f'{data_config["ssl_data_path"]}/slice_{data_config["slice"]}/{data_config["label_rate"]}/unsupervised.txt', mode='train')
    

    train_loader = DataLoader(train_dset, batch_size=train_config['label_batch_size'], shuffle=True, pin_memory=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dset, batch_size=train_config['eval_batch_size'], pin_memory=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dset, batch_size=train_config['eval_batch_size'], pin_memory=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ReVeal(input_channels=train_config['input_channels'], hidden_channels=train_config['hidden_channels'], num_layers=train_config['num_layers'])
    load_checkpoint(model, teacher_path, logger)

    unlabeled_data_sampler = AugmentedDataSelector(
        unlabeled_dset,
        num_augmented_data_rate=sample_config['sampling_rate'],
        eval_pool_size=sample_config['eval_pool_size'],
        eval_batch_size=train_config['eval_batch_size'],
        mc_dropout_iters=sample_config['mc_dropout_iters'],
        sampling_scheme=sample_config['sampling_scheme'],
        majority_votes=True
    )

    # logger.print('Number of classes: {}'.format(num_classes))
    logger.print('Length of val/unlabeled/test datasets: {}, {}, {}'.format(len(val_dset), len(unlabeled_dset), len(test_dset)))
    logger.print('Length of val/unlabeled/test dataloaders: {}, {}, {}'.format(len(val_loader), int(1 / sample_config['sampling_rate']),
        len(test_loader)))
    logger.print(f'Using {torch.cuda.device_count()} GPUs:'
                 + ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    logger.print('Config:\n' + str(config))
    logger.print('\n')

    # ------------- Start of Self-training -------------#
    model = model.to(device)
    model.device = device
    best_model_path = os.path.join(ckpt_path, 'best_student.pth')
    metric_meter = MetricMeters()
    epoch = 1
    self_training_early_stopper = EarlyStopping(model, patience=train_config['self_train_patience'],
                                                delta=train_config['patience_delta'], print_fn=logger.print,
                                                mode=train_config['stopper_mode'])
    metric_meter.reset(mode=f'teacher_val')
    val(val_loader, model, metric_meter, loss_config)
    teacher_scores = metric_meter.epoch_get_scores()

    best_mcc = teacher_scores['mcc']
    logger.print(f'teacher val mcc {best_mcc}')
    best_loss = teacher_scores['class_loss']
    logger.print(f'teacher val loss {best_loss}')
    best_f1 = teacher_scores['f1']
    logger.print(f'teacher val f1 {best_f1}')
    best_acc = teacher_scores['accuracy']
    logger.print(f'teacher val acc {best_acc}')
    best_recall = teacher_scores['recall']
    logger.print(f'teacher val recall {best_recall}')

    metric_meter.reset(mode=f'teacher_test_1')
    confusion = val(test_loader, model, metric_meter, loss_config)
    teacher_scores = metric_meter.epoch_get_scores()
    logger.print(
        'Test Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Test Acc: {:5f}%, prec: {:5f}%, '
        'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%, kappa: {:5f}%, auc: {:5f}%, TN: {:d}, FP: {:d}, FN: {:d}, TP: {:d}'.format(
            teacher_scores['total_loss'], teacher_scores['class_loss'], teacher_scores['contrastive_loss'],
            teacher_scores['accuracy'] * 100.0,
            teacher_scores['precision'], teacher_scores['recall'], teacher_scores['f1'], teacher_scores['mcc'],
            teacher_scores['kappa'], teacher_scores['auc'], confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]))
    metrics_data = [['teacher', teacher_scores['accuracy'], teacher_scores['precision'], teacher_scores['recall'], teacher_scores['f1'], teacher_scores['mcc'],
                     teacher_scores['auc'], teacher_scores['kappa'], confusion[1][1], confusion[0][1], confusion[0][0], confusion[1][0]]]
    save_excel(metrics_data, excel_path)

    teacher_test_mcc = teacher_scores['mcc']
    logger.print(f'teacher test mcc {teacher_test_mcc}')
    teacher_test_loss = teacher_scores['class_loss']
    logger.print(f'teacher test loss {teacher_test_loss}')
    teacher_test_f1 = teacher_scores['f1']
    logger.print(f'teacher test f1 {teacher_test_f1}')
    teacher_test_acc = teacher_scores['accuracy']
    logger.print(f'teacher test accuracy {teacher_test_acc}')
    teacher_test_recall = teacher_scores['recall']
    logger.print(f'teacher test recall {teacher_test_recall}')


    self_training_early_stopper.max_mcc = best_mcc
    self_training_early_stopper.min_loss = best_loss
    self_training_early_stopper.max_f1 = best_f1
    self_training_early_stopper.max_acc = best_acc
    self_training_early_stopper.max_recall = best_recall

    while epoch < train_config['self_train_max_epoch'] + 1:
        logger.print('----- Start of Self Training Epoch {:2d} -----'.format(epoch))
        epoch += 1
        # ------------- Select samples from unlabeled dataset -------------#
        ### Load best model

        if os.path.exists(best_model_path):
            load_checkpoint(model, best_model_path, logger)
        else:
            load_checkpoint(model, teacher_path, logger)
        selected_ids, pseudo_labels, conf_scores, y_var, unselected_ids, unselected_labels, selected_prob, unselected_prob = unlabeled_data_sampler.select_samples(model)
        used_data_percent = len(unlabeled_data_sampler.finished_ids) / len(unlabeled_dset)
        if used_data_percent > 0.9:
            break
        if selected_ids is None:
            continue
        if len(selected_ids) == 0:
            break
        else:
            logger.print(f'{len(selected_ids)} samples are selected. ')

        logger.print(f'length of finished ids: {len(unlabeled_data_sampler.finished_ids)}')
        logger.print('Selected uncertainty (max/min/avg): ' +
                     f'{conf_scores.max():.4f}/{conf_scores.min():.4f}/{conf_scores.mean():.4f}')
        logger.print(
            f'selected positive num {sum((pseudo_labels == 1))} selected negtive num {sum((pseudo_labels == 0))}')
        samples_per_class = [sum((pseudo_labels == 0)), sum((pseudo_labels == 1))]

        # Update dataset with pseudo-labels and predicted variance
        for index, label, var in zip(selected_ids, pseudo_labels, y_var):
            unlabeled_dset.update_data(index, content={'pseudo_label': label, 'variance': var[label]})

        true_label_selected = []
        true_label_unselected = []
        for index in selected_ids:
            true_label_selected.append(unlabeled_dset[index]['label'])
        for index in unselected_ids:
            true_label_unselected.append(unlabeled_dset[index]['label'])

        selected_confusion = confusion_matrix(true_label_selected, pseudo_labels)
        unselected_confusion = confusion_matrix(true_label_unselected, unselected_labels)

        tn, fp, fn, tp = selected_confusion[0][0], selected_confusion[0][1], selected_confusion[1][0], selected_confusion[1][1]
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        kappa = cohen_kappa_score(true_label_selected, pseudo_labels)
        mcc = matthews_corrcoef(np.array([1 if x == 1 else -1 for x in true_label_selected]), np.array([1 if x == 1 else -1 for x in pseudo_labels]))
        auc = roc_auc_score(true_label_selected, selected_prob[:, 1])
        logger.print(f'Pseudo label accuracy: {accuracy} | precision: {precision} | '
                     f'recall: {recall} | kappa: {kappa} | mcc: {mcc} | auc: {auc} | f1: {f1} | tn {tn} | fp {fp} | fn {fn} | tp {tp}')
        metrics_data = [[f'pseudo-select-epoch{epoch}', accuracy, precision, recall, f1, mcc,
                    auc, kappa, tp, fp, tn, fn]]
        save_excel(metrics_data, excel_path)

        tn, fp, fn, tp = unselected_confusion[0][0], unselected_confusion[0][1], unselected_confusion[1][0], unselected_confusion[1][1]
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        kappa = cohen_kappa_score(true_label_unselected, unselected_labels)
        mcc = matthews_corrcoef(np.array([1 if x == 1 else -1 for x in true_label_unselected]), np.array([1 if x == 1 else -1 for x in unselected_labels]))
        auc = roc_auc_score(true_label_unselected, unselected_prob[:, 1])
        logger.print(f'Pseudo label accuracy: {accuracy} | precision: {precision} | '
                     f'recall: {recall} | kappa: {kappa} | mcc: {mcc} | auc: {auc} | f1: {f1} | tn {tn} | fp {fp} | fn {fn} | tp {tp}')
        metrics_data = [[f'pseudo-unselect-epoch{epoch}', accuracy, precision, recall, f1, mcc,
                    auc, kappa, tp, fp, tn, fn]]
        save_excel(metrics_data, excel_path)

        unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_dset,
                                                            batch_size=train_config['batch_size'],
                                                            sampler=IndicesSampler(selected_ids, shuffle=True), collate_fn=collate_fn, num_workers=4)

        ### Reset model and optimizer each epoch
        max_steps = train_config['epoch_stop_patience'] * len(unlabeled_data_loader) * 3
        warmup_steps = len(unlabeled_data_loader) * train_config['epoch_stop_patience'] // 2
        optimizer, scheduler = get_optimizer(model, train_config, max_steps, warmup_steps)
        print(f'LR {optimizer.state_dict()["param_groups"][0]["lr"]}')
        if os.path.exists(best_model_path):  # No best model at first iteration
            load_checkpoint(model, best_model_path, logger)
        else:
            load_checkpoint(model, teacher_path, logger)

        # ------------- Training with the selected unlabeled dataset -------------#
        epoch_early_stopper = EarlyStopping(model, patience=train_config['epoch_stop_patience'], print_fn=logger.print,
                                            delta=train_config['patience_delta'], mode=train_config['stopper_mode'])
        if train_config['stopper_mode'] == 'mcc':
            epoch_early_stopper.max_mcc = self_training_early_stopper.max_mcc
            logger.print(f'target {epoch_early_stopper.max_mcc}')
        if train_config['stopper_mode'] == 'loss':
            epoch_early_stopper.min_loss = self_training_early_stopper.min_loss
            logger.print(f'target {epoch_early_stopper.min_loss}')
        if train_config['stopper_mode'] == 'f1':
            epoch_early_stopper.max_f1 = self_training_early_stopper.max_f1
            logger.print(f'target {epoch_early_stopper.max_f1}')
        if train_config['stopper_mode'] == 'acc':
            epoch_early_stopper.max_acc = self_training_early_stopper.max_acc
            logger.print(f'target {epoch_early_stopper.max_acc}')
        if train_config['stopper_mode'] == 'f1acc':
            epoch_early_stopper.max_acc = self_training_early_stopper.max_acc
            epoch_early_stopper.max_f1 = self_training_early_stopper.max_f1
            logger.print(f'target {epoch_early_stopper.max_acc} f1 {epoch_early_stopper.max_f1}')
        if train_config['stopper_mode'] == 'recallacc':
            epoch_early_stopper.max_acc = self_training_early_stopper.max_acc
            epoch_early_stopper.max_recall = self_training_early_stopper.max_recall
            logger.print(f'target {epoch_early_stopper.max_acc} recall {epoch_early_stopper.max_recall}')
        train_steps = 0
        while True:
            train_steps += 1
            metric_meter.reset(mode=f'train_epoch{train_steps}')
            train(unlabeled_data_loader, train_loader, model, optimizer, scheduler, metric_meter, loss_config, loss_fun, samples_per_class)
            scores = metric_meter.epoch_get_scores()
            logger.print(
                'Steps {:2d} | Train Loss(total/cls/ct): {:5f}, {:5f}, {:5f} | Train Acc: {:5f}%, prec: {:5f}%, '
                'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%'.format(
                    train_steps, scores['total_loss'], scores['class_loss'], scores['contrastive_loss'],
                    scores['accuracy'] * 100.0,
                    scores['precision'], scores['recall'], scores['f1'], scores['mcc']))
            ### Validating

            metric_meter.reset(mode=f'val_epoch{train_steps}')
            val(val_loader, model, metric_meter, loss_config)

            scores = metric_meter.epoch_get_scores()
            logger.print(
                'Steps {:2d} | Val Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Val Acc: {:5f}%, prec: {:5f}%, '
                'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%, kappa: {:5f}%, auc: {:5f}%'.format(
                    train_steps, scores['total_loss'], scores['class_loss'], scores['contrastive_loss'],
                    scores['accuracy'] * 100.0,
                    scores['precision'], scores['recall'], scores['f1'], scores['mcc'], scores['kappa'], scores['auc']))

            save_ckpt = epoch_early_stopper(scores['mcc'], scores['class_loss'], scores['f1'], scores['accuracy'], scores['recall'])

            if save_ckpt:
                # Test
                metric_meter.reset(mode=f'test_epoch{train_steps}')
                confusion = val(test_loader, model, metric_meter, loss_config)
                scores = metric_meter.epoch_get_scores()
                logger.print(
                    'Test Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Test Acc: {:5f}%, prec: {:5f}%, '
                    'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%, kappa: {:5f}%, auc: {:5f}%, TN: {:d}, FP: {:d}, FN: {:d}, TP: {:d}'.format(
                        scores['total_loss'], scores['class_loss'], scores['contrastive_loss'],
                        scores['accuracy'] * 100.0,
                        scores['precision'], scores['recall'], scores['f1'], scores['mcc'], scores['kappa'],
                        scores['auc'], confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]))
                metric_meter.dump(path=os.path.join(ckpt_path, f'student_metric_log.json'))
                metrics_data = [[f'student-epoch{epoch}', scores['accuracy'], scores['precision'], scores['recall'], scores['f1'], scores['mcc'],
                    scores['auc'], scores['kappa'], confusion[1][1], confusion[0][1], confusion[0][0], confusion[1][0]]]
                save_excel(metrics_data, excel_path)

                if train_config['stopper_mode'] == 'mcc':
                    test_mcc_dif = scores['mcc'] - teacher_test_mcc
                    best_model_path = os.path.join(ckpt_path, 'best_student_{:.5f}.pth'.format(test_mcc_dif))
                    save_checkpoint(model, path=best_model_path, logger=logger)
                    logger.print(f'mcc diff = {test_mcc_dif:.5f} achieved. Checkpoint saved.')
                if train_config['stopper_mode'] == 'f1':
                    test_f1_dif = scores['f1'] - teacher_test_f1
                    best_model_path = os.path.join(ckpt_path, 'best_student_{:.5f}.pth'.format(test_f1_dif))
                    save_checkpoint(model, path=best_model_path, logger=logger)
                    logger.print(f'f1 diff = {test_f1_dif:.5f} achieved. Checkpoint saved.')
                if train_config['stopper_mode'] == 'loss':
                    test_loss_dif = scores['loss'] - teacher_test_loss
                    best_model_path = os.path.join(ckpt_path, 'best_student_{:.5f}.pth'.format(test_loss_dif))
                    save_checkpoint(model, path=best_model_path, logger=logger)
                    logger.print(f'loss diff = {test_loss_dif:.5f} achieved. Checkpoint saved.')
                if train_config['stopper_mode'] == 'acc':
                    test_acc_dif = scores['accuracy'] - teacher_test_acc
                    best_model_path = os.path.join(ckpt_path, 'best_student_{:.5f}.pth'.format(test_acc_dif))
                    save_checkpoint(model, path=best_model_path, logger=logger)
                    logger.print(f'acc diff = {test_acc_dif:.5f} achieved. Checkpoint saved.')
                if train_config['stopper_mode'] == 'f1acc':
                    test_acc_dif = scores['accuracy'] - teacher_test_acc
                    test_f1_dif = scores['f1'] - teacher_test_f1
                    best_model_path = os.path.join(ckpt_path, 'best_student_{:.5f}_{:.5f}.pth'.format(test_acc_dif, test_f1_dif))
                    save_checkpoint(model, path=best_model_path, logger=logger)
                    logger.print(f'acc diff = {test_acc_dif:.5f} f1 diff = {test_f1_dif:.5f} achieved. Checkpoint saved.')
                if train_config['stopper_mode'] == 'recallacc':
                    test_acc_dif = scores['accuracy'] - teacher_test_acc
                    test_recall_dif = scores['recall'] - teacher_test_recall
                    best_model_path = os.path.join(ckpt_path, 'best_student_{:.5f}_{:.5f}.pth'.format(test_acc_dif, test_recall_dif))
                    save_checkpoint(model, path=best_model_path, logger=logger)
                    logger.print(f'acc diff = {test_acc_dif:.5f} recall diff = {test_recall_dif:.5f} achieved. Checkpoint saved.')
            if epoch_early_stopper.early_stop:
                break
        self_training_early_stopper(epoch_early_stopper.max_mcc, epoch_early_stopper.min_loss,
                                    epoch_early_stopper.max_f1, epoch_early_stopper.max_acc, epoch_early_stopper.max_recall)
        logger.print('----- End of Self Training Epoch {:2d} -----'.format(epoch))
        logger.print('\n\n')
        if self_training_early_stopper.early_stop:
            break

    # ------------- Evaluate with the test dataset -------------#
    if not os.path.exists(best_model_path):
        logger.print('self training can not obtain better model')
        return
    load_checkpoint(model, best_model_path, logger)
    metric_meter.reset(mode='test', clear=True)
    val(test_loader, model, metric_meter, loss_config)

    scores = metric_meter.epoch_get_scores()
    logger.print(
        'Test Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Test Acc: {:5f}%, prec: {:5f}%, '
        'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%'.format(
            scores['total_loss'], scores['class_loss'], scores['contrastive_loss'],
            scores['accuracy'] * 100.0,
            scores['precision'], scores['recall'], scores['f1'], scores['mcc']))
    metric_meter.dump(path=os.path.join(ckpt_path, f'student_metric_log.json'))

    return scores


def train(unlabel_train_loader, label_train_loader, model, optimizer, scheduler, metric_meter, config, loss_fun, samples_per_class):
    device = model.device
    model.train()
    landa = 0.4
    # optimizer.zero_grad()
    label_loader = infinite_iter(label_train_loader)
    for i, batch in enumerate(unlabel_train_loader, start=1):
        labeled_batch = next(label_loader)
        input_ids = batch['input_ids'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_type = batch['edge_type'].to(device)
        label_input_ids = labeled_batch['input_ids'].to(device)
        label_edge_index = labeled_batch['edge_index'].to(device)
        label_edge_type = labeled_batch['edge_type'].to(device)
        label_label = labeled_batch['label'].to(device)
        label = batch['pseudo_label'].to(device)

        selected_ids = []

        output = model(input_ids, edge_index, edge_type)
        logits = output['logits']
        labels_np = label.cpu().detach().numpy()
        output_np = logits.cpu().argmax(1).detach().numpy()
        label_output = model(label_input_ids, label_edge_index, label_edge_type)
        label_logits = label_output['logits']
        class_loss = F.cross_entropy(logits, label)
        class_loss = landa * class_loss.mean()
        label_class_loss = F.cross_entropy(label_logits, label_label)
        label_class_loss = (1 - landa) * label_class_loss.mean()

        loss = class_loss + label_class_loss
        # loss = loss / len(unlabel_train_loader)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        metric_meter.update('total_loss', loss.item(), total=1)
        metric_meter.update('class_loss', class_loss.item(), total=1)
        metric_meter.update('contrastive_loss', 0, total=1)
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
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # optimizer.step()
    # scheduler.step()

def val(val_loader, model, metric_meter, config):
    device = model.device
    model.eval()
    confusion = None
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
            ### Calculate class loss
            if config['loss_type'] == 'NRCE':
                class_loss = NRCE(logits, label, tau=config['PHuber_tau'], reduction='mean')
            else:
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
            metric_meter.update('contrastive_loss', contrastive_loss.item(), total=1)
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
            metric_meter.update('kappa',
                                correct=cohen_kappa_score(labels_np, output_np), total=1)
            metric_meter.update('auc',
                                correct=roc_auc_score(labels_np, output_np), total=1)
    tn, fp, fn, tp = confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    metric_meter.precision = precision
    metric_meter.recall = recall
    metric_meter.f1 = f1
    return confusion

def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)

def save_excel(metrics_data, excel_path):
    metric_excel = pd.read_excel(excel_path)
    df = pd.concat([metric_excel, pd.DataFrame(metrics_data, columns=metric_excel.columns)], ignore_index=True)
    df.to_excel(excel_path, index=False)

def set_random_seed(seed):
    # print('Random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_optimizer(model, train_config, max_steps, warmup_steps):
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters()],
         'weight_decay': train_config['weight_decay'], 'lr': train_config['bert_lr']}, ], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    return optimizer, scheduler


def save_checkpoint(model, path, logger):
    logger.print(f'Save model to {path}.')

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path, logger):
    logger.print(f'Load model from {path}.')

    if isinstance(model, nn.DataParallel):
        if torch.cuda.is_available():
            model.module.load_state_dict(torch.load(path))
        else:
            model.module.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(path, map_location='cpu'))


if __name__ == '__main__':
    pass
