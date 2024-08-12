import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.sampler import Sampler
from .dataset import collate_fn


class AugmentedDataSelector:
    """
        Select proper samples from the unlabeled data
    """

    def __init__(self, unlabeled_dataset, num_augmented_data_rate, eval_pool_size, eval_batch_size,
                 sampling_scheme='threshold', threshold=0.0, mc_dropout_iters=15, majority_votes=True):

        self.unlabeled_dataset = unlabeled_dataset
        self.num_augmented_data_rate = num_augmented_data_rate
        self.eval_pool_size = eval_pool_size
        self.eval_batch_size = eval_batch_size

        # Sampling config
        self.sampling_scheme = sampling_scheme
        self.threshold = threshold
        self.mc_dropout_iters = mc_dropout_iters
        self.majority_votes = majority_votes

        self.selected_avg_confidence = []
        self.finished_ids = np.array([])

        if 'threshold' in self.sampling_scheme or self.sampling_scheme == 'uniform':
            self.mc_dropout_iters = 1

    def select_samples(self, model):

        if self.mc_dropout_iters == 0:
            return None

        eval_pool_size = self.eval_pool_size
        if len(self.unlabeled_dataset) <= eval_pool_size:
            eval_pool_size = len(self.unlabeled_dataset)
        eval_pool_ids = np.random.choice(len(self.unlabeled_dataset), eval_pool_size, replace=False)

        # Evaluate uncertainty for the selected evaluation pool
        unlabeled_data_loader = torch.utils.data.DataLoader(self.unlabeled_dataset,
                                                            batch_size=self.eval_batch_size,
                                                            sampler=IndicesSampler(eval_pool_ids), collate_fn=collate_fn)
        y_T, y_pred = self._mc_dropout_evaluate(model, unlabeled_data_loader)

        # Sampling according to the sampling_scheme
        selected_pool_ids, conf_scores = None, None
        if 'uniform' in self.sampling_scheme:
            selected_pool_ids, conf_scores = self._uniform_sampling(y_T)

        elif 'threshold' in self.sampling_scheme:
            # Select measures
            if 'confidence' in self.sampling_scheme:
                target = 'CONF'
            # per_class_variance
            elif 'PV' in self.sampling_scheme:
                target = 'PV'
            # IG: information gain
            elif 'IG' in self.sampling_scheme:
                target = 'IG'
            # Normalized entropy
            elif 'NE' in self.sampling_scheme:
                target = 'NE'

            selected_pool_ids, conf_scores = self._BNN_threshold_sampling(y_T, y_pred, eval_pool_ids, target)

        elif 'IG' in self.sampling_scheme:

            selected_pool_ids, conf_scores = self._IG_sampling(y_T, y_pred, eval_pool_ids)
            if selected_pool_ids is None:
                return None, None, None, None
        selected_ids = eval_pool_ids[selected_pool_ids]
        pseudo_labels = y_pred[selected_pool_ids]
        conf_scores = conf_scores[selected_pool_ids]

        y_var = np.var(y_T, axis=0)  # (len_unlabeled_data, num_classes)

        unselected_ids = eval_pool_ids[[i for i in range(len(eval_pool_ids)) if i not in selected_pool_ids]]
        unselected_labels = y_pred[[i for i in range(len(y_pred)) if i not in selected_pool_ids]]

        return selected_ids, pseudo_labels, conf_scores, y_var[selected_pool_ids], unselected_ids, unselected_labels

    def _uniform_sampling(self, y_T):

        y_mean = np.mean(y_T, axis=0)
        selected_size = len(y_mean) * self.num_augmented_data_rate
        selected_pool_ids = np.random.choice(len(y_mean), int(selected_size), replace=False)
        conf_scores = y_mean.max(1)

        return selected_pool_ids, conf_scores

    def _IG_sampling(self, y_T, y_pred, eval_pool_ids):

        IG_acq = get_IG_acquisition(y_T)

        def real_to_pool(real_ids):
            converter = {real_ids: pool_ids
                         for pool_ids, real_ids in enumerate(eval_pool_ids)}
            pool_ids = np.array([int(converter[ids]) for ids in real_ids])
            return pool_ids

        if 'class' in self.sampling_scheme:

            selected_pool_ids = []

            for class_id in [1, 0]:
                class_ids = np.argwhere(y_pred == class_id).squeeze(1)
                if len(class_ids) == 0:
                    print(f"No instances are selected for class {class_id}")
                    selected_pool_ids = []
                    return None, None
                real_ids = eval_pool_ids[class_ids]
                unused_real_ids = np.setdiff1d(real_ids, self.finished_ids, assume_unique=True)
                unused_pool_ids = real_to_pool(unused_real_ids)

                IG_acq_class = IG_acq[unused_pool_ids]
                prob_norm = np.maximum(np.zeros(len(IG_acq_class)), (1. - IG_acq_class) / np.sum(1. - IG_acq_class))
                prob_norm = prob_norm / np.sum(prob_norm)
                selected_class_ids = unused_pool_ids[
                    np.random.choice(len(unused_pool_ids), int(len(unused_pool_ids) * self.num_augmented_data_rate),
                                     p=prob_norm)]
                selected_pool_ids.append(selected_class_ids)
            selected_pool_ids = np.concatenate(selected_pool_ids)

        else:
            unused_real_ids = np.setdiff1d(eval_pool_ids, self.finished_ids, assume_unique=True)
            unused_pool_ids = real_to_pool(unused_real_ids)

            unused_IG_acq = IG_acq[unused_pool_ids]
            prob_norm = np.maximum(np.zeros(len(unused_IG_acq)), (1. - unused_IG_acq) / np.sum(1. - unused_IG_acq))
            prob_norm = prob_norm / np.sum(prob_norm)
            selected_pool_ids = unused_pool_ids[np.random.choice(len(unused_pool_ids),
                                                                 int(len(
                                                                     unused_pool_ids) * self.num_augmented_data_rate),
                                                                 p=prob_norm)]

        if 'replacement' not in self.sampling_scheme:
            self.finished_ids = np.union1d(self.finished_ids, eval_pool_ids[selected_pool_ids])

        conf_scores = IG_acq

        return selected_pool_ids, conf_scores

    def _BNN_threshold_sampling(self, y_T, y_pred, eval_pool_ids, target='IG'):
        ''' y_pred: pseudo-labels (np.array of shape (len_unlabeled_data,)) '''

        # Lower means model is well-learned on it
        num_classes = y_T.shape[-1]
        target_measure = get_acquisitions(y_T)[target]

        def real_to_pool(real_ids):
            converter = {real_ids: pool_ids
                         for pool_ids, real_ids in enumerate(eval_pool_ids)}
            pool_ids = np.array([int(converter[ids]) for ids in real_ids])
            return pool_ids

        selected_pool_ids = None

        # class-separated
        if 'class' in self.sampling_scheme:

            selected_pool_ids = []

            for class_id in range(num_classes):
                class_ids = np.argwhere(y_pred == class_id).squeeze(1)
                target_measure_class = np.argsort(target_measure[class_ids])  # ascending order
                # target_measure_class = (target_measure < 0.05)

                # Without replacement: ids in self.finished_ids will not be sampled
                real_ids = eval_pool_ids[target_measure_class]
                selected_class_ids = np.setdiff1d(real_ids, self.finished_ids, assume_unique=True)
                selected_class_ids = selected_class_ids[:int(len(selected_class_ids) * self.num_augmented_data_rate)]
                selected_pool_ids.append(real_to_pool(selected_class_ids))

            selected_pool_ids = np.concatenate(selected_pool_ids).astype(int)

        else:
            selected_pool_ids = []
            target_measure_ids = np.argsort(target_measure)  # ascending order
            real_ids = eval_pool_ids[target_measure_ids]
            selected_ids = np.setdiff1d(real_ids, self.finished_ids)  # [:sample_size]
            selected_ids = selected_ids[:int(len(selected_ids) * self.num_augmented_data_rate)]
            selected_pool_ids = [i for i in real_to_pool(selected_ids)]

        if 'replacement' not in self.sampling_scheme:
            self.finished_ids = np.union1d(self.finished_ids, eval_pool_ids[selected_pool_ids])

        conf_scores = target_measure

        return selected_pool_ids, conf_scores

    def _mc_dropout_evaluate(self, model, unlabeled_data_loader):

        print("Evaluate by MC Dropout...", end="\r")
        device = next(model.parameters()).device

        if self.sampling_scheme == 'threshold':
            model.eval()
        else:
            model.train()

        y_T = []
        trange = tqdm(range(self.mc_dropout_iters), desc="Evaluating by MC Dropout", ncols=100)
        for i in trange:

            y_pred = torch.tensor([])

            with torch.no_grad():

                for i, batch in enumerate(unlabeled_data_loader, start=1):
                    input_ids = batch['input_ids'].to(device)
                    edge_index = batch['edge_index'].to(device)
                    edge_type = batch['edge_type'].to(device)
                    output = model(input_ids, edge_index, edge_type)
                    pred = output['logits']
                    y_pred = torch.cat((y_pred, pred.cpu()), dim=0)

                    trange.set_postfix_str(f"{i / len(unlabeled_data_loader) * 100.0:.2f}% Data")

            # y_pred: (len_unlabeled_data, num_classes)
            y_T.append(torch.nn.functional.softmax(y_pred, dim=1))

        y_T = torch.stack(y_T, dim=0).numpy()  # (T, len_unlabeled_data, num_classes)

        # compute majority prediction: y_pred.shape=(len_unlabeled_data,)
        if self.majority_votes:
            y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])

        else:  # Use hard labels

            y_pred = []
            model.eval()
            with torch.no_grad():

                for i, batch in enumerate(unlabeled_data_loader, start=1):
                    input_ids = batch['input_ids'].to(device)
                    edge_index = batch['edge_index'].to(device)
                    edge_type = batch['edge_type'].to(device)
                    output = model(input_ids, edge_index, edge_type)
                    logits = output['logits']
                    y_pred += logits.cpu().argmax(1).tolist()
            y_pred = np.array(y_pred)
        return y_T, y_pred


def to_prob(vec):
    vec = np.array(vec)
    prob = vec / np.sum(vec)
    prob = np.maximum(np.zeros(len(vec)), prob)
    prob = prob / np.sum(prob)

    return prob


def get_IG_acquisition(y_T, eps=1e-16):
    ''' y_T: numpy array of size: (T, len_dset, num_class) '''

    expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + eps), axis=-1), axis=0)
    expected_p = np.mean(y_T, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + eps), axis=-1)
    return (entropy_expected_p - expected_entropy)


def get_acquisitions(y_T, eps=1e-16):
    '''
    y_T: numpy array of size: (T, len_dset, num_class)
    '''

    u_c = np.mean(y_T, axis=0)  # (len_dset, num_class)
    H_t = - np.sum(y_T * np.log10(y_T + eps), axis=-1)  # (T, len_dset)

    # per_class_variance = np.mean((y_T - u_c)**2, axis=0)
    PV = np.sum(np.var(y_T, axis=0), axis=-1)

    # IG
    expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + eps), axis=-1), axis=0)
    expected_p = np.mean(y_T, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + eps), axis=-1)
    IG = (entropy_expected_p - expected_entropy)

    # Normalized entropy
    NE = np.mean(H_t, axis=0) / (np.log10(4))

    # Confidence
    CONF = u_c.max(1)  # (len_dset, )

    return {
        "PV": PV,
        "NE": NE,
        "IG": IG,
        "CONF": CONF
    }


class IndicesSampler(Sampler):
    ''' Data loader will only sample specific indices '''

    def __init__(self, indices, shuffle=False):

        self.indices = np.array(indices)
        self.len_ = len(self.indices)
        self.shuffle = shuffle

    def __len__(self):
        return self.len_

    def __iter__(self):

        if self.shuffle:
            np.random.shuffle(self.indices)  # in-place shuffle

        for index in self.indices:
            yield index


if __name__ == '__main__':
    pass