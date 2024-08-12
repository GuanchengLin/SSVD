class EarlyStopping:
    """
        This class determines when to stop the training early
        or increase the number of training epochs.
    """

    def __init__(self, model, patience=10, delta=0, print_fn=print, mode='loss'):
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.early_stop = False
        self.max_mcc = 0.0
        self.max_f1 = 0.0
        self.max_acc = 0.0
        self.min_loss = 1000
        self.max_recall = 0.0
        self.print_fn = print_fn
        self.model = model
        self.mode = mode

    def __call__(self, val_mcc, val_loss, val_f1, val_acc, val_recall):
        save_ckpt = False

        if self.mode == 'f1':
            if val_f1 <= self.max_f1:
                self.counter += 1
                if val_f1 + self.delta > self.max_f1:
                    save_ckpt = True

                if self.counter >= self.patience:
                    self.early_stop = True
                    self.print_fn("Early stopped.")
                else:
                    self.print_fn(f"Early stop counter {self.counter}/{self.patience}.")
            else:
                save_ckpt = True
                self.counter = 0
                self.max_f1 = val_f1

        if self.mode == 'acc':
            if val_acc <= self.max_acc:
                self.counter += 1
                if val_acc + self.delta > self.max_acc:
                    save_ckpt = True

                if self.counter >= self.patience:
                    self.early_stop = True
                    self.print_fn("Early stopped.")
                else:
                    self.print_fn(f"Early stop counter {self.counter}/{self.patience}.")
            else:
                save_ckpt = True
                self.counter = 0
                self.max_acc = val_acc

        if self.mode == 'f1acc':
            if val_acc < self.max_acc or val_f1 <= self.max_f1:
                self.counter += 1
                if val_f1 > self.max_f1 and val_acc + self.delta > self.max_acc:
                    save_ckpt = True
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.print_fn("Early stopped.")
                else:
                    self.print_fn(f"Early stop counter {self.counter}/{self.patience}.")
            else:
                save_ckpt = True
                self.counter = self.patience - 50
                self.max_acc = val_acc
                self.max_f1 = val_f1

        return save_ckpt

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.max_mcc = 0.0
        self.max_f1 = 0.0
        self.min_loss = 1000
        self.max_acc = 0.0
        self.max_recall = 0.0