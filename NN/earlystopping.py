#####################
#early stopping
#####################

class EarlyStopping():
    
    def __init__(self, patience = 20, delta = 0):
        self.patience = patience
        self.delta = delta
        self.earlystop = False
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            # reset counter if validation loss improving
            self.counter = 0
        elif self.best_loss - val_loss < self.delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.earlystop = True