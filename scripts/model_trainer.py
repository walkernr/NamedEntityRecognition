import numpy as np
from tqdm import tqdm
import torch
from seqeval.metrics import accuracy_score, f1_score


class ModelTrainer(object):
    def __init__(self, model, data, optimizer_cls, criterion_cls, device):
        '''
        
        class for basic functions common to the trainer objects used in this project

        model: the model to be trained
        data: the data corpus to be used for training/validation/testing
        optimizer_cls: the optimizer function (note - pass Adam instead of Adam() or Adam(model.parameters()))
        criterion_cls: the optimization criterion (loss) (note - pass the function name instead of the called function)
        device: torch device

        '''
        self.device = device
        # send model to device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = optimizer_cls(self.model.parameters())
        # ignoes the padding in the tags
        self.criterion = criterion_cls(ignore_index=self.data.tag_pad_idx).to(device)
        # initialize empty lists for training
        self.train_batch_history = []
        self.valid_batch_history = []
    

    def save_model(self, model_path):
        ''' saves entire model to file '''
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, model_path)
    

    def load_model(self, model_path):
        ''' loads entire model from file '''
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model = self.model.to(self.device)
    

    def save_history(self, train_path, valid_path):
        ''' save training histories to file '''
        torch.save(self.train_batch_history, train_path)
        torch.save(self.valid_batch_history, valid_path)
    

    def load_history(self, train_path, valid_path):
        ''' load training histories from file '''
        self.train_batch_history = torch.load(train_path)
        self.valid_batch_history = torch.load(valid_path)


    def get_history(self):
        ''' get history '''
        return self.train_batch_history, self.valid_batch_history
    

    def train_evaluate_epoch(self, epoch, n_epoch, iterator, train, mode):
        '''
        
        train or evaluate epoch (calls the iterate_batches method from a subclass that inherits from this class)

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        train: switch for whether or not to train this epoch
        mode: string that just labels the epoch in the output

        '''
        if train:
            # make sure the model is set to train if it is training
            self.model.train()
            # train all of the batches and collect the batch/epoch loss/metrics
            batch_loss, batch_accuracy, batch_f1 = self.iterate_batches(epoch, n_epoch, iterator, train, mode)
        else:
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                batch_loss, batch_accuracy, batch_f1 = self.iterate_batches(epoch, n_epoch, iterator, train, mode)
        # return batch/epoch loss/metrics
        return batch_loss, batch_accuracy, batch_f1
    

    def train(self, n_epoch):
        '''

        trains the model (with validation)

        n_epoch: number of training epochs

        '''
        for epoch in range(n_epoch):
            # training
            train_batch_loss, train_batch_accuracy, train_batch_f1 = self.train_evaluate_epoch(epoch, n_epoch, self.data.train_iter, True, 'train')
            # validation
            valid_batch_loss, valid_batch_accuracy, valid_batch_f1 = self.train_evaluate_epoch(epoch, n_epoch, self.data.valid_iter, False, 'validate')
            # append histories
            self.train_batch_history.append([train_batch_loss, train_batch_accuracy, train_batch_f1])
            self.valid_batch_history.append([valid_batch_loss, valid_batch_accuracy, valid_batch_f1])
    
    
    def test(self):
        ''' evaluates the test set '''
        return self.train_evaluate_epoch(0, 1, self.data.test_iter, False, 'test')


class NERTrainer(ModelTrainer):
    def __init__(self, model, data, optimizer_cls, criterion_cls, device):
        ''' trainer for named entity recognition model. inherits from model trainer class '''
        super().__init__(model, data, optimizer_cls, criterion_cls, device)
    

    def iterate_batches(self, epoch, n_epoch, iterator, train, mode):
        '''
        
        iterates through batchs in an epoch

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        train: switch for whether or not to train this epoch
        mode: string that just labels the epoch in the output

        '''
        # initialize lists for batch losses and metrics 
        batch_loss = []
        batch_accuracy = []
        batch_f1 = []
        # initialize batch range
        batch_range = tqdm(iterator, desc='')
        for batch in batch_range:
            # fetch texts, characters, and tags from batch
            text = batch.text.to(self.device)
            char = batch.char.to(self.device)
            tag = batch.tag.to(self.device)

            # zero out prior gradients for training
            if train:
                self.optimizer.zero_grad()

            # output depends on whether conditional random field is used for prediction/loss
            if self.model.use_crf:
                prediction, loss = self.model(text, char, tag)
            else:
                logit = self.model(text, char, tag)
                loss = self.criterion(logit.view(-1, logit.shape[-1]), tag.view(-1))
                logit = logit.detach().cpu().numpy()
                prediction = [list(p) for p in np.argmax(logit, axis=2)]

            # send the true tags to python list on the cpu
            true = list(tag.to('cpu').numpy())

            # put the prediction tags and valid tags into a nested list form for the scoring metrics
            prediction_tags = [[self.data.tag_field.vocab.itos[ii] for ii, jj in zip(i, j) if self.data.tag_field.vocab.itos[jj] != self.data.pad_token] for i, j in zip(prediction, true)]
            valid_tags = [[self.data.tag_field.vocab.itos[ii] for ii in i if self.data.tag_field.vocab.itos[ii] != self.data.pad_token] for i in true]

            # calculate the accuracy and f1 scores
            accuracy = accuracy_score(valid_tags, prediction_tags)
            f1 = f1_score(valid_tags, prediction_tags)

            # append to the lists
            batch_loss.append(loss.item())
            batch_accuracy.append(accuracy)
            batch_f1.append(f1)

            # backpropagate the gradients and step the optimizer forward
            if train:
                loss.backward()
                self.optimizer.step()

            # calculate means across the batches so far
            means = (np.mean(batch_loss), np.mean(batch_accuracy), np.mean(batch_f1))
            # display progress
            batch_range.set_description('| epoch: {:d}/{} | {} | loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f} |'.format(epoch+1, n_epoch, mode, *means))
        # return the batch losses and metrics
        return batch_loss, batch_accuracy, batch_f1