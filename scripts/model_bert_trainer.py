import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from seqeval.metrics import accuracy_score, f1_score


class BERTTrainer(nn.Module):
    def __init__(self, model, data, optimizer_cls, full_finetuning, max_grad_norm, device):
        '''
        
        class for training the bert named entity recognition model

        model: the model to be trained
        data: the data corpus to be used for training/validation/testing
        optimizer_cls: the optimizer function (note - pass Adam instead of Adam() or Adam(model.parameters()))
        criterion_cls: the optimization criterion (loss) (note - pass the function name instead of the called function)
        max_grad_norm: flor clipping gradients to prevent blow-up
        device: torch device

        '''
        super().__init__()
        self.device = device
        self.data = data
        self.model = model.to(self.device)
        self.max_grad_norm = max_grad_norm
        self.full_finetuning = full_finetuning
        self.finetuning()
        self.optimizer = optimizer_cls(self.optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
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
    

    def finetuning(self):
        ''' determines the parameters to be finetuned '''
        # optimize all network parameters if doing a full finetuning
        if self.full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            self.optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                                                 {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate': 0.0}]
        # otherwise, optimize only the classification layers
        else:
            param_optimizer = list(self.model.bert.classifier.named_parameters())
            if self.model.use_crf:
                param_optimizer += list(self.model.crf.named_parameters())
            self.optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    

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
            # fetch texts and tags from batch
            text = batch.text.to(self.device)
            tag = batch.tag.to(self.device)
            # ignore pad tokens in the attention mask
            attention_mask = torch.tensor(np.array([[tt != self.data.text_field.pad_token for tt in t] for t in text])).to(self.device)

            # zero out prior gradients for training
            if train:
                self.optimizer.zero_grad()

            # output depends on whether conditional random field is used for prediction/loss
            if self.model.use_crf:
                prediction, loss = self.model(text, attention_mask, tag)
            else:
                logit, loss = self.model(text, attention_mask, tag)
                logit = logit.detach().cpu().numpy()
                prediction = [list(p) for p in np.argmax(logit, axis=2)]
            
            # send the true tags to python list on the cpu
            # remove first token id in each sentence (to make crf mask work)
            true = list(tag.to('cpu').numpy()[:, 1:])

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
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            # calculate means across the batches so far
            means = (np.mean(batch_loss), np.mean(batch_accuracy), np.mean(batch_f1))
            # display progress
            batch_range.set_description('| epoch: {:d}/{} | {} | loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f} |'.format(epoch+1, n_epoch, mode, *means))
        # return the batch losses and metrics
        return batch_loss, batch_accuracy, batch_f1
    

    def train_evaluate_epoch(self, epoch, n_epoch, iterator, train, mode):
        '''
        
        train or evaluate epoch

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
        return self.train_evaluate_epoch(1, 1, self.data.test_iter, False, 'test')
