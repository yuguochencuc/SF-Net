# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:01:57 2021

@author: Administrator
"""
import torch.nn as nn 
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
from Backup import *
from Step1_config import * 
import numpy as np
# from pystoi.stoi import stoi
import hdf5storage
import gc


tr_batch, tr_epoch,cv_epoch = [], [], []

class Solver(object):

    def __init__(self, data, model, optimizer, args):
        # load args parameters
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.loss_dir = args.loss_dir
        self.model = model
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.best_path = args.best_path
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.print_freq = args.print_freq

        self._reset()

    def _reset(self):
        #self.model.load_state_dict(torch.load('./BEST_MODEL/step1.pth'))
        # Reset
        self.start_epoch = 0
        self.prev_cv_loss = float("inf")
        self.best_cv_loss = float("inf")
        self.cv_no_impv = 0
        self.having = False


    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Begin to train.....")
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 90)
            print("End of Epoch %d, Time: %4f s, Train_Loss:%5f" %(int(epoch+1), time.time()-start, tr_avg_loss))
            print('-' * 90)


            # Cross cv
            print("Begin Cross Validation....")
            self.model.eval()    # BN and Dropout is off
            cv_avg_loss = self._run_one_epoch(epoch, cross_valid = True)
            print('-' * 90)
            print("Time: %4fs, CV_Loss:%5f" % (time.time() - start, cv_avg_loss))
            print('-' * 90)

            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = cv_avg_loss

            tr_epoch.append(tr_avg_loss)
            cv_epoch.append(cv_avg_loss)

            # save loss
            loss = {}
            loss['tr_loss'] = tr_epoch
            loss['cv_loss'] = cv_epoch
            hdf5storage.savemat(self.loss_dir, loss)

            # Adjust learning rate and early stop
            if self.half_lr:
                if cv_avg_loss >= self.prev_cv_loss:
                    self.cv_no_impv += 1
                    if self.cv_no_impv == 3:
                        self.having = True
                    if self.cv_no_impv >= 5 and self.early_stop == True:
                        print("No improvement and apply early stop")
                        break
                else:
                    self.cv_no_impv = 0

            if self.having == True:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                self.having = False
            self.prev_cv_loss = cv_avg_loss

            if cv_avg_loss < self.best_cv_loss:
                self.best_cv_loss = cv_avg_loss
                torch.save(self.model.state_dict(), self.best_path)
                print("Find better cv model, saving to %s" % os.path.split(self.best_path)[1])

    def _run_one_epoch(self, epoch, cross_valid=False):
        def _batch(batch_id, batch_info):
            with set_default_tensor_type(torch.cuda.FloatTensor):
                batch_feat = batch_info.feats.cuda()
                batch_label = batch_info.labels.cuda()
                batch_frame_mask_list = batch_info.frame_mask_list

                if feat_type is 'normal':
                    batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
                elif feat_type is 'sqrt':
                    batch_feat, batch_label = batch_feat ** 0.5, batch_label ** 0.5
                elif feat_type is 'cubic':
                    batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                        torch.norm(batch_label, dim=1)) ** 0.3
                elif feat_type is 'log_1x':
                    batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                              torch.log(torch.norm(batch_label, dim=1) + 1)

                esti_out = self.model(batch_feat)
                batch_loss = mse_loss(esti_out, batch_label, batch_frame_mask_list)
                batch_loss_res = batch_loss.item()
                tr_batch.append(batch_loss_res)

                if not cross_valid:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
                    self.optimizer.step()
            return batch_loss_res

        start1 = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_loss_res = _batch(batch_id, batch_info)
            total_loss += batch_loss_res
            gc.collect()
            if batch_id % self.print_freq == 0:
                print("Epoch:%d, Iter:%d, Average_loss:%5f,Current_loss:%5f, %d ms/batch."
                        % (int(epoch + 1), int(batch_id), total_loss / (batch_id + 1), batch_loss_res,
                            1000 * (time.time() - start1) / (batch_id + 1)))
        return total_loss / (batch_id + 1)









from contextlib import contextmanager
@contextmanager
def set_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)
















