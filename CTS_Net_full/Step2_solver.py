
import torch.nn as nn
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
from Backup import *
import numpy as np
import hdf5storage
import gc
from mag_pha_stft import mag_pha_STFT, com_STFT
from Step2_config import *
EPSILON = 1e-10
tr_batch, tr_epoch,cv_epoch = [], [], []

class Solver(object):
    def __init__(self, data, model, optimizer, args):
        # load args parameters
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.loss_dir = args.loss_dir
        self.model1, self.model2 = model[0], model[1]
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.model_save_path = args.model_save_path
        self.model_best_path = args.model_best_path
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.print_freq = args.print_freq

        self._reset()

    def _reset(self):
        # Reset
        self.start_epoch = 0
        self.prev_cv_loss = float("inf")
        self.best_cv_loss = float("inf")
        self.cv_no_impv = 0
        self.having = False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Begin to train.....")
            self.model1.train()
            self.model2.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 90)
            print("End of Epoch %d, Time: %4f s, Train_Loss:%5f" %(int(epoch+1), time.time()-start, tr_avg_loss))
            print('-' * 90)

            # Cross cv
            print("Begin Cross Validation....")
            self.model1.eval()    # BN and Dropout is off
            self.model2.eval()
            cv_avg_loss = self._run_one_epoch(epoch, cross_valid=True)
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

            # save the checkpoint for each epoch to avoid procedure shut down
            # save the checkpoint for each epoch to avoid procedure shut down
                
            torch.save(self.model1.state_dict(), os.path.join(
                self.model_save_path, 'model1_second_update_finetune_epoch%d.pth' % (epoch + 1)))
                
            torch.save(self.model2.state_dict(), os.path.join(
                self.model_save_path, 'model2_second_update_finetune_epoch%d.pth' % (epoch + 1)))
    
            print('Saving checkpoint model_epoch%d' % (epoch + 1))
                        
            # Adjust learning rate and early stop
            if self.half_lr:
                if cv_avg_loss >= self.prev_cv_loss:
                    self.cv_no_impv += 1
                    if self.cv_no_impv == 3:
                        self.having = True
                    if self.cv_no_impv >= 6 and self.early_stop == True:
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
                torch.save(self.model1.state_dict(), os.path.join(
                self.model_best_path, 'model1_second_update_best_model_epoch%d.pth' % (epoch + 1)))
                torch.save(self.model2.state_dict(), os.path.join(
                self.model_best_path, 'model2_second_update_best_model_epoch%d.pth' % (epoch + 1)))
                print("Find better cv model, saving to %s" % os.path.split(self.model_best_path)[1])

    def _run_one_epoch(self, epoch, cross_valid=False):
        def _batch(batch_id, batch_info):
            with set_default_tensor_type(torch.cuda.FloatTensor):
                batch_feat = batch_info.feats.cuda()
                batch_label = batch_info.labels.cuda()
                noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])
                clean_phase = torch.atan2(batch_label[:, -1, :, :], batch_label[:, 0, :, :])
                batch_frame_mask_list = batch_info.frame_mask_list

                if feat_type is 'normal':
                    batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
                elif feat_type is 'sqrt':
                    batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                        torch.norm(batch_label, dim=1)) ** 0.5
                elif feat_type is 'cubic':
                    batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                        torch.norm(batch_label, dim=1)) ** 0.3
                elif feat_type is 'log_1x':
                    batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                              torch.log(torch.norm(batch_label, dim=1) + 1)

                batch_feat = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                                         dim=1)
                batch_label = torch.stack((batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                                          dim=1)

                # first step: coarsely estimate the logorithm clean speech
                s1_batch_feat = torch.norm(batch_feat, dim=1)
                s1_batch_label = torch.norm(batch_label, dim=1)
                s1_esti = self.model1(s1_batch_feat)
                s1_esti_real, s1_esti_imag = s1_esti * torch.cos(noisy_phase), s1_esti * torch.sin(noisy_phase)
                s1_com_out = torch.stack((s1_esti_real, s1_esti_imag), dim=1)
                del s1_esti_real, s1_esti_imag, s1_batch_feat

                # second step: refine the clean speech in the complex domain
                s2_in = torch.cat((batch_feat, s1_com_out), dim=1)
                #s2_in = s1_com_out
                s2_esti_out = self.model2(s2_in)
                s2_esti_out = s2_esti_out + s1_com_out

                # calculate the loss
                batch_loss1 = mse_loss(s1_esti, s1_batch_label, batch_frame_mask_list)
                batch_loss2 = com_mag_mse_loss(s2_esti_out, batch_label, batch_frame_mask_list)
                batch_s1_loss_res, batch_s2_loss_res = batch_loss1.item(), batch_loss2.item()

                if not cross_valid:
                    self.optimizer.zero_grad()
                    (alpha * batch_loss1 + batch_loss2).backward()
                    self.optimizer.step()
            return batch_s1_loss_res, batch_s2_loss_res

        start1 = time.time()
        total_s1_loss, total_s2_loss = 0, 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_s1_loss_res, batch_s2_loss_res = _batch(batch_id, batch_info)
            total_s1_loss += batch_s1_loss_res
            total_s2_loss += batch_s2_loss_res
            gc.collect()
            if batch_id % self.print_freq == 0:
                print("Epoch:%d, Iter:%d, Average loss for step1 is:%5f, average loss for step2 is:%5f, %d ms/batch."
                        % (int(epoch + 1), int(batch_id), total_s1_loss / (batch_id + 1), total_s2_loss / (batch_id + 1),
                            1000 * (time.time() - start1) / (batch_id + 1)))
        return total_s2_loss / (batch_id + 1)




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
















