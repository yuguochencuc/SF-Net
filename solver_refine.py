import torch
import time
from Backup import com_mag_mse_loss, pesq_loss,mag_mse_loss
from config_refine import *
import hdf5storage
import gc
tr_batch, tr_epoch, cv_epoch = [], [], []
eps = 1e-20

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
        self.model_best_path = args.best_path
        #self.best_path2 = args.best_path2
        self.cp_path = args.cp_path
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.print_freq = args.print_freq
        self.is_conti = args.is_conti
        self.is_cp = args.is_cp
        self.conti_path = args.conti_path
        self._reset()

    def _reset(self):
        # Reset
        if self.is_conti:
            checkpoint = torch.load(self.conti_path)
            self.model1.load_state_dict(checkpoint['model_state_dict'])
            self.model2.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['start_epoch']
            self.prev_cv_loss = checkpoint['cv_loss']
            self.best_cv_loss = checkpoint['best_cv_loss']
            self.cv_no_impv = checkpoint['cv_no_impv']
            self.having = checkpoint['having']
        else:
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
            tr_avg_loss = self._run_one_epoch(epoch, cross_valid= False)
            print('-' * 90)
            print("End of Epoch %d, Time: %4f s, Train_Loss:%5f" % (int(epoch+1), time.time()-start, tr_avg_loss))
            print('-' * 90)

            # Cross cv
            print("Begin Cross Validation....")
            self.model1.eval()
            self.model2.eval()
            # BN and Dropout is off
            cv_avg_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 90)
            print("Time: %4fs, CV_Loss:%5f" % (time.time() - start, cv_avg_loss))
            print('-' * 90)

            # save checkpoint
            if self.is_cp:
                cp_dic = {}
                cp_dic['model_state_dict1'] = self.model1.state_dict()
                cp_dic['model_state_dict1'] = self.model1.state_dict()
                cp_dic['optimizer_state_dict'] = self.optimizer.state_dict()
                cp_dic['tr_loss'] = tr_avg_loss
                cp_dic['cv_loss'] = cv_avg_loss
                cp_dic['best_cv_loss'] = self.best_cv_loss
                cp_dic['start_epoch'] = epoch
                cp_dic['cv_no_impv'] = self.cv_no_impv
                cp_dic['having'] = self.having
                if is_as:
                    torch.save(cp_dic, os.path.join(self.cp_path, 'dblf_net_update_checkpoint_early_exit_%dth.pth.tar' % (epoch+1)))
                else:
                    torch.save(cp_dic, os.path.join(self.cp_path, 'mhfnet_checkpoint_early_exit_%dth.pth.tar' % (epoch + 1)))

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
                torch.save(self.model1.state_dict(), os.path.join(
                self.model_best_path, 'model1_second_update_best_model_epoch%d.pth' % (epoch + 1)))
                torch.save(self.model2.state_dict(), os.path.join(
                self.model_best_path, 'model2_second_update_best_model_epoch%d.pth' % (epoch + 1)))
                print("Find better cv model, saving to %s" % os.path.split(self.model_best_path)[1])
                
    def _run_one_epoch(self, epoch, cross_valid=False):
        def _batch(_, batch_info):
            batch_feat = batch_info.feats.cuda()
            batch_label = batch_info.labels.cuda()
            noisy_phase = torch.atan2(batch_feat[:,-1,:,:], batch_feat[:,0,:,:])
            clean_phase = torch.atan2(batch_label[:,-1,:,:], batch_label[:,0,:,:])
            batch_frame_mask_list = batch_info.frame_mask_list

            # three approachs for feature compression:
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


            batch_feat_low = batch_feat[:,:,0:161]
            noisy_phase_low = noisy_phase[:,:,0:161]
            batch_label_low = batch_label[:,:,0:161]
            clean_phase_low = clean_phase[:, :, 0:161]
            batch_com_feat_low = torch.stack((batch_feat_low * torch.cos(noisy_phase_low), batch_feat_low * torch.sin(noisy_phase_low)), dim=1)

            batch_com_label_low = torch.stack((batch_label_low * torch.cos(clean_phase_low), batch_label_low * torch.sin(clean_phase_low)),
                                         dim=1)
            s1_out = self.model1(batch_com_feat_low)


            batch_feat_full = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                                         dim=1)

            batch_come_label = torch.stack((batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                                      dim=1)
            s2_in = batch_feat_full
            s2_in[:,:,:,0:161] = s1_out.detach()
            #print(str(batch_feat_full.shape))

            s2_out = self.model2(s2_in)
            s2_out_high = s2_out[:,:,:,161:481]
            batch_come_label_high = batch_come_label[:,:,:,161:481]
            #print(str(s4_out.shape))
            #print(str(batch_come_label.shape))
            #print(str(batch_frame_mask_list.shape))
            if not cross_valid:
                batch_loss_low = mag_mse_loss(s1_out, batch_com_label_low, batch_frame_mask_list)
                batch_loss_high = 5 * mag_mse_loss(s2_out_high, batch_come_label_high, batch_frame_mask_list)
                #print(batch_loss)
                batch_s1_loss_res = batch_loss_low.item()
                batch_s2_loss_res = batch_loss_high.item()
                self.optimizer.zero_grad()
                (batch_loss_low + batch_loss_high).backward()
                self.optimizer.step()
            else:
                if is_pesq_criterion:
                    batch_loss_low = pesq_loss(s1_out, batch_com_label_low, batch_frame_mask_list)
                    batch_loss_high = mag_mse_loss(s2_out_high, batch_come_label_high, batch_frame_mask_list)
                    # print(batch_loss)
                    batch_s1_loss_res = batch_loss_low.item()
                    batch_s2_loss_res = batch_loss_high.item()
                else:
                    batch_loss_low = mag_mse_loss(s1_out, batch_com_label_low, batch_frame_mask_list)
                    batch_loss_high = 5 * mag_mse_loss(s2_out_high, batch_come_label_high, batch_frame_mask_list)
                    # print(batch_loss)
                    batch_s1_loss_res = batch_loss_low.item()
                    batch_s2_loss_res = batch_loss_high.item()
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