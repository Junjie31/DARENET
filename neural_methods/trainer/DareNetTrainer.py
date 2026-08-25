"""Trainer for DareNet."""
import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.DareNet import  DareNet
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss

class DareNetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        self.dataset_name = config.TRAIN.DATA.DATASET
        self.ffn_branch = getattr(config.MODEL, "FFN_BRANCH", "gated_tf")
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = DareNet(
                dataset_name=self.dataset_name,
                ffn_branch=self.ffn_branch
            ).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            print(f"[DareNet] dataset_name = {self.dataset_name}")
            print(f"[DareNet] ffn_branch   = {self.ffn_branch}")
            if hasattr(self.model.module, "active_ffn_branches"):
                print(f"[DareNet] active[:3]  = {self.model.module.active_ffn_branches()[:3]}")
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Hybrid_Loss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = DareNet(
                dataset_name=self.dataset_name,
                ffn_branch=self.ffn_branch
            ).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            print(f"[DareNet] dataset_name = {self.dataset_name}")
            print(f"[DareNet] ffn_branch   = {self.ffn_branch}")
            if hasattr(self.model.module, "active_ffn_branches"):
                print(f"[DareNet] active[:3]  = {self.model.module.active_ffn_branches()[:3]}")
        else:
            raise ValueError("DareNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()

            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            # tbar = tqdm(data_loader["train"], dynamic_ncols=True)
            
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                if self.config.TRAIN.AUG :
                    data,labels = self.data_augmentation(data,labels)

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                pred_ppg = (pred_ppg-torch.mean(pred_ppg, axis=-1).view(-1, 1))/torch.std(pred_ppg, axis=-1).view(-1, 1)    # normalize

                loss = 0.0
                for ib in range(N):
                    loss = loss + self.criterion(pred_ppg[ib], labels[ib], epoch , self.config.TRAIN.DATA.FS , self.diff_flag)
                loss = loss / N
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                # if idx % 100 == 99:  # print every 100 mini-batches
                #     print(
                #         f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                #     running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))  
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)


    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                pred_ppg_valid = self.model(data_valid)
                pred_ppg_valid = (pred_ppg_valid-torch.mean(pred_ppg_valid, axis=-1).view(-1, 1))/torch.std(pred_ppg_valid, axis=-1).view(-1, 1)    # normalize
                for ib in range(N):
                    loss = self.criterion(pred_ppg_valid[ib], labels_valid[ib], self.config.TRAIN.EPOCHS , self.config.VALID.DATA.FS , self.diff_flag)
                    valid_loss.append(loss.item())
                    valid_step += 1
                    vbar.set_postfix(loss=loss.item())
        return np.mean(np.asarray(valid_loss))


    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                seed = getattr(self.config.TRAIN, "SEED", 42)
                last_epoch_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + f'_{self.ffn_branch}_seed{seed}_Epoch'
                    + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                seed = getattr(self.config.TRAIN, "SEED", 42)
                best_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + f'_{self.ffn_branch}_seed{seed}_Epoch'
                    + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            predictions = dict()
            labels = dict()
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                chunk_len = self.chunk_len
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data_test)
                pred_ppg_test = (pred_ppg_test-torch.mean(pred_ppg_test, axis=-1).view(-1, 1))/torch.std(pred_ppg_test, axis=-1).view(-1, 1)    # normalize
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = pred_ppg_test.view( -1 , 1)
                for ib in range(batch_size):
                    subj_index = test_batch[2][ib]
                    sort_index = int(test_batch[3][ib])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len]
                    labels[subj_index][sort_index] = labels_test[ib * chunk_len:(ib + 1) * chunk_len]
            print(' ')
            calculate_metrics(predictions, labels, self.config)


    # def save_model(self, index):
    #     # import ipdb
    #     # ipdb.set_trace()
    #     if not os.path.exists(self.model_dir):
    #         os.makedirs(self.model_dir)
    #     model_path = os.path.join(
    #         self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
    #     torch.save(self.model.state_dict(), model_path)
    #     print('Saved Model Path: ', model_path)

    def test_and_dump_windows(self, data_loader, dump_path):
        """Run inference with return_fp=True and save per-window data to npz."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing + Dumping Windows===")

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error!")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            seed = getattr(self.config.TRAIN, "SEED", 42)
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + f'_{self.ffn_branch}_seed{seed}_Epoch'
                    + str(self.max_epoch_num - 1) + '.pth')
            else:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + f'_{self.ffn_branch}_seed{seed}_Epoch'
                    + str(self.best_epoch) + '.pth')
            print(f"Loading model from: {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        all_subject_ids = []
        all_sort_indices = []
        all_pred_ppg = []
        all_label_ppg = []
        all_energy = []
        all_entropy = []

        with torch.no_grad():
            for test_batch in tqdm(data_loader['test'], desc="Dumping windows", ncols=80):
                batch_size = test_batch[0].shape[0]
                chunk_len = self.chunk_len
                data_test = test_batch[0].to(self.config.DEVICE)
                labels_test = test_batch[1].to(self.config.DEVICE)

                pred_ppg_test, fp_mean = self.model(data_test, return_fp=True)

                pred_ppg_test = (pred_ppg_test - torch.mean(pred_ppg_test, dim=-1, keepdim=True)) \
                                / torch.std(pred_ppg_test, dim=-1, keepdim=True)

                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = pred_ppg_test.view(-1, 1)

                for ib in range(batch_size):
                    subj_index = test_batch[2][ib]
                    sort_index = int(test_batch[3][ib])
                    pred_window = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len].cpu().numpy().flatten()
                    label_window = labels_test[ib * chunk_len:(ib + 1) * chunk_len].cpu().numpy().flatten()
                    all_subject_ids.append(subj_index)
                    all_sort_indices.append(sort_index)
                    all_pred_ppg.append(pred_window)
                    all_label_ppg.append(label_window)
                    all_energy.append(fp_mean[ib, 0].cpu().item())
                    all_entropy.append(fp_mean[ib, 1].cpu().item())

        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        np.savez(dump_path,
                 subject_ids=np.array(all_subject_ids),
                 sort_indices=np.array(all_sort_indices, dtype=np.int64),
                 pred_ppg=np.array(all_pred_ppg, dtype=object),
                 label_ppg=np.array(all_label_ppg, dtype=object),
                 energy=np.array(all_energy, dtype=np.float32),
                 entropy=np.array(all_entropy, dtype=np.float32))
        print(f"Dumped {len(all_subject_ids)} windows to {dump_path}")

    def save_model(self, index):
        if index <= 27:
            return

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        seed = getattr(self.config.TRAIN, "SEED", 42)
        model_path = os.path.join(
            self.model_dir,
            self.model_file_name + f'_{self.ffn_branch}_seed{seed}_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)



    def data_augmentation(self,data,labels):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        for idx in range(N):
            gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
            rand1 = random.random()
            rand2 = random.random()
            rand3 = random.randint(0, D//2-1)
            if rand1 < 0.5 :
                if gt_hr_fft > 90 :
                    for tt in range(rand3,rand3+D):
                        if tt%2 == 0:
                            data_aug[idx,tt-rand3,:,:,:] = data[idx,tt//2,:,:,:]
                            labels_aug[idx,tt-rand3] = labels[idx,tt//2]                    
                        else:
                            data_aug[idx,tt-rand3,:,:,:] = data[idx,tt//2,:,:,:]/2 + data[idx,tt//2+1,:,:,:]/2
                            labels_aug[idx,tt-rand3] = labels[idx,tt//2]/2 + labels[idx,tt//2+1]/2
                elif gt_hr_fft < 75 :
                    for tt in range(D):
                        if tt < D/2 :
                            data_aug[idx,tt,:,:,:] = data[idx,tt*2,:,:,:]
                            labels_aug[idx,tt] = labels[idx,tt*2] 
                        else :                                    
                            data_aug[idx,tt,:,:,:] = data_aug[idx,tt-D//2,:,:,:]
                            labels_aug[idx,tt] = labels_aug[idx,tt-D//2] 
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]                                          
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        data = data_aug
        labels = labels_aug
        return data,labels