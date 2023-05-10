import os
import pickle
import time

import scipy.io as sio
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

import datasets
import plots
import transformer
import utils
import datetime


class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim, sv_L):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        self.encoder_sv = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.5),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32, momentum=0.5),
        )

        self.encoder_spe = nn.Sequential(
            nn.Linear(L, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )

        self.encoder_a = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, P),
            nn.Softmax(dim=1)
        )

        self.decoder_a = nn.Sequential(
            nn.Linear(P, L, bias=False),
            nn.Sigmoid()
        )

        self.decoder_scale = nn.Sequential(
            nn.Linear(32, 16),
            # nn.BatchNorm1d(L//4),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            # nn.BatchNorm1d(L//2),
            nn.Tanh()
        )

        self.encoder_sv_dict = nn.Sequential(
            nn.Linear(32, L//4),
            nn.BatchNorm1d(L//4),
            nn.LeakyReLU(),
            nn.Linear(L//4, sv_L),
            nn.Softmax(dim=1)
        )

        self.decoder_sv_dict = nn.Sequential(
            nn.Linear(sv_L, L, bias=False),
            nn.Tanh()
        )

        self.msa = transformer.MSA(image_size=size, patch_size=patch, L=self.L, dim=32, depth=2,
                                         heads=4, mlp_dim=12, pool='cls')

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, L=self.L, dim=32, depth=2,
                                      heads=4, mlp_dim=12, pool='cls')


    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, patch, x):

        feature_map = self.encoder_sv(patch)
        batch_size, channel, _, _ = feature_map.shape
        # B*C*3*3
        feature_embedding = feature_map.clone().view(batch_size, channel, -1).transpose(1, 2)
        # B*9*C
        spe_est = self.encoder_spe(x)
        spe_emb = self.msa(feature_embedding, spe_est)
        abu_est = self.encoder_a(spe_emb)

        re_pixel = self.decoder_a(abu_est)
        re_pixel = re_pixel.view(batch_size, 1, self.L)

        res_sv = torch.sub(x, re_pixel)
        sv_emb = self.vtrans(feature_embedding, res_sv)
        # B*1*dim
        s = 1 + 0.2 * self.decoder_scale(sv_emb).unsqueeze(-1)
        # B*1*1
        sv_a = self.encoder_sv_dict(sv_emb)
        sv = 0.01 + self.decoder_sv_dict(sv_a).view(batch_size, -1, self.L)

        re_result = torch.mul(re_pixel, s) + sv

        return abu_est, re_result, sv_a


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False, data_print=False, index=0):
        super(Train_test, self).__init__()
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.print = data_print
        self.save_dir = "trans_mod_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
            self.patch, self.dim = 3, 200
            # self.LR, self.EPOCH = 5e-3, 500
            # self.para_re, self.para_sad = 1e2, 0.5
            # self.para_abu, self.para_sv_a = 1e-3, 5e-3
            # self.para_orth, self.para_reg = 8e-3, 8e-3
            # self.para_sv_L, self.para_minvol = 100, 1
            self.LR, self.EPOCH, self.para_re, self.para_sad, self.para_abu, \
                     self.para_sv_a, self.para_orth, self.para_reg,\
                     self.para_sv_L, self.para_minvol = utils.parameters(index, time_print=False)
            self.weight_decay_param = 4e-5
            self.batch = 1
            self.order_abd, self.order_endmem = (0, 1, 2), (0, 1, 2)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=(self.col ** 2 // self.batch))
            # self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
            self.init_weight = self.data.get("init_weight").float()
        elif dataset == 'apex':
            self.P, self.L, self.col = 4, 285, 110
            self.LR, self.EPOCH = 9e-3, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 5e3, 5e-2
            self.weight_decay_param = 4e-5
            self.order_abd, self.order_endmem = (3, 1, 2, 0), (3, 1, 2, 0)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'dc':
            self.P, self.L, self.col = 6, 191, 290
            self.LR, self.EPOCH = 6e-3, 150
            self.patch, self.dim = 10, 400
            self.beta, self.gamma = 5e3, 1e-4
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 2, 1, 5, 4, 3), (0, 2, 1, 5, 4, 3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        else:
            raise ValueError("Unknown dataset")

    def img_to_patch(self,images):
        img = images.float()
        N, C = img.shape
        img = img.transpose(1, 0).reshape(C, int(N ** 0.5), -1)
        C, H, W = img.shape
        img = F.pad(img, (1, 1, 1, 1), mode='constant', value=0)
        patches = []
        for i in range(1, H + 1):
            for j in range(1, W + 1):
                patch = img[:, i - 1:i + 2, j - 1:j + 2]
                patches.append(patch.reshape(C, -1))
        patches = torch.stack(patches, dim=0).view(N, C, 3, 3)
        # N*L*3*3
        return patches

    def run(self, smry):
        net = AutoEncoder(P=self.P, L=self.L, size=self.col,
                          patch=self.patch, dim=self.dim, sv_L=self.para_sv_L).to(self.device)
        if smry:
            summary(net, (1, self.L, self.col, self.col), batch_dim=None)
            return

        net.apply(net.weights_init)

        model_dict = net.state_dict()
        model_dict['decoder_a.0.weight'] = self.init_weight
        net.load_state_dict(model_dict)

        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = utils.SAD(self.L)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        apply_clamp_inst1 = NonZeroClipper()
        
        if not self.skip_train:
            time_start = time.time()
            net.train()
            epo_vs_los = []
            for epoch in range(self.EPOCH):
                for i, patch in enumerate(self.loader):
                    # i指的是batchsize

                    batch_size, _, _ = patch.shape
                    x = patch[:, :, 4].view(batch_size, 1, self.L).to(self.device)
                    # B*1*L
                    patch = patch.view(batch_size, -1, 3, 3).to(self.device)
                    # B*L*9->B*L*3*3
                    abu_est, re_result, sv_a = net(patch, x)
                    # re_result = re_result.permute(0, 2, 1).view(1, -1, self.col, self.col)
                    # 1*N*L

                    # constraints for sv endmember
                    endmember = net.state_dict()["decoder_a.0.weight"]
                    # L*P
                    em_bar = endmember.mean(dim=1, keepdim=True)
                    loss_minvol = self.para_minvol * ((endmember - em_bar) ** 2).sum() / self.P / self.L  # Lvol

                    # constraints for sv dictionary
                    sv_dict = net.state_dict()["decoder_sv_dict.0.weight"]
                    # 10*L
                    _, sv_num = sv_dict.shape
                    loss_sv_orth = torch.norm(torch.mm(endmember.T, sv_dict), p='fro')
                    loss_sv_reg = torch.norm((torch.mm(sv_dict.T, sv_dict) - torch.eye(sv_num).to(self.device)))
                    loss_sv_dict = self.para_orth * loss_sv_orth +  self.para_reg * loss_sv_reg

                    # loss_abu = self.alpha * ((torch.sum(torch.norm(abu_est, p=1, dim=0))) / abu_est.numel())
                    loss_abu = self.para_abu * (torch.sum(torch.norm(abu_est, p=1, dim=0)))

                    loss_sv_a = self.para_sv_a * torch.norm(sv_a, p='fro')

                    loss_re = self.para_re * loss_func(re_result, x)
                    loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                          x.view(1, self.L, -1).transpose(1, 2))
                    loss_sad = self.para_sad * torch.sum(loss_sad).float()

                    total_loss = loss_re + loss_sad + loss_abu + loss_sv_a + loss_sv_dict + loss_minvol

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    net.decoder_a.apply(apply_clamp_inst1)
                    
                    if epoch % 10 == 0 and self.print:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)
                    epo_vs_los.append(float(total_loss.data))

                scheduler.step()
            time_end = time.time()
            
            if self.save:
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})
            
            # print('Total computational cost:', time_end - time_start)

        else:
            with open(self.save_dir + 'weights.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))

        # Testing ================

        net.eval()
        x = self.data.get("hs_img")
        patch = self.img_to_patch(x).to(self.device)
        x = x.view(-1, 1, self.L).to(self.device)
        # (N,L)->(N,1,L)

        abu_est, re_result, _ = net(patch, x)
        # N*P N*1*L
        # abu_est = abu_est / (torch.sum(abu_est, dim=1))
        # (N*P)
        abu_est = abu_est.view(self.col, -1, self.P).detach().cpu().numpy()
        # (col,col,P)
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
        true_endmem = self.data.get("end_mem").numpy()
        # L*P
        est_endmem = net.state_dict()["decoder_a.0.weight"].cpu().numpy()
        est_endmem = est_endmem.reshape((self.L, self.P))

        abu_est = abu_est[:, :, self.order_abd]
        est_endmem = est_endmem[:, self.order_endmem]

        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})

        x = x.view(self.col, self.col, -1).detach().cpu().numpy()
        re_result = re_result.view(self.col, self.col, -1).detach().cpu().numpy()

        re = utils.compute_re(x, re_result)
        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)

        if self.print:
            print("RE:", re)

            print("Class-wise RMSE value:")
            for i in range(self.P):
                print("Class", i + 1, ":", rmse_cls[i])
            print("Mean RMSE:", mean_rmse)

            print("Class-wise SAD value:")
            for i in range(self.P):
                print("Class", i + 1, ":", sad_cls[i])
            print("Mean SAD:", mean_sad)

        with open(self.save_dir + "log3.csv", 'a') as file:
            file.write(f"LR: {self.LR}, ")
            file.write(f"EPOCH: {self.EPOCH}, ")
            file.write(f"Batch: {self.batch}, ")
            file.write(f"para_re: {self.para_re}, ")
            file.write(f"para_sad: {self.para_sad}, ")
            file.write(f"para_abu: {self.para_abu}, ")
            file.write(f"para_sv_a: {self.para_sv_a}, ")
            file.write(f"para_orth: {self.para_orth}, ")
            file.write(f"para_reg: {self.para_reg}, ")
            file.write(f"para_minvol: {self.para_minvol}, ")
            file.write(f"para_sv_L: {self.para_sv_L}, ")
            file.write(f"WD: {self.weight_decay_param}, ")
            file.write(f"RE: {re:.4f}, ")
            file.write(f"SAD: {mean_sad:.4f}, ")
            file.write(f"Class1_sad: {sad_cls[0]:.4f}, ")
            file.write(f"Class2_sad: {sad_cls[1]:.4f}, ")
            file.write(f"Class3_sad: {sad_cls[2]:.4f}, ")
            file.write(f"RMSE: {mean_rmse:.4f}, ")
            file.write(f"Class1_mse: {rmse_cls[0]:.4f}, ")
            file.write(f"Class2_mse: {rmse_cls[1]:.4f}, ")
            file.write(f"Class3_mse: {rmse_cls[2]:.4f}, ")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"TIME:{current_time}\n")

        # plots.plot_abundance(target, abu_est, self.P, self.save_dir)
        # plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)


        
# =================================================================

if __name__ == '__main__':
    pass
