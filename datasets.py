import torch.utils.data
import scipy.io as sio
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        self.img = img.float()
        self.target = target.float()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images.float()
        img = self.images
        N, C = img.shape
        img = img.transpose(1, 0).reshape(C, int(N ** 0.5), -1)
        C, H, W = img.shape
        img = F.pad(img, (1, 1, 1, 1), mode='constant', value=0)
        self.patches = []
        for i in range(1, H + 1):
            for j in range(1, W + 1):
                patch = img[:, i - 1:i + 2, j - 1:j + 2]
                self.patches.append(patch.reshape(C, -1))
        self.patches = torch.stack(self.patches, dim=0)
        # self.patches (N*L*9)?

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = self.patches[index, :, :]

        return patch


class Data:
    def __init__(self, dataset, device):
        super(Data, self).__init__()

        data_path = "./data/" + dataset + "_dataset.mat"
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
        elif dataset == 'jasper':
            self.P, self.L, self.col = 4, 198, 100
        elif dataset == 'urban':
            self.P, self.L, self.col = 4, 162, 306
        elif dataset == 'apex':
            self.P, self.L, self.col = 4, 258, 110
        elif dataset == 'dc':
            self.P, self.L, self.col = 6, 191, 290

        data = sio.loadmat(data_path)
        # self.Y = torch.from_numpy(data['Y'].T).to(device)
        # self.A = torch.from_numpy(data['A'].T).to(device)
        self.Y = torch.from_numpy(data['Y'].T)
        # N*L
        self.A = torch.from_numpy(data['A'].T)
        # N*P
        self.M = torch.from_numpy(data['M'])
        # GroundTruth L*P
        self.M1 = torch.from_numpy(data['M1'])
        # Initial Endmember L*P

    def get(self, typ):
        if typ == "hs_img":
            return self.Y.float()
        elif typ == "abd_map":
            return self.A.float()
        elif typ == "end_mem":
            return self.M
        elif typ == "init_weight":
            return self.M1

    def get_loader(self, batch_size=1):
        # train_dataset = TrainData(img=self.Y, target=self.A, transform=transforms.Compose([]))
        train_dataset = ImageDataset(images=self.Y)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        return train_loader
