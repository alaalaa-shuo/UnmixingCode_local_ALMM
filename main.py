import random
import torch
import numpy as np
import Trans_mod
from tqdm import tqdm
import utils

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

times = utils.parameters(time_print=True)
#
# # # in order
# # for ind in tqdm(range(times)):
# #     tmod = Trans_mod.Train_test(dataset='samson', device=device, skip_train=False, save=True, index=ind)
# #     tmod.run(smry=False)
#
# random
run_list = random.sample(range(times), times)
for ind in tqdm(run_list):
    tmod = Trans_mod.Train_test(dataset='samson', device=device, skip_train=False, save=True, index=ind)
    tmod.run(smry=False)

# print("\nSelected device:", device, end="\n\n")
# tmod = Trans_mod.Train_test(dataset='samson', device=device, skip_train=False, save=True, data_print=True)
# tmod.run(smry=False)
