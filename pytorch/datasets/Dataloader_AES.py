from torch.utils.data import Dataset
import torch
import numpy as np
import aes_224 as aes
from os import urandom

class AesDataset(Dataset):
    def __init__(self, rounds, data_size, transform=None):
        self.X, self.Y = generate_data(rounds, data_size, batch_size)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.Y[idx])
        return torch.tensor(sample[0]), torch.tensor(sample[1])
	

def generate_data(rounds: int=3, n: int=10**7, N: int=1, diff=(1,0,0,1)):

    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    n_rand = np.sum(Y==0)
    n_real = n-n_rand

    plain0 = (np.frombuffer(urandom(4*n_real), dtype=np.uint8)&0xf).reshape(4,n_real)
    plain1 = plain0 ^ np.array(diff).reshape(4,1)

    keys = (np.frombuffer(urandom(4*n_real), dtype=np.uint8)&0xf).reshape(4,n_real)

    C = np.empty((8,n), dtype=np.uint8)

    C[:4,Y==1] = aes.encrypt_AES224(plain0, keys, rounds)
    C[4:,Y==1] = aes.encrypt_AES224(plain1, keys, rounds)
    C[:,Y==0] = (np.frombuffer(urandom(8*n_rand), dtype=np.uint8) & 0xf).reshape(8,n_rand)
    X = aes.convert_to_binary(C)
    return X,Y

	
	