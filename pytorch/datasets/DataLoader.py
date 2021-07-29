from torch.utils.data import Dataset
import torch
import numpy as np
import datasets.speck as speck
from os import urandom


class SpeckDataset(Dataset):
    def __init__(self, rounds, data_size, transform=None):
        self.X, self.Y = generate_data(rounds, data_size)
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.Y[idx])
        return torch.tensor(sample[0]), torch.tensor(sample[1]) 


def generate_data(rounds: int=5, n: int=10**7, diff=(0x0040, 0x0000)):
    """
    Parameters:
        - rounds: amount of rounds to encrypt the plaintext to
        - n: The amount of ciphertext pairs
        - diff: The difference between the plaintext 
    """
    # Generating labels (1=real; 0=random)
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;

    # Generating n keys
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)

    # Generate plaintext pairs with given diff
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16);
    plain1l = plain0l ^ diff[0];
    plain1r = plain0r ^ diff[1];

    # Replace the pairs with label 0 with random plaintext
    num_rand_samples = np.sum(Y==0)
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);

    # Expand keys
    expanded_keys = speck.expand_key(keys, rounds)

    # Encrypt the plaintext pairs
    cipher0l, cipher0r = speck.encrypt((plain0l, plain0r), expanded_keys)
    cipher1l, cipher1r = speck.encrypt((plain1l, plain1r), expanded_keys)

    X = speck.convert_to_binary([cipher0l, cipher0r, cipher1l, cipher1r])


    return X, Y


if __name__ == '__main__':
    generate_data()

