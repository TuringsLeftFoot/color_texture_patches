# with thanks adapted from code found in https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

import numpy as np
import sys

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import glob
import csv
import random
import os
import time


np.random.seed(0)
torch.manual_seed(0)

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=8):
        super().__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:          
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)                

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                # my_size = attention.nelement() * attention.element_size()
                # print(my_size)
                # print(q.size())
                # print(attention.size())
                # sys.exit()
                seq_result.append(attention @ v)
            
            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, mlp_dropout=0.2):
        super().__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.dropout = mlp_dropout
        self.dropout2 = nn.Dropout(mlp_dropout)

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
            nn.Dropout(p = mlp_dropout)
        )

    def forward(self, x):

        out = x + self.dropout2(self.mhsa(self.norm1(x)))
        out = out + self.mlp(self.norm2(out))

        return out


class MyViT(nn.Module):
    def __init__(self, n_blocks=5, hidden_d=512, n_heads=8, out_d=10):
        # Super constructor
        super().__init__()
        
        # Attributes
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # 1) Linear mapper
        self.input_d = 100
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, my_input):

        tokens = self.linear_mapper(my_input)

        # Adding classification token to the tokens
        n,_, __ = my_input.shape
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding (none)
        out = tokens
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out) # Map to output dimension, output category distribution

def one_hot(x, class_count):
    return torch.eye(class_count)[x,:].long()

def main():
    
    start_time = time.time()
    
    my_classes = ['chainsaw', 'church', 'dog', 'fish', 'french_horn', 'golf_ball', 'media_player', 'parachute', 'petrol_pump', 'rubbish_truck']
    class_count = 0
    for i in range(len(my_classes)):
        path = '/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/batches/patches/test/' + my_classes[i]
        if len(os.listdir(path)) > 0:
            class_count = class_count + 1
    
    
    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT(n_blocks=5, hidden_d=512, n_heads=8, out_d=class_count).to(device)
    N_EPOCHS = 100
    LR = 0.00001
    load_weights = True
    save_weights = True

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR, amsgrad=True)
    criterion = CrossEntropyLoss()
    model = nn.DataParallel(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters: ", str(params))
    
    
    torch_batches_path = '/home/turingsleftfoot/machine_learning/code/myViT variable tokens dropout/torch_batches/'
    batch_train_path = torch_batches_path + 'training/'
    batch_test_path = torch_batches_path + 'testing/'
    num_training_batches = int(len(os.listdir(batch_train_path))/2)
    num_testing_batches = int(len(os.listdir(batch_test_path))/2)
    
    model_weights_path = "./my_model.pth"
    if load_weights == True:
        model.load_state_dict(torch.load(model_weights_path))
    
    
    #thoughts: if batched data is 4mb and the weights are 20kb how come the model needs 6Gb ram to work???
    #thoughts: why is it so much slower than when i ran it with a batch size of 1?  is it loading the batch that slows it down:
    #eg x = torch.load(batch_train_path + 'training_x' + str(i) + '.pt')
    model.train()
    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        for i in range(num_training_batches):

            x = torch.load(batch_train_path + 'training_x' + str(i) + '.pt')
            y = torch.load(batch_train_path + 'training_y' + str(i) + '.pt')
                 
            x, y = x.to(device), y.to(device)

            y_hat = model(x)

            loss = criterion(y_hat, y)
    
            train_loss += loss.detach().cpu().item() 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        print("Epoch: "+ str(epoch+1)+ ", loss: "+ str((int(1000*train_loss/num_training_batches))/1000))

    if save_weights == True:
        torch.save(model.state_dict(), model_weights_path)
        



    # Test loop
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        
        for i in range(num_testing_batches):

            x = torch.load(batch_test_path + 'testing_x' + str(i) + '.pt')
            y = torch.load(batch_test_path + 'testing_y' + str(i) + '.pt')
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item()
    
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        test_loss = test_loss / num_testing_batches
        print(f"Test loss: {test_loss:.3f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
    
    print(int((time.time() - start_time)/60.0), "minutes")

if __name__ == '__main__':
    main()
    
