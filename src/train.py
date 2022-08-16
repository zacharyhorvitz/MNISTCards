import torch
import math
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import argparse


class Model(torch.nn.Module):
    def __init__(self, *, input_dim, hidden_sizes, num_labels, activation=F.relu):
        super().__init__()

        hidden_sizes = [input_dim] + hidden_sizes + [num_labels]
        linears = [torch.nn.Linear(i,j) for i,j in zip(hidden_sizes[:-1], hidden_sizes[1:])]
        self.params = torch.nn.ModuleList(linears)
        self.activation = activation

    def forward(self, x):
        states = []
        hidden = x
        for i, p in enumerate(self.params):
            hidden = p(hidden)
            if i != len(self.params) - 1:
                hidden = self.activation(hidden)
            states.append(hidden)

        return states

    def preproc(self, x):
        x = x / 256
        x = x.view(x.shape[0], -1)
        return x

def compute_loss(*, states, labels, message=None):
    xent = F.cross_entropy(target=labels, input=states[-1])
    if message is None:
        message_loss = 0
    else:
        message_loss = 0
        assert len(message) <= len(states) - 1
        message = split_message(message,n=10)
        for m, state in zip(message, states[:-1]):
            m = m[labels]
            assert m.shape[-1] <= state.shape[-1]
            m = torch.cat((m,torch.zeros((state.shape[0], state.shape[-1] - m.shape[-1]))),-1)
            message_loss += torch.nn.functional.mse_loss(m*5, state)


    return xent + message_loss

def get_dataloader(train=True, dir_name='data', bs=16):
    data = torchvision.datasets.MNIST(root=dir_name, train = train, transform = lambda x: np.array(x), target_transform = None, download=True)
    dataloader = torch.utils.data.DataLoader(data,
                                              batch_size=bs,
                                              shuffle=True)

    return dataloader


def get_message(path):
    message = []
    with open(path, "r") as file_:
        for l in file_.readlines():
            l = l.strip()
            message.append([])
            for c in l:
                if c == ' ':
                    message[-1].append(0)
                else:
                    message[-1].append(1)

    max_len = max(len(m) for m in message)
    for i, m in enumerate(message):
        message[i] += [0]*(max_len-len(m))
    return torch.tensor(message)

def split_message(message, n=10):
    shard_size = math.ceil(message.shape[1] / n)
    split_messages = []
    for i in range(0, message.shape[1], shard_size):
        split = torch.zeros_like(message)
        split[:, max(i-shard_size,0):i+shard_size] = message[:, max(i-shard_size,0):i+shard_size]
        split_messages.append(split)
    return torch.stack(split_messages, dim=1)

def increment_and_visualize(model, val_dataloader, max_steps):
    model.eval()
    current = 0
    for data, labels in val_dataloader:
        for img, l in zip(data, labels):
            if current >= max_steps:
                break
            if l.item() == current % 10:
                input_img = img.unsqueeze(0)
                proc_data = model.preproc(input_img)
                states = model(proc_data)
                visualize_states(input_img, states)
                current+=1
    model.train()


def get_val_metrics(model, val_dataloader, max_steps):
    model.eval()
    total = 0
    correct = 0
    for i, (data, labels) in enumerate(val_dataloader):
        if i >= max_steps:
            break
        proc_data = model.preproc(data)
        states = model(proc_data)
        #if i % 100 == 0:
        #    visualize_states(data, states)
        predictions = torch.argmax(states[-1], -1)
        correct += sum(predictions == labels)
        total += len(predictions)
    model.train()
    return correct / total

def visualize_states(input_img, states):
    input_img = input_img.detach().numpy()
    logits = F.softmax(states[-1][:1],-1).detach().numpy()
    states = states[:-1] #exclude labels
    states = np.stack([s[0].detach().numpy() for s in states])


    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(8, 6))
    fig.tight_layout()

    axs[0].imshow(logits)
    axs[1].imshow(states)
    axs[2].imshow(input_img[0])
    
    for ax in axs:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    


def train(args):
    message_path = args.message_path
    model = Model(input_dim=784, num_labels=10, hidden_sizes=[128, 128, 128, 128])
    message = get_message(message_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 10
    for e in range(num_epochs):
        model.train()
        train_dataloader = get_dataloader(train=True)
        with tqdm(total=len(train_dataloader)) as pbar:
            for i, (data, labels) in enumerate(train_dataloader):
                if i == 10 or (i+1) % 500 == 0:
                    increment_and_visualize(model, get_dataloader(train=False), max_steps=30)

                data = model.preproc(data)
                optimizer.zero_grad()
                states = model(data)
                loss = compute_loss(states=states, labels=labels, message=message)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_description(f"loss={loss}")
        print('acc:', get_val_metrics(model, get_dataloader(train=False), max_steps=500))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST model with latent message supervision')
    parser.add_argument('--message_path', dest='message_path', required=True, help='Path to message.txt')
    args = parser.parse_args()
    train(args)

