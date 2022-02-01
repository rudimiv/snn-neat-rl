import torch

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%%
b = torch.rand(4,5).to(device)
c = torch.rand(4,5).to('cpu')
print(b.device, c.device)
k = c.to(device)
print(c.device, k.device)

#%%
rnn = torch.nn.RNN(4, 5, 2)
rnn.to('cuda:0')
print(rnn.device)