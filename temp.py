from net import Net
import torch

net = Net(9)
torch.save(net.state_dict(),'models/model 0.pt')
