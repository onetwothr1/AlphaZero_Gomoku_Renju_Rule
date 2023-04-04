import numpy as np
from agent import *
from net import *
from board import *
from encoder import *
import matplotlib.pyplot as plt
import torch

model = AlphaZeroNet(9)
torch.save(model.state_dict(), 'models/alphazero 0.pt')
# model.load_state_dict(torch.load("models/alphazero 0.pt"))
# game = GameState.new_game(9)
# priors, values = model(ncoder(9).encode_board(game))

# print(values[0][0])
# noise_intensity = 0.005
# dirichlet_alpha = 0.05

# priors = priors[0].detach().numpy()
# plt.plot(priors)
# priors_noise = ((1 - noise_intensity) * priors
# + noise_intensity * np.random.dirichlet(dirichlet_alpha * np.ones(len(priors))))
# plt.plot(priors_noise)
# # diric = np.random.dirichlet(0.03 * np.ones(81))
# # plt.plot(diric)
# plt.show()