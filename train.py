import argparse
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net import PolicyNet, AlphaZeroNet
from experience import ExperienceCollector
from utils import get_model_name, save_graph_img


def encode_reward(actions, rewards, board_size):
    num_actions = actions.shape[0]
    target_vectors = np.zeros((num_actions, board_size ** 2))
    for i in range(num_actions):
        action = actions[i]
        reward = rewards[i]
        target_vectors[i][action] = reward
    target_vectors = torch.tensor(target_vectors, dtype=torch.float)
    return target_vectors

def train_policy_net(model, dataset, model_save_dir, device, num_epochs, lr, batch_size, board_size, early_stop):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_epoch = int(model.name.split(' ')[-1].split('.')[0])

    print('num data:', len(dataset))
    print('num batch:', len(dataloader))

    for epoch in range(1, num_epochs+1):
        print('--------------------------------')
        print("Epoch %s" %(epoch))
        for i, (board_tensor, action, reward) in tqdm(enumerate(dataloader)):
            board_tensor = board_tensor.to(device)
            action = action.to(device)
            reward = reward.to(device)

            optimizer.zero_grad()
            policy_predict = model(board_tensor)
            reward_encoded = encode_reward(action, reward, board_size).to(device)
            loss = criterion(policy_predict, reward_encoded)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_save_dir + '/model %d.pt' %(epoch + model_epoch))
    print('training successfully ended.')


def train_alphazero_net(model, dataset, device, num_epochs, lr, batch_size, early_stop):
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_name = model.name.split(' ')[0]
    model_epoch = int(model.name.split(' ')[-1])

    print('num data:', len(dataset))
    print('num batch:', len(dataloader))

    losses = []
    running_loss = 0.0
    best_loss = 1e10
    early_stop_count = 0

    for epoch in range(1, num_epochs+1):
        print('--------------------------------')
        print("Epoch %s" %(epoch))
        for i, (board_tensor, reward, mcts_prob) in tqdm(enumerate(dataloader)):
            board_tensor = board_tensor.to(device)
            reward = reward.to(device)
            mcts_prob = mcts_prob.to(device)

            optimizer.zero_grad()

            policy_pred, value_pred = model(board_tensor)
            
            policy_loss = policy_criterion(policy_pred, mcts_prob)
            value_loss = value_criterion(value_pred.squeeze(), reward)
            loss = policy_loss + value_loss

            loss.backward()

            optimizer.step()
            
            running_loss += loss.item()
        scheduler.step(loss)

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        running_loss = 0.0
        print('average loss %.4f' %(avg_loss))
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_count = 0
            torch.save(model.state_dict(), 'models/%s %d.pt' %(model_name, model_epoch + epoch))
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop:
                print('training early stopped.')
                break

    with open('models/%s %d loss.pickle' %(model_name, model_epoch + epoch), 'wb') as f:
        pickle.dump(losses, f)
    save_graph_img(losses, 'models/%s %d loss.png' %(model_name, model_epoch + epoch))
    print('training successfully ended.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='alphazero_net')
    parser.add_argument('--model-load-path', '-m')
    parser.add_argument('--data-path', '-d')
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--num-epochs', '-n', type=int, default=100)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--early-stop', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(args.board_size).to(device)
    model.load_state_dict(torch.load(args.model_load_path))
    model.name = get_model_name(args.model_load_path)

    dataset = ExperienceCollector()
    dataset.load_experience(args.data_path)

    if args.network == 'alphazero_net':
        train_alphazero_net(model = model,
            dataset = dataset,
            device = device,
            num_epochs = args.num_epochs,
            lr = args.learning_rate,
            batch_size = args.batch_size,
            early_stop = args.early_stop)
    else:
        train_policy_net(model = model,
            dataset = dataset,
            device = device,
            num_epochs = args.num_epochs,
            lr = args.learning_rate,
            batch_size = args.batch_size,
            board_size = args.board_size,
            early_stop = args.early_stop)

if __name__=='__main__':
    main()