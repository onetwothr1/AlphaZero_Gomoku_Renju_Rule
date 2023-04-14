import argparse
from IPython.core.interactiveshell import re
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net.alphazero_net import AlphaZeroNet
from experience import ExperienceCollector
from utils import get_model_name, save_graph_img


def train(model, dataset, save_dir, device, num_epochs, lr, lr_decay, batch_size, early_stop):
    policy_criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
    value_criterion = nn.MSELoss()
    l2_const = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr
    # , weight_decay=l2_const
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('num data:', len(dataset))
    print('num mini-batch:', len(dataloader))

    losses = []
    policy_losses = []
    value_losses = []
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
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

            log_policy_pred, value_pred = model(board_tensor)

            policy_loss = policy_criterion(log_policy_pred, mcts_prob)
            value_loss = value_criterion(value_pred.squeeze(), reward)
            weight1 = 1 / (1 + torch.exp(-policy_loss*2))
            weight2 = 1 / (1 + torch.exp(-value_loss))
            total_weight = weight1 + weight2
            weight1 = weight1 / total_weight
            weight2 = weight2 / total_weight
            # loss = policy_loss + value_loss
            loss = weight1 * policy_loss + weight2 * value_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()
        scheduler.step()

        # record loss
        avg_loss = running_loss / len(dataloader)
        avg_policy_loss = running_policy_loss / len(dataloader)
        avg_value_loss = running_value_loss / len(dataloader)
        losses.append(avg_loss)
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        running_loss = 0.0
        running_policy_loss = 0.0
        running_value_loss = 0.0
        print('average loss %.4f (policy %.4f,  value %.4f)' %(avg_loss, avg_policy_loss, avg_value_loss))

        if avg_loss + 3e-4 < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            early_stop_count = 0
            torch.save(model.state_dict(), save_dir + '/%s %d.pt' %(model.name, epoch))
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop:
                print('Training early stopped.')
                break

    save_graph_img(losses, policy_losses, value_losses, save_dir + '/%s %d loss.png' %(model.name, epoch))
    print('Training successfully ended.')
    print('Epoch %d is the best' %best_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-load-path', '-m')
    parser.add_argument('--data-path', '-d')
    parser.add_argument('--model-save-dir', '-save', type=str, default='models')
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--num-epochs', '-n', type=int, default=100)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=float, default=0.95)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--early-stop', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(args.board_size).to(device)
    model.load_model(args.model_load_path)
    model.name = get_model_name(args.model_load_path)

    dataset = ExperienceCollector()
    dataset.load_experience(args.data_path)

    train(model = model,
        dataset = dataset,
        save_dir = args.model_save_dir,
        device = device,
        num_epochs = args.num_epochs,
        lr = args.learning_rate,
        lr_decay = args.lr_decay,
        batch_size = args.batch_size,
        early_stop = args.early_stop)

if __name__=='__main__':
    main()