# PolicyValue Net is a simple neural network that combines the policy network and the value network.
# We use PyTorch to implement the PolicyValue Net.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_LR(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class VanillaNet(nn.Module):
    def __init__(self, board_width, board_height):
        super(VanillaNet, self).__init__()
        self.width = board_width
        self.height = board_height
        # Shared layers
        # 3 convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Policy layers, 1 convolutional layer 1 liner layer
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        # Value layers, 1 convolutional layer 2 liner layer
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.width * self.height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # Value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.width * self.height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val
    

class PolicyValueNet():
    def __init__(self, board_width, board_height, modle=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4 # coefficient  of l2 penalty

        if self.use_gpu:
            self.policy_value_net = VanillaNet(board_width, board_height).cuda()
        else:
            self.policy_value_net = VanillaNet(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        # Load the trained module to play
        if modle is not None:
            net_params = torch.load(modle)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, batch):
        # Converts batch of states into a batch of actions probabilities and state values
        # batch is a list of states
        if self.use_gpu:
            batch = Variable(torch.FloatTensor(batch).cuda()) 
            log_act_probs, value = self.policy_value_net(batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
    
    def policy_value_fn(self, board):
        # Converts a board into a list of (action, probability) tuples for each actions avaliable

        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value
    
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        # Performs a training step using a batch of states, MCTS probs, and the winner of each game in the batch
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        
        self.optimizer.zero_grad()
        set_LR(self.optimizer, lr)
        
        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()
    
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params
    
    def save_model(self, model_file):
        torch.save(self.get_policy_param(), model_file)