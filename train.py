import random
import numpy as np
import torch
from collections import deque, defaultdict
from game import board as Board, Game
from mcts import MCTSPurePlayer as MCTS_Pure
from mcts_zero import MCTSPlayer as MCTS_Zero
from PolicyValueNet import PolicyValueNet
import matplotlib.pyplot as plt
import logging
import time

class Train():
    def __init__(self, init_model=None):
        # params of the board and the game
        # self.board_width = 6
        self.board_width = 8
        # self.board_height = 6
        self.board_height = 8
        #self.n_in_row = 4
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.checkPointPer = 50
        # self.checkPointPer = 10 # test case
        # self.game_batch_num = 10 # test case
        # self.game_batch_num = 1500
        # self.game_batch_num = 2000
        self.game_batch_num = 3000
        self.best_win_ratio = 0.0
        
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTS_Zero(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)
    
    def extend_dataset(self, play_data):
        # Enlarge the dataset by rotation and flipping.
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate 90 degree
                new_state = np.array([np.rot90(s, i) for s in state])
                new_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((new_state, np.flipud(new_mcts_prob).flatten(), winner))
                
                # flip horizontally
                new_state = np.array([np.fliplr(s) for s in new_state])
                new_mcts_prob = np.fliplr(new_mcts_prob)   
                extend_data.append((new_state, np.flipud(new_mcts_prob).flatten(), winner))
            
        return extend_data
    
    def collect_selfplay_data(self, n_games=1):
        """
        collect self-play data for training
        """
        for i in range(n_games):
            winner, play_data = self.game.self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)

            
    def policy_update(self):
        """
        update the policy-value net
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):     
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  
                break#Shutting down if D_KL diverges badly
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        
        print('kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{}'.format(kl, self.lr_multiplier, loss, entropy))
        logging.info('kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{}'.format(kl, self.lr_multiplier, loss, entropy))
        return loss, entropy
    
    def policy_evaluate(self, n_games=10):
        # Evaluate the trained policy by playing against the pure MCTS player 10 games
        # If we reachies 10-0 win, our pure_mcts_playout_num += 1000

        zero_mcts_player = MCTS_Zero(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_count = defaultdict(int)
        for i in range(n_games):
            winner = self.game.play(zero_mcts_player, pure_mcts_player, start_player=i % 2, load_game_graph=0)
            print("The winner is: {}".format(winner))
            win_count[winner] += 1
        win_rate = 1.0*(win_count[1] + 0.5*win_count[-1])/n_games
        print("Our model playouts: {},  pure MCTS playouts: {}, num of games: {}, win: {}, lose: {}, tie: {}, win_ratio: {}".format(self.n_playout,
            self.pure_mcts_playout_num, n_games, win_count[1], win_count[2], win_count[-1], win_rate))
        logging.info("Our model playouts: {},  pure MCTS playouts: {}, num of games: {}, win: {}, lose: {}, tie: {}, win_ratio: {}".format(self.n_playout,
            self.pure_mcts_playout_num, n_games, win_count[1], win_count[2], win_count[-1], win_rate))
        return win_rate
    
    def run(self):
        losses = []
        entropies = []
        start_time = time.perf_counter()
        print("Start training...\n")
        print("Training for {} batches...\n".format(self.game_batch_num))
        print("Training game, board size: {} * {}, number in a row: {}".format(self.board_width, self.board_height, self.n_in_row))
        logging.info("Start training...\n")
        logging.info("Training for {} batches...\n".format(self.game_batch_num))
        logging.info("Training game, board size: {} * {}, number in a row: {}".format(self.board_width, self.board_height, self.n_in_row))
        try:
            for batch_num in range(self.game_batch_num):
                tic1 = time.perf_counter()
                self.collect_selfplay_data(self.play_batch_size)
                print("batch_num: {}, episode_len: {}".format(batch_num+1, self.episode_len))
                logging.info("batch_num: {}, episode_len: {}".format(batch_num+1, self.episode_len))
                
                if (len(self.data_buffer) > self.batch_size):
                    loss, entropy = self.policy_update()
                    losses.append(loss)
                    entropies.append(entropy)
                
                toc1 = time.perf_counter()
                print("batch_num: {} used time: {:.4f}s".format(batch_num+1, toc1-tic1))
                logging.info("batch_num: {} used time: {:.4f}s".format(batch_num+1, toc1-tic1))

                if (batch_num+1) % self.checkPointPer == 0:
                    # Evaluate our model every checkPointPer (defalut 50)
                    print("Reaches the evaluation check point, starting evaluation...\n")
                    print("Current batch: {}".format(batch_num+1))
                    logging.info("Reaches the evaluation check point, starting evaluation...\n")
                    logging.info("Current batch: {}".format(batch_num+1))
                    self.policy_value_net.save_model('./current_policy.model')
                    win_rate = self.policy_evaluate()
                    if win_rate > self.best_win_ratio:
                        print("New best policy, saving to best_policy.model...")
                        logging.info("New best policy, saving to best_policy.model...")
                        self.best_win_ratio = win_rate
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
            fig_loss, ax_loss = plt.subplots()
            fig_entropy, ax_entropy = plt.subplots()
            ax_loss.set_title("Loss per batch")
            ax_entropy.set_title("Entropy per batch")
            ax_loss.set_xlabel("Batch")
            ax_loss.set_ylabel("Loss")
            ax_entropy.set_xlabel("Batch")
            ax_entropy.set_ylabel("Entropy")
            ax_loss.plot(losses)
            ax_entropy.plot(entropies)
            fig_loss.savefig("./loss.png")
            fig_entropy.savefig("./entropy.png")
            
            ending_time = time.perf_counter()
            print("Training is done, total training time: {:.4f}h.".format((ending_time-start_time)/60/60))
            

        except KeyboardInterrupt:
            print('\n\rKeyboard Interrupt occurs, training quit')
            logging.info('\n\rKeyboard Interrupt occurs, training quit')
        



if __name__ == '__main__':
    
    logging.basicConfig(filename='output.log', level=logging.INFO)
    training = Train()
    training.run()

                    

                    
