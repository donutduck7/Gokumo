#Gokumo Game
import numpy as np

class board(object):
    #Board for our current game

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {}

        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1,2]
        #player 1 and player 2
    
    def init_board(self, start_palyer = 0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception("Board width and height can not be less than {}". format(self.n_in_row))
        self.current_player = self.players[start_palyer]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        #Input the number, convert it into (h, w)
        h = move // self.width
        w = move % self.width
        return [h,w]

    def location_to_move(self, location):
        if len(location)!=2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move
    
    def current_state(self):
        # Get the current state of the board
        # State shape: (4, width, height)
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            #print(move_curr, move_oppo)
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            
            # indicate the last move location
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]
    
    
    def do_move(self, move):
        #Do the move, update the board
        self.states[move] = self.current_player
        self.availables.remove(move)
        # previous_player = self.current_player
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move       
        # If previous player is player 1, then current player is 0, and vice versa

    def has_a_winner(self):
        #Check if there is a winner
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moves = list(set(range(width * height)) - set(self.availables)) #all moves
        if len(moves) < self.n_in_row*2 - 1:
            return False, -1
        
        for move in moves:
            h = move // width
            w = move % width
            player = states[move]

            # check horizontally
            if (w in range(width - n + 1) and len(set(states.get(i, -1) for i in range(move, move + n))) == 1):
                return True, player
            # check vertically
            if (h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(move, move + n*width, width))) == 1):
                return True, player
            
            # check / diagonal
            if (w in range(width - n + 1) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(move, move + n*(width + 1), width + 1))) == 1):
                return True, player
            
            # check \ diagonal
            if (w in range(n - 1, width) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(move, move + n*(width - 1), width - 1))) == 1):
                return True, player
            
            
        return False, -1
    
    def get_current_player(self):
        return self.current_player
    
    def game_end(self):
        # Chech whether the game ends
        # If there is a winner or there is no availables move, Game end
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1
    
class Game(object):
    # Gokumo game class
    def __init__(self, board, **kwargs):
        self.board = board
        
    def graphic(self, board, player1, player2):
        # Draw the board and game info
        w = board.width
        h = board.height

        print("Player", player1, " with X".rjust(3))
        print("Player", player2, " with O".rjust(3))
        print()

        for x in range(w):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(h - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(w):
                location = i * w + j
                player = board.states.get(location, -1)
                if player == player1:
                    print("X".center(8), end='')
                elif player == player2:
                    print("O".center(8), end='')
                else:
                    print("+".center(8), end='')
            print('\r\n\r\n')

    def play(self, player1, player2, start_player=0, load_game_graph = 1):
        # Play the game with loaded module
        # self.board.init_board()
        if start_player not in (0,1):
            raise Exception("start_player should be either 0 (player1 first) or 1 (player2 first)")
            
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_id(p1)
        player2.set_player_id(p2)
        players = {p1: player1, p2: player2}
        if load_game_graph:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            #print("Player ", current_player, " takes the action ", move)
            if load_game_graph:
                self.graphic(self.board, player1.player, player2.player)
                    
            end, winner = self.board.game_end()
            if end:
                if load_game_graph:
                    if winner != -1:
                        print("Player", players[winner], "win the game!")
                    else:
                        print("Nobody win the game!")
                return winner

    def self_play(self, player, load_game_graph=0, temp = 1e-3):
        # Start the selfplay game using MCTS palyer
        self.board.init_board()
        p1, p2 = self.board.players
        states, probs, current_players = [], [], []
        while True:
                
            move, move_probs = player.get_action(self.board, temp=temp, return_prob = 1)
            # store the data
            states.append(self.board.current_state())
            probs.append(move_probs)
            current_players.append(self.board.get_current_player())
                
            # perform a move
            self.board.do_move(move)
            #print("Player ", self.board.current_player(), " takes the action ", move)

            if load_game_graph:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                winners_zeros = np.zeros(len(current_players))
                if winner != -1:
                    winners_zeros[np.array(current_players) == winner] = 1.0
                    winners_zeros[np.array(current_players) != winner] = -1.0
                # Reset MCTS root node
                player.reset_player()
                if load_game_graph:
                    if winner != -1:
                        print("Player", p1 if winner == p1 else p2, "win the game!")
                    else:
                        print("Nobody win the game!, It's a draw!")
                return winner, zip(states, probs, winners_zeros)
                   
            
            