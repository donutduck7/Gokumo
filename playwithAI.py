from game import board as Board, Game
from mcts_zero import MCTSPlayer
from PolicyValueNet import PolicyValueNet as Net

class HumanPlayer(object):
    #Human Player
    def __init__(self):
        self.player = None
    
    def set_player_id(self, p):
        self.player = p
    
    def get_action(self, board):
        try:
            human_action = input("Please enter your move, in x,y format: ")
            if isinstance(human_action, str):
                loc = [int(n, 10) for n in human_action.split(',')]
            human_move = board.location_to_move(loc)
        except Exception as e:
            human_move = -1
            print("NoNoNo")
            print(e)
        if human_move not in board.availables or human_move == -1:
            print("Invalid move, try again! ")
            human_move = self.get_action(board)
        return human_move
    
    def __str__(self):
        return "Human: {}".format(self.player)
    
    
def run():
    # Initialize the game
    # Choose go first or not
    # Default go second
    start_player_decider = input("Human first?(y/n)")
    if start_player_decider == 'y':
        start_player = 0        
    else:
        start_player = 1
    
    # Defaut using AI with 400 playouts, playing with 4 in a row on 6*6 board
    # Can change to using AI with 400 playouts, playing with 5 in a row on 8*8 board
    n_playout = 400
    n_in_row = 4
    # n_in_row = 5
    # width, height = 8, 8
    width, height = 6, 6
    model_file = 'best_policy663_3000.model'
    # model_file = 'best_policy_885.model'

    try:
        board = Board(width=width, height=height, n_in_row=n_in_row)
        gokumo_game = Game(board)

        best_policy_AI = Net(width, height, model_file)
        ai_player = MCTSPlayer(best_policy_AI.policy_value_fn, c_puct=5,n_playout=n_playout)

        human_player = HumanPlayer() 
        gokumo_game.play(human_player, ai_player, start_player, load_game_graph=1)
    except KeyboardInterrupt:
        print('\n\rKeyboard Inerrupt occurs, game quit.')

if __name__ == '__main__':
    run()
