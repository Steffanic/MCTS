from collections import namedtuple
from random import choice
import time
from MCTS.MCTS import MCTS, Node

_TTTB = namedtuple('TTTB', "board_state player_to_move winner terminal")

class TicTacToeBoard(_TTTB, Node):
    def IS_TWO_PLAYER():
        return True
    
    def get_children(board):
        '''Return the new tic tac toe boards that result from taking all available actions in the board_state that are available to play in. The indices that are None in board_state.'''
        if board.is_terminal():
            return set()
        else:
            return {
                board.make_move(i) for i, val in enumerate(board.board_state) if val is None
            }
        
    def make_move(board, index):
        '''
        Makes a new tic tac toe board with the current player marking index
        This is where everything is calculated because this is where the new board object is created.
        '''
        if board.is_terminal():
            raise RuntimeError("You can't play on a terminal board state.")
        current_board_state = board.board_state
        if current_board_state[index] is not None:
            raise RuntimeError("You can't play there, that spot is already taken.")
        current_player = board.player_to_move
        new_board_state = current_board_state[:index]+(current_player,)+current_board_state[index+1:]
        next_player = not current_player
        winner = _find_winner(new_board_state)
        is_terminal = (winner is not None) or (None not in new_board_state) # Check if there is a winner or if the board is full
        return TicTacToeBoard(new_board_state, next_player, winner, is_terminal)
    
    def get_random_child(board):
        possible_actions = [i for i, val in enumerate(board.board_state) if val is None]
        chosen_action = choice(possible_actions)
        return board.make_move(chosen_action)
    
    def is_terminal(board):
        return board.terminal
        
    def get_reward(board):
        '''Only ever called on the agents's turn, aka, board.player_to_move should always be false.'''
        '''It is through backpropagation of the tree that we get a reward of one. See implementation of MCTS'''
        if not board.terminal:
            return 0
        if board.winner is board.player_to_move:
            raise RuntimeError("That's weird, how did you win on your opponent's turn?🤔")
        if board.player_to_move is (not board.winner):
            # Opponent wins
            return 0
        if board.winner is None:
            return 0.5
        raise RuntimeError("Wellp, something weird is happening. Here is the state of the game when you called get_reward"+board.to_pretty_string())
    
    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.board_state[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n Player to move: " + ("X" if board.player_to_move else "O") + "\n"
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None

def new_tic_tac_toe_board():
    return TicTacToeBoard(board_state=(None,) * 9, player_to_move=True, winner=None, terminal=False)

def play_game(number_of_games=100, agent_A_rollouts=5, agent_B_rollouts=5, human_input=False, difficulty_level=None, slow_time=False, print_games=False, print_intermediate_info=False):
    A_wins = 0
    B_wins = 0
    ties = 0
    treeA = MCTS()
    treeB = MCTS()

    if difficulty_level is not None:
        agent_B_rollouts = difficulty_level
    for i in range(number_of_games):
        board = new_tic_tac_toe_board()
        if print_games:print(board.to_pretty_string())
        while True:
            if human_input:
                row_col = input("enter row,col: ")
                row, col = map(int, row_col.split(","))
                index = 3 * (row - 1) + (col - 1)
                board = board.make_move(index)
            else:
                for _ in range(agent_A_rollouts):
                    treeA.do_rollout(board)
                board = treeA.choose(board)
            if print_games: print(board.to_pretty_string())
            if board.terminal:
                if board.winner is None:
                    ties+=1
                else: 
                    A_wins+=1
                break
            if slow_time: time.sleep(0.5)
            # You can train as you go, or only at the beginning.
            # Here, we train as we go, doing fifty rollouts each turn.
            for _ in range(agent_B_rollouts):
                treeB.do_rollout(board)
            board = treeB.choose(board)
            if print_games: print(board.to_pretty_string())
            if board.terminal:
                if board.winner is None:
                    ties+=1
                else:
                    B_wins+=1
                break
            if slow_time: time.sleep(0.5)
        if print_intermediate_info: print(f"Game {i+1} over. A_wins: {A_wins}, B_wins: {B_wins}, ties: {ties}")

    print(f"Player A won {A_wins} times, Player B won {B_wins} times, and there were {ties} ties.")
    return A_wins, B_wins, ties


if __name__=="__main__":
    board = TicTacToeBoard((None,)*9, True, None, False)
    print(board.to_pretty_string())
    #play_game(number_of_games=1000, agent_A_rollouts=2, agent_B_rollouts=5, print_intermediate_info=True)
    play_game(number_of_games=5, print_games=True, human_input=True, difficulty_level=50, print_intermediate_info=True)