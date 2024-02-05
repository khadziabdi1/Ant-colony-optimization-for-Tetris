import numpy as np
from random import randint
from copy import deepcopy

class Tetromino:
    def __init__(self):
        self.type = randint(1, 7)
        self.rotation = 0
        self.grid = self.get_grid()
        self.width, self.height = len(self.grid[0]), len(self.grid)

    def rotate_matrix_to_right(self, matrix):
        transposed_matrix = [list(row) for row in zip(*matrix)]
        rotated_matrix = [list(reversed(row)) for row in transposed_matrix]
        return rotated_matrix

    def set_rotation(self, rotation):
        if not rotation in [0,1,2,3]:
            raise AttributeError("Rotacija nije validna!")
        for i in range((4+rotation-self.rotation)%4):
            self.grid = self.rotate_matrix_to_right(self.grid)
        self.rotation = rotation
        self.width, self.height = len(self.grid[0]), len(self.grid)

    def get_grid(self):
        grid = []
        if self.rotation == 0:
            if self.type == 1:  # L
                grid = [[1, 0], [1, 0], [1, 1]]
            elif self.type == 2:  # RL
                grid = [[0, 1], [0, 1], [1, 1]]
            elif self.type == 3:  # Z
                grid = [[1, 1, 0], [0, 1, 1]]
            elif self.type == 4:  # RZ
                grid = [[0, 1, 1], [1, 1, 0]]
            elif self.type == 5:  # T
                grid = [[1, 1, 1], [0, 1, 0]]
            elif self.type == 6:  # I
                grid = [[1, 1, 1, 1]]
            elif self.type == 7:  # O
                grid = [[1, 1], [1, 1]]
        return grid
    
    def __str__(self):
        matrix_str = ""
        for row in self.grid:
            row_str = "\t".join(map(str, row))
            matrix_str += row_str + "\n"
        return matrix_str

class TetrisSimulation:
    def __init__(self, height, width, weight_vector):
        self.height = height
        self.width = width
        self.board = np.zeros((height, width), dtype=int)
        self.weight_vector = weight_vector
        self.removed_lines = 0
        self.number_of_moves = 0

    def place_tetromino(self, tetromino, col):
        new_board = deepcopy(self.board)
        for i in range(len(self.board)-tetromino.height+1):
            t_grid = deepcopy(tetromino.grid)
            for j in range(tetromino.height):
                for k in range(tetromino.width):
                    t_grid[j][k] += new_board[i+j][col+k]
            go_back = False
            for row in t_grid:
                if 2 in row:
                    go_back = True
            if go_back:
                if i == 0:
                    return
                for j in range(tetromino.height):
                    for k in range(tetromino.width):
                        new_board[i-1+j][col+k] += tetromino.grid[j][k]
                return new_board
            else:
                if i == len(self.board)-tetromino.height:
                    for j in range(tetromino.height):
                        for k in range(tetromino.width):
                            new_board[i+j][col+k] += tetromino.grid[j][k]
                    return new_board
                continue

    def get_all_possible_states(self, tetromino):
        all_states = []
        for rotation in range(4):  
            tetromino.set_rotation(rotation)
            for col in range(self.width-tetromino.width+1):
                all_states.append(self.place_tetromino(tetromino,col))
        return [value for value in all_states if value is not None]

    def pile_height(self, board):
        return np.max(np.argmax(board, axis=0), initial=-1) + 1

    def holes(self, board):
        return np.sum(board[:-1, :] == 0)

    def connected_holes(self, board):
        holes_count = 0
        visited = np.zeros_like(board, dtype=bool)

        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 1):
                if board[row, col] == 0 and not visited[row, col] and board[row + 1, col] == 1:
                    holes_count += 1
                    visited[row:, col] = True

        return holes_count

    def altitude_difference(self, board):
        occupied_rows, occupied_cols = np.where(board == 1)
        max_occupied_row = np.max(occupied_rows)
        min_free_row = np.min(np.where(board[max_occupied_row:, :] == 0)[0] + max_occupied_row)
        return max_occupied_row - min_free_row

    def max_well_depth(self, board):
        well_depths = []
        for col in range(board.shape[1] - 1):
            if np.any(board[:, col] == 1) and np.any(board[:, col + 1] == 1):
                well_depths.append(max(0, np.min(np.where(board[:, col] == 1)[0]) - np.min(np.where(board[:, col + 1] == 1)[0])))
            else:
                well_depths.append(0)
        return max(well_depths, default=0)

    def sum_of_wells(self, board):
        well_depths = []
        for col in range(board.shape[1] - 1):
            occurrences_col = np.where(board[:, col] == 1)[0]
            occurrences_col_next = np.where(board[:, col + 1] == 1)[0]
            if occurrences_col.size > 0 and occurrences_col_next.size > 0:
                well_depths.append(max(0, np.min(occurrences_col) - np.min(occurrences_col_next)))
            else:
                well_depths.append(0)
        return sum(well_depths)

    def blocks(self, board):
        return np.sum(board)

    def weighted_blocks(self, board):
        return np.sum([row * np.sum(board[row, :]) for row in range(board.shape[0])])

    def row_transitions(self, board):
        return np.sum(board[:, :-1] != board[:, 1:])

    def column_transitions(self, board):
        return np.sum(board[:-1, :] != board[1:, :])

    def remove_completed_rows(self, board):
        completed_rows = np.all(board, axis=1)
        rows_removed = np.sum(completed_rows)
        if np.any(completed_rows):
            board = board[~completed_rows, :]
            new_rows = np.zeros((np.sum(completed_rows), board.shape[1]), dtype=int)
            board = np.vstack((new_rows, board))
        return board, rows_removed

    def evaluate_state(self, state):
        f = np.zeros(11, dtype=int)
        board = np.array(state)
        board, f[3] = self.remove_completed_rows(board)
        f[0] = self.pile_height(board)
        f[1] = self.holes(board)
        f[2] = self.connected_holes(board)
        f[4] = self.altitude_difference(board)
        f[5] = self.max_well_depth(board)
        f[6] = self.sum_of_wells(board)
        f[7] = self.blocks(board)
        f[8] = self.weighted_blocks(board)
        f[9] = self.row_transitions(board)
        f[10] = self.column_transitions(board)
        return np.sum(self.weight_vector * f), f[3]

    def make_move(self, tetromino):
        states = self.get_all_possible_states(tetromino)
        if len(states) == 0:
            return 1
        best_state = None
        best_state_value = None
        best_state_removed_lines = 0
        for state in states:
            evaluation, removed_lines = self.evaluate_state(state)
            if best_state_value is None:
                best_state_value = evaluation
                best_state = state
                best_state_removed_lines = removed_lines
            elif evaluation > best_state_value:
                best_state_value = evaluation
                best_state = state
                best_state_removed_lines = removed_lines
        self.board = best_state
        self.removed_lines += removed_lines
        self.number_of_moves += 1 
        return 0

    def __str__(self):
        hline = "+"
        for i in range(self.width):
            hline += "-"
        hline += "+\n"
        rez = hline
        for row in self.board:
            rez += "|"
            row_str = "".join(map(str, row))
            for char in row_str:
                if char == "0":
                    rez += ' '
                else:
                    rez += char
            rez += "|\n"
        rez += hline
        return rez

    def simulate_game(self):
        i = 0
        while(1):
            i+=1
            t = Tetromino()
            exit = self.make_move(t)
            if exit:
                break
        return self.number_of_moves, self.removed_lines


#MAIN
weight_vector = np.random.rand(11) # To be optimised using ACO
tetris_sim = TetrisSimulation(20, 10, weight_vector)

m, l = tetris_sim.simulate_game()
print("Moves made:", m)
print("Lines removed:", l)
