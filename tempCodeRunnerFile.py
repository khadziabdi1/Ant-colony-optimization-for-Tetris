    def max_well_depth(self, board):
        well_depths = []

        for col in range(board.shape[1]):
            left_side = board[:, col-1] if col-1 >= 0 else np.ones_like(board[:, :1])
            right_side = board[:, col+1] if col+1 <= board.shape[1] - 1 else np.ones_like(board[:, -1:])

            left_occupied_row = np.min(np.where(left_side == 1)[0]) if np.any(left_side == 1) else board.shape[0]
            right_occupied_row = np.min(np.where(right_side == 1)[0]) if np.any(right_side == 1) else board.shape[0]

            occurrences = np.where(board[:, col] == 1)[0]
            well_depth = min(occurrences) - max(left_occupied_row, right_occupied_row) if occurrences.size > 0 else 0

            if(well_depth > 0):
                well_depths.append(well_depth)
            else:
                well_depths.append(0)

        return max(well_depths, default=0)

    def sum_of_wells(self, board):
        well_depths = []

        for col in range(board.shape[1]):
            left_side = board[:, col-1] if col-1 >= 0 else np.ones_like(board[:, :1])
            right_side = board[:, col+1] if col+1 <= board.shape[1] - 1 else np.ones_like(board[:, -1:])

            left_occupied_row = np.min(np.where(left_side == 1)[0]) if np.any(left_side == 1) else board.shape[0]
            right_occupied_row = np.min(np.where(right_side == 1)[0]) if np.any(right_side == 1) else board.shape[0]

            occurrences = np.where(board[:, col] == 1)[0]
            well_depth = min(occurrences) - max(left_occupied_row, right_occupied_row) if occurrences.size > 0 else 0

            if(well_depth > 0):
                well_depths.append(well_depth)
            else:
                well_depths.append(0)
        return sum(well_depths)