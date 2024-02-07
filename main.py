from ACO import AntColony
from tetrisSimulation import TetrisSimulation

def eval_function(solution):
    weight_vector = []
    for i in range(0, len(solution), len(solution)//11):
        chunk = solution[i:i + len(solution)//11]
        sign_bit = chunk[0]
        remaining_bits = chunk[1:]
        decimal_value = int(''.join(map(str, remaining_bits)), 2)
        if sign_bit == 1:
            decimal_value = -decimal_value
        weight_vector.append(decimal_value)
    tetris_sim = TetrisSimulation(20, 10, weight_vector)
    return tetris_sim.simulate_game()

vector_size = 66
num_ants = 15
max_iterations = 2
ant_colony = AntColony(vector_size, num_ants,eval_function,max_iterations)
best_solution, best_score, best_lines_removed = ant_colony.run()

print(f"Best weight vector {best_solution} | Best score {best_score} | Lines removed {best_lines_removed}")

