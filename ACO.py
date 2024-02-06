import numpy as np
from tetrisSimulation import TetrisSimulation
class AntColony:
    def __init__(self, vector_size, num_ants, fitness_function, max_iterations=50, alpha=1, beta=2, evaporation_rate=0.5):
        self.vector_size = vector_size
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_matrix = np.ones((2, vector_size))
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations

    def update_pheromones(self, delta_pheromones):
        self.pheromone_matrix = (1 - self.evaporation_rate) * self.pheromone_matrix + delta_pheromones

    def generate_solution(self):
        solution = np.zeros(self.vector_size, dtype=int)
        for i in range(self.vector_size):
            probabilities = self.calculate_probabilities(solution, i)
            choice = np.random.choice([0, 1], p=probabilities)
            solution[i] = choice
        return solution

    def calculate_probabilities(self, solution, current_index):
        pheromones = self.pheromone_matrix[:, current_index]
        visibility = 1 / (np.abs(1 - solution[current_index]) + 1)  # Simple visibility function

        probabilities = (pheromones ** self.alpha) * (visibility ** self.beta)
        probabilities /= probabilities.sum()

        return probabilities
    
    def run(self):
        best_solutions = []
        for iteration in range(self.max_iterations):
            solutions = []
            for ant in range(self.num_ants):
                solution = self.generate_solution()
                solutions.append((solution, self.fitness_function(solution)))
            
            best_solutions.append(max(solutions, key=lambda x: x[1]))
            delta_pheromones = np.zeros((2, self.vector_size))

            for ant_solution, ant_score in solutions:
                for i, choice in enumerate(ant_solution):
                    delta_pheromones[choice, i] += ant_score / max(best_solutions, key= lambda x:x[1])[1]

            self.update_pheromones(delta_pheromones)

        return max(best_solutions, key= lambda x:x[1])

def eval_function(solution):
    weight_vector = np.zeros(11,dtype=int)
    for i in range(11):
        for j in range(1,(int(len(solution)/11))):
            weight_vector[i] += solution[j+i*(int(len(solution)/11))] * 2 ** ((int(len(solution)/11))-1-j)
        weight_vector[i] *= -solution[i*(int(len(solution)/11))]
    tetris_sim = TetrisSimulation(20, 10, weight_vector)
    m, l = tetris_sim.simulate_game()
    return m

def main():
    vector_size = 110
    num_ants = 10
    max_iterations = 50
    ant_colony = AntColony(vector_size, num_ants,eval_function,max_iterations)
    best_solution, best_score = ant_colony.run()

    weight_vector = np.zeros(11,dtype=int)
    for i in range(11):
        for j in range(1,(int(len(best_solution)/11))):
            weight_vector[i] += best_solution[j+i*(int(len(best_solution)/11))] * 2 ** ((int(len(best_solution)/11))-1-j)
        weight_vector[i] *= -best_solution[i*(int(len(best_solution)/11))]

    print(f"Best Solution {weight_vector} | Best Score {best_score}")

if __name__ == "__main__":
    main()
