import numpy as np
#np.random.seed(42)
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
                fitness = self.fitness_function(solution)
                solutions.append((solution, fitness))
            
            best_ant_solution, best_ant_score = max(solutions, key=lambda x: x[1])
            best_solutions.append((best_ant_solution, best_ant_score))
            delta_pheromones = np.zeros((2, self.vector_size))
            global_best_solution = max(best_solutions, key= lambda x:x[1])[1]
            for ant_solution, ant_score in solutions:
                for i, choice in enumerate(ant_solution):
                    delta_pheromones[choice, i] += ant_score / global_best_solution

            self.update_pheromones(delta_pheromones)
            if best_ant_score > 1000:
                break
            print(f"Iteration {iteration + 1}: Best Score {best_ant_score}")
        
        weight_vector = []
        best_solution = max(best_solutions, key= lambda x:x[1])
        for i in range(0, len(best_solution[0]), len(best_solution[0])//11):
            chunk = best_solution[0][i:i + len(best_solution[0])//11]
            sign_bit = chunk[0]
            remaining_bits = chunk[1:]
            decimal_value = int(''.join(map(str, remaining_bits)), 2)
            if sign_bit == 1:
                decimal_value = -decimal_value
            weight_vector.append(decimal_value)
        return weight_vector, best_solution[1]

