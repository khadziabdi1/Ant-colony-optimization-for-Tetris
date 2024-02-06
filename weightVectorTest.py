from tetrisSimulation import TetrisSimulation

weight_vector = [-12, 4, -12, -2, 14, -7, 1, -15, -11, -11, -15]

sim = TetrisSimulation(20,10,weight_vector)
rez = sim.simulate_game()
print(rez)
