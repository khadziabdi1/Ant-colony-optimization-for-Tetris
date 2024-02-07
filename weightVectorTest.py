from tetrisSimulation import TetrisSimulation
import random
weight_vector = [-17636, 18574, -23919, -16797, 21146, -11704, 30208, -30106, -21136, -31399, -29620]

#for i in range(11):
#    weight_vector.append(random.randint(-1000,1000))

sim = TetrisSimulation(20,10,weight_vector)
rez = sim.simulate_game()
print(rez)
