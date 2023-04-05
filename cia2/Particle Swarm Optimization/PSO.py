import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('/home/akshay/Documents/semester4/D.A.A/cia2/Bank_Personal_Loan_Modelling.csv')
X = data.drop('Personal Loan', axis=1).values
y = data['Personal Loan'].values

# Define objective function
def objective_function(position, X, y):
    y_pred = sigmoid(np.dot(X, position))
    return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Particle:
    def _init_(self, position, velocity, pbest_position, pbest_fitness):
        self.position = position
        self.velocity = velocity
        self.pbest_position = pbest_position
        self.pbest_fitness = pbest_fitness
        
    def evaluate_fitness(self, objective_function, X, y):
        fitness = objective_function(self.position, X, y)
        if fitness > self.pbest_fitness:
            self.pbest_fitness = fitness
            self.pbest_position = self.position
        return fitness
        
class ParticleSwarmOptimization:
    def _init_(self, num_particles, num_dimensions, max_iterations, c1, c2, w):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.swarm = []
        self.gbest_position = np.random.rand(num_dimensions)
        self.gbest_fitness = -np.inf
        
    def initialize_swarm(self):
        for i in range(self.num_particles):
            position = np.random.rand(self.num_dimensions)
            velocity = np.random.rand(self.num_dimensions)
            pbest_position = position
            pbest_fitness = -np.inf
            particle = Particle(position, velocity, pbest_position, pbest_fitness)
            self.swarm.append(particle)
    
    def update_swarm(self, objective_function, X, y):
        for particle in self.swarm:
            particle.velocity = self.w*particle.velocity \
                                 + self.c1*np.random.rand()*(particle.pbest_position-particle.position) \
                                 + self.c2*np.random.rand()*(self.gbest_position-particle.position)
            particle.position = particle.position + particle.velocity
            particle.evaluate_fitness(objective_function, X, y)
            if particle.pbest_fitness > self.gbest_fitness:
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_position = particle.pbest_position
        
    def run(self, objective_function, X, y):
        self.initialize_swarm()
        for iteration in range(self.max_iterations):
            self.update_swarm(objective_function, X, y)
        return self.gbest_position, self.gbest_fitness