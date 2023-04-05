import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read data from CSV file
data = pd.read_csv('/home/akshay/Documents/semester4/D.A.A/cia2/Bank_Personal_Loan_Modelling.csv')

# Split the data into features (X) and target (y)
X = data.drop('Personal Loan', axis=1).values
y = data['Personal Loan'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function
def objective_function(positions):
    # Create a binary mask for the features
    mask = positions.astype(bool)
    
    # Apply the mask to select the features
    X_train_masked = X_train[:, mask]
    X_test_masked = X_test[:, mask]
    
    # Train a logistic regression model on the masked data
    model = LogisticRegression()
    model.fit(X_train_masked, y_train)
    
    # Evaluate the model on the masked test data
    y_pred = model.predict(X_test_masked)
    fitness = accuracy_score(y_test, y_pred)
    
    return fitness

class AntColonyOptimization:
    def _init_(self, num_ants, num_dimensions, max_iterations, alpha, beta, rho, q, evaporation_rate):
        self.num_ants = num_ants
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.evaporation_rate = evaporation_rate
        
    def run(self):
        # Initialize pheromone matrix
        pheromone = np.ones((self.num_dimensions,))
        
        # Initialize best solution and fitness
        best_solution = None
        best_fitness = -np.inf
        
        # Iterate for max_iterations
        for iteration in range(self.max_iterations):
            # Initialize ant solutions and fitnesses
            ant_solutions = np.zeros((self.num_ants, self.num_dimensions))
            ant_fitnesses = np.zeros((self.num_ants,))
            
            # Generate ant solutions
            for ant in range(self.num_ants):
                # Initialize the ant's position
                ant_position = np.zeros((self.num_dimensions,))
                
                # Traverse the solution space
                for dim in range(self.num_dimensions):
                    # Compute the probabilities of moving to each dimension
                    unvisited_indices = np.where(ant_position == 0)[0]
                    probabilities = np.zeros((len(unvisited_indices),)))
                    for i, index in enumerate(unvisited_indices):
                        numerator = pheromone[index] ** self.alpha * objective_function(np.concatenate((ant_position[:dim], [1], ant_position[dim+1:])))
                        denominator = np.sum(pheromone[unvisited_indices] ** self.alpha * objective_function(np.concatenate((ant_position[:dim], [1], ant_position[dim+1:]))))
                        probabilities[i] = numerator / denominator
                    
                    # Select the next dimension to visit based on the probabilities
                    next_dim = np.random.choice(unvisited_indices, p=probabilities)
                    
                    # Update the ant's position
                    ant_position[next_dim] = 1
                
                # Save the ant's solution and fitness
                ant_solutions[ant] = ant_position
                ant_fitnesses[ant] = objective_function(ant_position)
                
            # Update the best solution and fitness
            if np.max(ant_fitnesses) > best_fitness:
                # Update the best solution and fitness
                for i in range(self.num_ants):
                    if fitness[i] > ant_best_fitness[i]:
                        ant_best_fitness[i] = fitness[i]
                        ant_best_solution[i] = positions[i].copy()

                    if fitness[i] > best_fitness:
                        best_fitness = fitness[i]
                        best_solution = positions[i].copy()

            # Update pheromone trails
            delta_pheromone = np.zeros((self.num_cities, self.num_cities))
            for i in range(self.num_ants):
                for j in range(self.num_cities - 1):
                    k, l = path[i][j], path[i][j + 1]
                    delta_pheromone[k][l] += self.Q / distance[i]
                    delta_pheromone[l][k] += self.Q / distance[i]

            self.pheromone = (1 - self.rho) * self.pheromone + delta_pheromone

            # Update iteration counter
            iteration += 1

        return best_solution, best_fitness