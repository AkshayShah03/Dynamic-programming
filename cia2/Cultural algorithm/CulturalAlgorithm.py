import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class CulturalAlgorithm:
    def _init_(self, num_ants, num_dimensions, max_iterations, pc, pm, objective_function):
        self.num_ants = num_ants
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.pc = pc
        self.pm = pm
        self.objective_function = objective_function

    def run(self, X_train, y_train):
        # Initialize population
        population = np.random.randint(
            2, size=(self.num_ants, self.num_dimensions))

        # Initialize best solution and fitness
        best_solution = None
        best_fitness = -np.inf

        # Iterate for max_iterations
        for iteration in range(self.max_iterations):
            # Evaluate fitness of population
            fitness = np.array([self.objective_function(
                positions, X_train, y_train) for positions in population])

            # Update best solution
            if np.max(fitness) > best_fitness:
                best_solution = population[np.argmax(fitness)]
                best_fitness = np.max(fitness)

            # Sort population by fitness
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Perform crossover and mutation
            for i in range(self.num_ants):
                # Perform crossover
                if np.random.rand() < self.pc:
                    j = np.random.randint(self.num_ants)
                    k = np.random.randint(self.num_dimensions)
                    child = np.concatenate(
                        (population[i, :k], population[j, k:]))
                    population = np.vstack((population, child))

                # Perform mutation
                if np.random.rand() < self.pm:
                    j = np.random.randint(self.num_dimensions)
                    population[i, j] = 1 - population[i, j]

            # Remove worst solutions
            population = population[:self.num_ants, :]

        return best_solution, best_fitness


# Load the dataset
df = pd.read_csv('/home/akshay/Documents/semester4/D.A.A/cia2/Bank_Personal_Loan_Modelling.csv')

# Drop irrelevant columns
df = df.drop(['ID', 'ZIP Code'], axis=1)

# Split features and target
X = df.drop('Personal Loan', axis=1).values
y = df['Personal Loan'].values

# Scale features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function
def objective_function(positions, X, y):
    selected_features = [i for i, p in enumerate(positions) if p == 1]
    if len(selected_features) == 0:
        return -np.inf
    X_selected = X[:, selected_features]
    X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    lr = LogisticRegression()
    lr.fit(X_train_selected, y_train)
    return accuracy_score(y_test, lr.predict(X_test_selected))

# Initialize and run the Cultural Algorithm
num_ants = 20
num_dimensions = X.shape[1]
max_iterations = 50
pc = 0.7
pm = 0.001
ca = CulturalAlgorithm(num_ants=num_ants,
                       num_dimensions=num_dimensions,
                       max_iterations=max_iterations,
                       alpha=alpha,
                       beta=beta,
                       rho=rho,
                       objective_function=objective_function,
                       bounds=bounds,
                       knowledge_sources=[knowledge_source_1, knowledge_source_2],
                       knowledge_percentages=[knowledge_percentage_1, knowledge_percentage_2])

# Run the Cultural Algorithm
best_solution, best_fitness = ca.run()

# Print the best solution and fitness
print('Best solution:', best_solution)
print('Best fitness:', best_fitness)