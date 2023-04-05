import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('/home/akshay/Documents/semester4/D.A.A/cia2/Bank_Personal_Loan_Modelling.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop('Personal Loan', axis=1).values
y = df['Personal Loan'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function to evaluate the chromosomes
def fitness_function(chromosome):
    # Create a Logistic Regression model
    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train[:, chromosome], y_train)

    # Predict the target variable on the testing data
    y_pred = model.predict(X_test[:, chromosome])

    # Return the accuracy of the model as the fitness value
    return accuracy_score(y_test, y_pred)

# Define the Genetic Algorithm class
class GeneticAlgorithm:
    def _init_(self, population_size, num_generations, mutation_rate, crossover_rate, num_elites, chromosome_length, fitness_function):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_elites = num_elites
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function

    def run(self):
        # Initialize population
        population = np.random.randint(2, size=(self.population_size, self.chromosome_length))

        # Iterate for num_generations
        for generation in range(self.num_generations):
            # Evaluate fitness of population
            fitness = np.array([self.fitness_function(chromosome) for chromosome in population])

            # Select the top performing individuals as elites
            elites_indices = np.argsort(fitness)[::-1][:self.num_elites]
            elites = population[elites_indices]

            # Perform roulette wheel selection to generate the mating pool
            selection_probabilities = fitness / np.sum(fitness)
            mating_pool_indices = np.random.choice(range(self.population_size), size=self.population_size - self.num_elites, replace=True, p=selection_probabilities)
            mating_pool = population[mating_pool_indices]

            # Perform crossover
            for i in range(0, self.population_size - self.num_elites, 2):
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.chromosome_length)
                    mating_pair_1 = mating_pool[i]
                    mating_pair_2 = mating_pool[i + 1]
                    offspring_1 = np.concatenate((mating_pair_1[:crossover_point], mating_pair_2[crossover_point:]))
                    offspring_2 = np.concatenate((mating_pair_2[:crossover_point], mating_pair_1[crossover_point:]))
                    mating_pool[i] = offspring_1
                    mating_pool[i + 1] = offspring_2

            # Perform mutation
            for i in range(self.population_size - self.num_elites):
                for j in range(self.chromosome_length):
                    if np.random.rand() < self.mutation_rate:
                        mating_pool[i, j] = 1 - mating_pool[i, j]

# Create the next generation population by combining the elites and the offspring from the mating pool
next_gen_population = np.vstack((elites, offspring))

# Evaluate fitness of next generation population
next_gen_fitness = np.array([fitness_function(individual, X_train, y_train) for individual in next_gen_population])

# Select the fittest individuals for the next generation
sorted_indices = np.argsort(next_gen_fitness)[::-1]
next_gen_population = next_gen_population[sorted_indices]
next_gen_fitness = next_gen_fitness[sorted_indices]
next_gen_population = next_gen_population[:population_size]
next_gen_fitness = next_gen_fitness[:population_size]

# Update current population and fitness
current_population = next_gen_population
current_fitness = next_gen_fitness

# Print current best fitness
print('Generation', generation, 'Best Fitness:', current_fitness[0])

# Check for early stopping
if np.isclose(current_fitness[0], 1.0):
    print('Target fitness achieved. Stopping early.')
    