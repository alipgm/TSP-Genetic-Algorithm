import random
import numpy as np

# --- 1. Problem Definition & Setup ---

# The distance matrix for our 5 cities (A, B, C, D, E)
# Same data as in our manual example.
distance_matrix = np.array([
    [0, 2.8, 6.0, 8.5, 8.1],  # Distances from A (0)
    [2.8, 0, 3.2, 5.8, 5.8],  # Distances from B (1)
    [6.0, 3.2, 0, 2.8, 3.6],  # Distances from C (2)
    [8.5, 5.8, 2.8, 0, 5.0],  # Distances from D (3)
    [8.1, 5.8, 3.6, 5.0, 0]   # Distances from E (4)
])

NUM_CITIES = 5

# --- 2. Genetic Algorithm Parameters ---

POPULATION_SIZE = 20
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1 # 10% chance to mutate an offspring

# --- 3. Core GA Functions ---

def calculate_total_distance(tour, matrix):
    """Calculates the total distance of a tour."""
    total_distance = 0
    for i in range(NUM_CITIES):
        # Get the distance from the current city to the next one
        # The modulo operator (%) handles the loop back from the last city to the first
        from_city = tour[i]
        to_city = tour[(i + 1) % NUM_CITIES]
        total_distance += matrix[from_city, to_city]
    return total_distance

def create_initial_population(pop_size, num_cities):
    """Creates a starting population of random tours."""
    population = []
    base_tour = list(range(num_cities))
    for _ in range(pop_size):
        tour = random.sample(base_tour, len(base_tour)) # Creates a random permutation
        population.append(tour)
    return population

def roulette_wheel_selection(population, fitness_scores):
    """Selects a single parent using the roulette wheel method."""
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]
    # The `choices` function performs the weighted random selection
    return random.choices(population, weights=selection_probs, k=1)[0]

def ordered_crossover(parent1, parent2):
    """Performs ordered crossover (OX1) to create a valid offspring."""
    size = len(parent1)
    child = [None] * size

    # 1. Select a random subsequence from parent1
    start, end = sorted(random.sample(range(size), 2))
    
    # 2. Copy the subsequence to the child
    child[start:end+1] = parent1[start:end+1]

    # 3. Fill the remaining slots using genes from parent2
    parent2_genes = [item for item in parent2 if item not in child]
    
    pointer = 0
    for i in range(size):
        if child[i] is None:
            child[i] = parent2_genes[pointer]
            pointer += 1
            
    return child

def swap_mutation(tour):
    """Swaps two random cities in a tour to perform mutation."""
    # Select two distinct random indices
    idx1, idx2 = random.sample(range(len(tour)), 2)
    
    # Swap the cities at these indices
    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour

# --- 4. Main Genetic Algorithm Loop ---

print("Starting Genetic Algorithm to solve TSP...")

# Create the initial population
population = create_initial_population(POPULATION_SIZE, NUM_CITIES)
best_tour_overall = None
best_distance_overall = float('inf')

# Evolve the population over many generations
for generation in range(NUM_GENERATIONS):
    
    # Calculate fitness for the current population
    distances = [calculate_total_distance(tour, distance_matrix) for tour in population]
    fitness_scores = [1 / (d + 1e-6) for d in distances] # Add a small epsilon to avoid division by zero

    # Keep track of the best tour found so far
    current_best_distance = min(distances)
    if current_best_distance < best_distance_overall:
        best_distance_overall = current_best_distance
        best_tour_overall = population[distances.index(current_best_distance)]
    
    print(f"Generation {generation + 1}: Best Distance = {best_distance_overall:.2f}")

    # Create the next generation
    new_population = []
    
    # Elitism: Keep the best individual from the current generation
    best_current_tour = population[np.argmax(fitness_scores)]
    new_population.append(best_current_tour)

    # Generate the rest of the new population through selection and reproduction
    while len(new_population) < POPULATION_SIZE:
        # Select two parents
        parent1 = roulette_wheel_selection(population, fitness_scores)
        parent2 = roulette_wheel_selection(population, fitness_scores)
        
        # Create an offspring
        offspring = ordered_crossover(parent1, parent2)
        
        # Apply mutation
        if random.random() < MUTATION_RATE:
            offspring = swap_mutation(offspring)
            
        new_population.append(offspring)
        
    # Replace the old population with the new one
    population = new_population

# --- 5. Final Result ---

print("\n-----------------------------------------")
print("Evolution complete.")
# Remember to add the start city to the end to show the full loop
final_tour_path = " -> ".join(map(str, best_tour_overall)) + f" -> {best_tour_overall[0]}"
print(f"Best tour found: {final_tour_path}")
print(f"Total distance: {best_distance_overall:.2f}")