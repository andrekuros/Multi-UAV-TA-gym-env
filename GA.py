


def genetic_algorithm(population_size, num_generations, crossover_prob, mutation_prob):
    # 1. Inicialize a população
    population = initialize_population(population_size)

    for generation in range(num_generations):
        # 2. Calcule a aptidão dos cromossomos
        fitness = calculate_fitness(population)

        # 3. Selecione os cromossomos para a próxima geração
        selected_chromosomes = select_chromosomes(population, fitness)

        # 4. Aplique o cruzamento
        new_chromosomes = crossover(selected_chromosomes, crossover_prob)

        # 5. Aplique a mutação
        mutated_chromosomes = mutate(new_chromosomes, mutation_prob)

        # 6. Atualize a população
        population = mutated_chromosomes

    # 7. Retorne o melhor cromossomo da população final
    best_chromosome = get_best_chromosome(population)
    return best_chromosome
