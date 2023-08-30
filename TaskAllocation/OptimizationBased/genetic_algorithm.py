import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

def create_initial_population(population_size, num_genes):
    return np.array([np.random.permutation(num_genes) for _ in range(population_size)])


def fitness_function(env, chromosome):
    
    # Redefinir o ambiente   
    env.reset(seed = 0)    

    # Converter o cromossomo em uma alocação de tarefas para os drones
    #task_allocation = chromosome_to_task_allocation(chromosome, env.NUM_DRONES, env.NUM_TARGETS)
    action = chromosome_to_action(chromosome, env.NUM_DRONES, env.NUM_TARGETS)
    
   
    # Atribuir as tarefas aos drones usando a alocação gerada pelo cromossomo
    #env.drone_tasks = task_allocation            
    env.step(action)
    
    done = False    
    
    while not done:
       
        # Executar a simulação e obter a observação, recompensa, done e info
        observation, reward, done, info = env.step(None)
            
    # Extrair as métricas do dicionário 'info'
    total_time = info["total_time"]
    total_distance = info["total_distance"]
    load_balancing_std = info["load_balancing_std"]

    
    # Normalizar as métricas (maior é melhor)
    normalized_time = 382 / total_time
    normalized_distance = 7320 / total_distance
    normalized_load_balancing = 314 / load_balancing_std

    # Combinação ponderada das métricas
    #fitness = 0.4 * normalized_time + 0.4 * normalized_distance + 0.2 * normalized_load_balancing
    fitness = normalized_distance#0.4 * normalized_time + 0.4 * normalized_distance + 0.2 * normalized_load_balancing

    return [fitness, action]


def selection(population, fitness_values):
    # Roleta-russa
    total_fitness = np.sum(fitness_values) 
    probabilities = fitness_values / total_fitness
    idx = np.random.choice(np.arange(len(population)), size=2, p=probabilities)
    return population[idx]

def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int)

    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i

    cx_point1, cx_point2 = np.sort(np.random.choice(range(size), 2, replace=False))

    for i in range(cx_point1, cx_point2):
        temp1 = parent1[i]
        temp2 = parent2[i]
        parent1[i], parent1[p1[temp2]] = temp2, temp1
        parent2[i], parent2[p2[temp1]] = temp1, temp2
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return parent1, parent2


def crossover(parents, crossover_prob):
    if random.random() < crossover_prob:
        offspring = partially_mapped_crossover(parents[0], parents[1])
        return np.vstack(offspring)
    else:
        return parents


def mutation(offspring, mutation_prob):
    for i in range(len(offspring)):
        if random.random() < mutation_prob:
            swap_points = np.random.choice(len(offspring[i]), 2, replace=False)
            offspring[i][swap_points[0]], offspring[i][swap_points[1]] = (
                offspring[i][swap_points[1]],
                offspring[i][swap_points[0]],
            )
    return offspring


def chromosome_to_action(chromosome, num_drones, num_targets):
    task_allocation = {drone_id: [] for drone_id in range(num_drones)}

    # Calcular a semente baseada no cromossomo
    seed = int(sum(chromosome) * 1000)
    random.seed(seed)

    # Alocar todas as tarefas a um drone de forma aleatória
    for task_id in range(num_targets):
        drone_id = random.randint(0, num_drones - 1)
        task_allocation[drone_id].append(task_id)

    # Aplicar os ajustes do cromossomo para redistribuir as tarefas
    for drone_id, adjustment in enumerate(chromosome):
        # Remover tarefas dos drones com ajustes negativos
        if adjustment < 0:
            task_ids_to_remove = task_allocation[drone_id][:int(abs(adjustment))]
            task_allocation[drone_id] = task_allocation[drone_id][int(abs(adjustment)):]

            # Redistribuir as tarefas removidas para outros drones
            for task_id in task_ids_to_remove:
                new_drone_id = (drone_id + 1) % num_drones
                task_allocation[new_drone_id].append(task_id)

        # Adicionar tarefas aos drones com ajustes positivos
        elif adjustment > 0:
            drone_id_to_take_from = (drone_id - 1) % num_drones
            task_ids_to_add = task_allocation[drone_id_to_take_from][-int(adjustment):]
            task_allocation[drone_id_to_take_from] = task_allocation[drone_id_to_take_from][:-int(adjustment)]
            task_allocation[drone_id].extend(task_ids_to_add)

    # Converter a alocação de tarefas em um dicionário de ações
    action = {}
    for drone_id in range(num_drones):
        tasks = task_allocation.get(drone_id, [])
        action[drone_id] = tasks if tasks else []

    return action




def chromosome_to_task_allocation(chromosome, num_drones, num_targets):
    min_tasks_per_drone, extra_tasks = divmod(num_targets, num_drones)

    task_allocation = {}
    task_idx = 0
    for drone_id in range(num_drones):
        tasks_count = min_tasks_per_drone + int(chromosome[drone_id]) + (1 if drone_id < extra_tasks else 0)
        task_allocation[drone_id] = list(range(task_idx, task_idx + tasks_count))
        task_idx += tasks_count

    # Garantir que os IDs das tarefas estejam no intervalo válido
    for drone_id in task_allocation:
        task_allocation[drone_id] = [task_id % num_targets for task_id in task_allocation[drone_id]]

    return task_allocation


def cluster_targets(targets, num_clusters):
    # Realizar a clusterização utilizando o k-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init = 10)
    kmeans.fit([t.position for t in targets])
    
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Ordenar os alvos de acordo com os clusters
    sorted_targets = []
    for cluster_idx in range(num_clusters):
        for target_idx, label in enumerate(cluster_labels):
            if label == cluster_idx:
                sorted_targets.append(targets[target_idx])

    return np.array(sorted_targets)

def genetic_algorithm(env, population_size, num_generations, crossover_prob, mutation_prob):
    
    num_genes = env.NUM_DRONES  # Número de genes igual ao número de drones
    population = create_initial_population(population_size, num_genes)
   
       
    num_clusters = 4  # Defina o número de clusters desejado
    env.targets = cluster_targets(env.targets, num_clusters)    
    
    #print (population)
    fitness_history = []
    for generation in range(num_generations):
        
        fitness_calc = np.array([fitness_function(env, chromosome) for chromosome in population])
        
        fitness_values = [v[0] for v in fitness_calc]
        cromossomes =  [c[1] for c in fitness_calc]       
        
        min_fitness = np.min(fitness_values)
        max_fitness = np.max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        fitness_history.append((min_fitness, mean_fitness, max_fitness))
        
        best_cromossome = cromossomes[fitness_values.index(max_fitness)]
                
        print("Generation: ", generation,  "\n", best_cromossome, "\n", fitness_history[-1])

        new_population = []
        for i in range(population_size // 2):
            parents = selection(population, fitness_values)
            offspring = crossover(parents, crossover_prob)
            offspring = mutation(offspring, mutation_prob)
            new_population.extend(offspring)

        population = np.array(new_population)

    best_chromosome = population[np.argmax(fitness_values)]

    # Plot the fitness history
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(["Min Fitness", "Mean Fitness", "Max Fitness"])
    plt.show()

    return best_chromosome

###_______________________------------------___________________
from DroneEnv import DroneEnv
#--------------- Evaluate GA--------------------------------

#recorde 2.67
# Configurar os hiperparâmetros do algoritmo genético
population_size = 30
num_generations = 20
crossover_prob = 0.7
mutation_prob = 0.2


# Criar o ambiente
env = DroneEnv(render_enabled=False,  action_mode= "TaskAssign")
env.reset( seed = 0)

# Executar o algoritmo genético e obter o melhor cromossomo
best_chromosome = genetic_algorithm(
    env,
    population_size,
    num_generations,
    crossover_prob,
    mutation_prob
)

