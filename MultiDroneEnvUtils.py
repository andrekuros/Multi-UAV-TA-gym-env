import random



def generate_random_tasks_all(drones, targets, seed = 0):
    
    """
    This function generates a random task assignment for each drone in the environment.
    
    Args:
    drones (list): A list of drones in the environment.
    targets (list): A list of targets in the environment.
    
    Returns:
    task_actions (dict): A dictionary containing task assignments for each drone, with the drone index as the key
                         and the list of assigned tasks as the value.
    """
    rndGen = random.Random(seed)
    task_list = list(range(len(targets)))
    rndGen.shuffle(task_list)

    # Calculate the minimum number of tasks per drone and the number of drones that will receive an extra task
    min_tasks_per_drone, extra_tasks = divmod(len(targets), len(drones))

    # Generate task actions for the drones
    task_actions = {}
    task_idx = 0
    
    for i in range(len(drones)):
        num_tasks = min_tasks_per_drone + (1 if i < extra_tasks else 0)
        task_actions[i] = task_list[task_idx:task_idx + num_tasks]
        task_idx += num_tasks

    return task_actions