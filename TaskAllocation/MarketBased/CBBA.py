import numpy as np
import random

class CBBA():
    def __init__(self, drones, tasks, max_dist):          
        self.max_dist = max_dist        
        
       
    def allocate_tasks(self, agents, tasks, Qs=None):
        actions = {}
        self.task_dict = {task.task_id: task for task in tasks}  # Map unique task IDs to tasks
        task_dict = {task.task_id: task for task in tasks}  # Map unique task IDs to tasks
        remaining_tasks = set(task_dict.keys())
        bundles = {agent.drone_id: [] for agent in agents}  # Map unique agent IDs to bundles
        paths = {agent.drone_id: [] for agent in agents}  # Map unique agent IDs to paths
        self.current_makespan = 0
        self.bids = {task.task_id: {'agent_id': None, 'bid': -np.inf} for task in tasks}

        while remaining_tasks:
            for task_id in remaining_tasks:
                random.shuffle(agents)  # Randomize the order of agents
                for agent in agents:
                    task = task_dict[task_id]
                    if task_id not in bundles[agent.drone_id]:
                        bid = self.calculate_bid(agent, task, paths[agent.drone_id], Qs=Qs)
                        if bid > self.bids[task_id]['bid']:
                            self.bids[task_id] = {'agent_id': agent.drone_id, 'bid': bid}
                            # Determine where to insert the task in the path
                            insertion_point = self.determine_insertion_point(agent, task, paths[agent.drone_id])
                            paths[agent.drone_id].insert(insertion_point, task_id)

            # Consensus phase
            for task_id, bid_info in self.bids.items():
                for agent in agents:
                    if task_id in bundles[agent.drone_id] and agent.drone_id != bid_info['agent_id']:
                        bundles[agent.drone_id].remove(task_id)
                        paths[agent.drone_id].remove(task_id)
                if bid_info['agent_id'] is not None and task_id not in bundles[bid_info['agent_id']]:
                    bundles[bid_info['agent_id']].append(task_id)
                    remaining_tasks.remove(task_id)
            # Update makespan
            self.current_makespan = max(self.calculate_total_time(agent, paths[agent.drone_id]) for agent in agents)           


        for agent_id, bundle in bundles.items():
            for task_id in bundle:
                agent_key = "agent" + str(agent_id)
                if agent_key not in actions:
                    actions[agent_key] = [task_id]
                else:
                    actions[agent_key].append(task_id)

        return actions

    def calculate_bid(self, agent, task, path, Qs=None):
        
        if Qs is None:
            max_score = -np.inf
            
            for i in range(len(path) + 1):
                new_path = path[:i] + [task.task_id] + path[i:]
                score = self.calculate_score(agent, new_path)
                if score > max_score:
                    max_score = score
            return max_score - self.calculate_score(agent, path)  # The bid is the increase in the score
        
        else:                        
            return Qs[agent.name][task.task_id]

    def determine_insertion_point(self, agent, task, path):
        max_score = -np.inf
        insertion_point = 0
        for i in range(len(path) + 1):
            new_path = path[:i] + [task.task_id] + path[i:]
            score = self.calculate_score(agent, new_path)
            if score > max_score:
                max_score = score
                insertion_point = i
        return insertion_point
    
    def calculate_score(self, agent, path):
        score = 0
        temp_position = agent.position  # Start with the agent's current position
        temp_time = agent.next_free_time  # Start with the agent's current free time
        for task_id in path:
            task = self.task_dict[task_id]
            score += self.calculate_task_score(agent, task, temp_position, temp_time)            
            total_distance = np.linalg.norm(temp_position - task.position)
            temp_position = task.position  # Update the temporary position to the task's position
            temp_time += total_distance / agent.max_speed  # Update the temporary free time
                              
        return score 



    def calculate_task_score(self, agent, task, temp_position, temp_time):
        total_distance = np.linalg.norm(temp_position - task.position)
        quality = agent.fit2Task[task.typeIdx]
        time = temp_time + total_distance / agent.max_speed
        if time < self.current_makespan:
            return (-2.5 * total_distance / self.max_dist + 160.0 * quality + 2.0 * (self.current_makespan - time))
        else:
            return (-2.5 * total_distance / self.max_dist + 160.0 * quality - 2.0 * (time - self.current_makespan))

    def calculate_total_time(self, agent, path):
        temp_position = agent.position  # Start with the agent's current position
        temp_time = agent.next_free_time  # Start with the agent's current free time
        for task_id in path:
            task = self.task_dict[task_id]
            total_distance = np.linalg.norm(temp_position - task.position)
            temp_position = task.position  # Update the temporary position to the task's position
            temp_time += total_distance / agent.max_speed  # Update the temporary free time
        return temp_time


