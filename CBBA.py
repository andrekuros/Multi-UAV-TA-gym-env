import numpy as np

class CBBA():
    def __init__(self, drones, tasks, max_dist):
        self.num_drones = len(drones)
        self.agents = drones 
        self.max_dist = max_dist
        self.n_all_tasks = len(tasks)

    def allocate_tasks(self, agents, tasks, Qs=None):
        actions = {}
        task_dict = {task.task_id: task for task in tasks}  # Map unique task IDs to tasks
        remaining_tasks = set(task_dict.keys())
        bundles = {agent.drone_id: [] for agent in agents}  # Map unique agent IDs to bundles
                
        i = 0
        #while remaining_tasks:
        while i == 0:
            i = i + 1
            max_bid = -np.inf
            max_bid_task_id = None
            max_bid_agent_id = None

            # Compute bids for all tasks and select the task with the maximum bid
            for task_id in remaining_tasks:
                for agent in agents:
                    task = task_dict[task_id]
                    if task_id not in bundles[agent.drone_id]:
                        bid = self.calculate_bid(agent, task, Qs=Qs)
                        if bid > max_bid:
                            max_bid = bid
                            max_bid_task_id = task_id
                            max_bid_agent_id = agent.drone_id

            # Assign the task with the maximum bid to the agent with the maximum bid
            if max_bid_task_id is not None:
                bundles[max_bid_agent_id].append(max_bid_task_id)
                remaining_tasks.remove(max_bid_task_id)

        for agent_id, bundle in bundles.items():
            for task_id in bundle:
                agent_key = "agent" + str(agent_id)
                if agent_key not in actions:
                    actions[agent_key] = [task_id]
                else:
                    actions[agent_key].append(task_id)

        return actions


    
    def calculate_bid(self, agent, task, Qs=None):
        
        if Qs == None:
            total_distance = np.linalg.norm(agent.next_free_position - task.position) 
            quality = agent.fit2Task[task.typeIdx] 
            time = agent.next_free_time + total_distance / agent.max_speed      
            return (-2.5 * total_distance / self.max_dist  + 6.0 * quality - 1.5 *  time/200 - len(agent.tasks))  
        else:                        
            return Qs[agent.name][task.task_id]