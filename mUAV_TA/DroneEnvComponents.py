import numpy as np
import math

from .MultiDroneEnvUtils import EnvUtils as utils

#---------- Class UAV ----------#
class UAV:
    
    def __init__(self, id, name, position, uavType, env, altitude=1000):
        
        self.env = env
        self.name = name
        self.id = id
        
        self.position = position
        self.altitude = altitude    

        #0 - Idle / 1 - Navigating / 2 - In Task / 3 - Retuning to Base / (4) - Out of Service
        self.state = 0            
        self.task_start = -1
        self.task_finished = -1
        self.fail_event = -1        
        
        self.type = uavType
        self.typeIdx = env.sceneData.UavIndex[self.type]        
        
        self.initialCap2Task = env.sceneData.UavCapTable[self.type] 
        self.currentCap2Task = env.sceneData.UavCapTable[self.type] 
        self.expectedCap2Task = env.sceneData.UavCapTable[self.type]
        #self.timeAtTask = 0

        self.attackCap = 0

        if uavType == "F1" or uavType == "F2" :
            self.attackCap = 4

        self.max_speed = env.sceneData.maxSpeeds[self.type]
        self.relay_area = env.sceneData.relayArea[self.type]             
        self.fail_multiplier = env.sceneData.failTable[self.type] 
        self.engage_range = env.sceneData.engage_range[self.type] 
            
        self.tasks = [env.task_idle]
        self.tasks_done = {}
        
        self.next_free_time = 0
        self.next_free_position = position
             
#---------- UAV Internal Capabilities ----------#
    def allocate(self, task):
                
        if task not in self.tasks and task.status != 2:
           
            if task.id != 0:
                                                        
                    time_to_task = np.linalg.norm(self.next_free_position - task.position) / self.max_speed 
                    end_time = self.next_free_time + time_to_task + task.task_duration
                    
                    #if end_time < max_time_steps: 
                    
                    if self.tasks[0].id == 0:
                        self.tasks[0] = task
                    else:
                        self.tasks.append(task)

                    self.next_free_time = end_time
                    self.next_free_position = task.position
                    
                    #Update the task requirements considering the new allocation
                    task.addAgentCap(self, time_to_task)
                                        
                    return True 
            else:
                
                self.tasks = [task]
                self.next_free_time = -1
                self.next_free_position = self.position   
                
                return False                 

        
        return False
    
    def desAllocate(self, task):
                
        if task in self.tasks and task.id != 0:
                                
            self.tasks.remove(task)                        
            self.next_free_time = self.env.time_steps
            self.next_free_position = self.position
                        
            task.removeAgentCap(self)  

            if len(self.tasks) == 0:
                self.tasks = [self.env.task_idle]
                                          
            return True
        else:                        
            if task.id != 0:
                print(f'Warming: Task {task.id} is not in agent {self.id} list')            
            return False

    def desallocateAll(self):

        for task in self.tasks:
            self.desAllocate(task)
        

    def outOfService(self):

        self.state = -1        
        for task in self.tasks:
            self.desAllocate(task)


        
    def doTask(self, agent_dir, task_dir, distance, task_type):
                        
        desired_dir = task_dir        
        
        if True:#task_type == "Rec":
            
            #if np.dot(agent_dir, task_dir) < 0 and distance < 20:
            #    return -task_dir
            return np.array([0,0])
                            
        return desired_dir
            

    def avoid_obstacles(self, agent, obstacles, movement, sceneData):
        
        avoid_vector = np.array([0.0, 0.0])
        
        for obstacle in obstacles:
            
            #direction_to_task = task - agent.position        
            #distance_to_task = np.linalg.norm(direction_to_task)
            
            direction_to_obstacle = obstacle.position - agent.position
            distance_to_obstacle = np.linalg.norm(direction_to_obstacle)
            distance_to_zone = distance_to_obstacle - obstacle.size
    
            # Verificar se o agent está muito próximo do obstáculo
            #if distance_to_obstacle < obstacle.size * 3:
            if distance_to_zone < 40:
                                
                direction_normalized = direction_to_obstacle / distance_to_zone
    
                # Calcular a força de desvio com base na distância à zona de segurança
                avoidance_force = 0.5 / (1 - math.log(max(distance_to_zone, 1.05)))
    
                # Calcular o ângulo entre o vetor de direção ao obstáculo e o vetor de direção ao alvo
                angle_between = np.arctan2(movement[1], movement[0]) - np.arctan2(direction_to_obstacle[1], direction_to_obstacle[0])                
    
                # Normalizar o ângulo para ficar entre -pi e pi
                angle_between = (angle_between + np.pi) % (2 * np.pi) - np.pi
    
                # Escolher a rotação tangente mais curta com base no ângulo entre os vetores
                if angle_between > 0:
                    rotated_vector = utils.rotate_vector_90_degrees_clockwise(direction_normalized)
                else:
                    rotated_vector = utils.rotate_vector_90_degrees_counterclockwise(direction_normalized)
    
                # Aplicar uma força de desvio proporcional à distância ao obstáculo
                avoid_vector += rotated_vector * avoidance_force
    
        return avoid_vector
    
#---------- Class Task ----------#                    

class Task:
    def __init__(self, task_id, position, task_type, task_reqs, task_window, sceneData, max_time_steps,  is_active=True, info = None ):
        
        self.id = task_id
        self.position = position
        self.info = info
        self.type = task_type
        
        #Task Requirements
        self.orgReqs       = self.getRequirements(task_reqs, sceneData) #reqs when created       

        self.allocationDetails = {} # {"Agent id" : (np.array["caps"], "time init")}

        self.allocatedReqs = np.zeros(len(sceneData.TaskTypes)) #reqs considering allocated Agents
        self.doneReqs      = np.zeros(len(sceneData.TaskTypes)) #reqs filled by concluded executions
        self.currentReqs   = self.orgReqs.copy()
                
        self.typeIdx     = sceneData.TaskIndex[self.type] #index of the type
        self.fit2Agent   = [cap[self.typeIdx] for cap in sceneData.UavCapTable.values()]
        
        #Status: 0 - waiting Allocation / 1 - Allocated / 2 - Concluded
        self.status      = 0 
        self.task_window = task_window
        self.max_time_steps = max_time_steps
                    
        self.task_duration = sceneData.getTaskDuration(self.type)

        self.allocated = 0
        self.initTime = -1
        self.doneTime = -1
                            
        self.final_quality = -1
    
    def getRequirements (self, task_reqs, sceneData):
       
        # SceneData task type ordering
        task_types = sceneData.TaskTypes

        # Initialize requirements array
        requirements = np.zeros(len(task_types))

        # Fill in the requirements array based on task_reqs
        for task_type, value in task_reqs.items():
            index = task_types.index(task_type)
            requirements[index] = value
            
        return requirements
    
    def removeAgentCap(self, agent):        
        
        if self.status != 2:
            if agent.id in self.allocationDetails:
                self.allocatedReqs -= agent.currentCap2Task
                details = self.allocationDetails.pop(agent.id)            
                
                if len(self.allocationDetails) > 0:
                                    
                    if details[1] == self.initTime:
                        self.initTime = min(self.allocationDetails.items(), key=lambda item: item[1][1])[1][1]
                    
                    if details[1] + self.task_duration == self.doneTime:
                        self.doneTime = max(self.allocationDetails.items(), key=lambda item: item[1][1])[1][1] + self.task_duration

                else:

                    self.initTime = -1
                    self.doneTime = -1
            else:
                print(f'Warning: Tried to desAllocate {agent.id} without allocation in task {self.id}')
        #else:
        #    print("Warning: Tried Desallocate Concluded Task")


    def addAgentCap(self, agent, time_at_task):  
        
        if self.status != 2:
            time_end_task = time_at_task + self.task_duration

            self.allocationDetails[agent.id] = (agent.currentCap2Task, time_at_task)

            self.allocatedReqs += agent.currentCap2Task
            
            if time_at_task < self.initTime or self.initTime == -1:
                self.initTime = time_at_task
                
                if self.doneTime  == -1:
                    self.doneTime = time_end_task
                        
            if time_end_task > self.doneTime:
                self.doneTime = time_end_task

            self.status = 1 
        else:
            print("Warning: Tried Allocate Concluded Task")
       
        

#---------- Class Task ----------#     
class Threat:
    def __init__(self, id, position, max_speed, engage_range, attack, defence, target_agent=None):
        self.id = id
        self.position = position
        self.max_speed = max_speed
        
        self.target_agent = None
        self.relative_task = None
        
        self.engage_range = engage_range
        self.attack = attack
        self.defence = defence
        self.attackCap = 4
        
        self.status = 1


#---------- Class Obstacle ----------#
class Obstacle:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.detected_segments = []
        
#----------- Square Area ------------#
class SquareArea:
    def __init__(self, center, width, height):
        self.center = center #[x,y]
        self.width = width
        self.height = height
        
        self.top_left     = (center[0] - width/2 , center[1] - height/2)
        self.top_right    = (center[0] + width/2 , center[1] - height/2) 
        self.bottom_left  = (center[0] - width/2 , center[1] + height/2)  
        self.bottom_right = (center[0] + width/2 , center[1] + height/2)                
         
       
        