import numpy as np
import math
import MultiDroneEnvUtils as utils


#---------- Class UAV ----------#
class Drone:
    
    def __init__(self, drone_id, name, position, uavType,  sceneData, altitude=1000 ):
        
        self.name = name
        self.drone_id = drone_id
        
        self.position = position
        self.altitude = altitude    

        #0 - Idle / 1 - Navigating / 2 - In Task / 3 - Retuning to Base / (4) - Out of Service
        self.state = 0            
        self.task_start = -1
        self.task_finished = -1
        self.fail_event = -1        
                
        self.type = uavType
        self.typeIdx = sceneData.UavIndex[self.type]        
        self.fit2Task = sceneData.UavCapTable[self.type]                        
        self.max_speed = sceneData.maxSpeeds[self.type]
        self.relay_area = sceneData.relayArea[self.type]             
        self.fail_multiplier = sceneData.failTable[self.type]  
            
        self.tasks = []
        self.tasks_done = []
        
        self.next_free_time = 0
        self.next_free_position = position
        
        self.has_capability = True        
        
#---------- UAV Internal Capabilities ----------#
    def allocate(self, task, time_step, max_time_steps):
                
        if not task.task_id in self.tasks:
        
            time_to_task = np.linalg.norm(self.next_free_position - self.position) / self.max_speed + task.task_duration                        
            end_time = self.next_free_time + time_to_task
            
            #if end_time < max_time_steps: 
            
            self.tasks.append(task.task_id)
            self.next_free_time = end_time
            self.next_free_position = task.position                
            
            return 1.0                        
        return -1.0    
    
    def desallocateAll(self, time_step):
        
        self.tasks = []
        self.next_free_time = time_step
        self.next_free_position = self.position   
        



    def doTask(self, drone_dir, task_dir, distance, task_type):
                        
        desired_dir = task_dir
        
        if True:#task_type == "Rec":
            
            #if np.dot(drone_dir, task_dir) < 0 and distance < 20:
            #    return -task_dir
            return np.array([0,0])
               
                
        return desired_dir
            


    def avoid_obstacles(self, drone, obstacles, movement, sceneData):
        
        avoid_vector = np.array([0.0, 0.0])
        
        for obstacle in obstacles:
            
            #direction_to_task = task - drone.position        
            #distance_to_task = np.linalg.norm(direction_to_task)
            
            direction_to_obstacle = obstacle.position - drone.position
            distance_to_obstacle = np.linalg.norm(direction_to_obstacle)
            distance_to_zone = distance_to_obstacle - obstacle.size
    
            # Verificar se o drone está muito próximo do obstáculo
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
    def __init__(self, task_id, position, task_type, task_window, sceneData, is_active=True ):
        
        self.task_id = task_id
        self.position = position
        self.type = task_type
        self.typeIdx = sceneData.TaskIndex[self.type]
        self.fit2Agent = [cap[self.typeIdx] for cap in sceneData.UavCapTable.values()]#sceneData.UavCapTable[self.type] 
        
        self.status = 0 # 0 - waiting Allocation / 1 - Allocated / 2 - Concluded
        self.task_window = task_window
        self.task_duration = sceneData.getTaskDuration(self.type)
                    
        self.final_quality = -1

#---------- Class Obstacle ----------#

class Obstacle:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.detected_segments = []
        
