import numpy as np
import math
import MultiDroneEnvUtils as utils


#---------- Class UAV ----------#

class Drone:
    
    def __init__(self, drone_id, position, uavType,  sceneData, altitude=1000  ):
        
        self.drone_id = drone_id
        self.position = position
        self.altitude = altitude    

        #0 - Idle / 1 - Navigating / 2 - In Task / 3 - Retuning to Base / 4 - Out of Service
        self.state = 0            
        self.task_start = -1
        self.task_finished = -1
        
        
        self.type = uavType
        self.typeIdx = sceneData.UavIndex[self.type]        
        self.fit2Task = sceneData.UavCapTable[self.type]
                        
        self.max_speed = sceneData.maxSpeeds[self.type]
        self.relay_area = sceneData.relayArea[self.type]        
        
        self.tasks = []
        self.tasks_done = []
        
        self.has_capability = True
        
#---------- UAV Internal Capabilities ----------#

    def doTask(self, drone_hdg, task_hdg, distance, task_type):
                        
        if task.type == "Rec":
            
            if abs(drone_direction - task_direction) > 90 and distance > 20:
                
                return 
            


    def avoid_obstacles(self, drone, obstacles, task, sceneData):
        
        avoid_vector = np.array([0.0, 0.0])
        
        for obstacle in obstacles:
            
            direction_to_task = task - drone.position        
            direction_to_obstacle = obstacle.position - drone.position
            distance_to_obstacle = np.linalg.norm(direction_to_obstacle)
            distance_to_zone = distance_to_obstacle - obstacle.size
    
            # Verificar se o drone está muito próximo do obstáculo
            #if distance_to_obstacle < obstacle.size * 3:
            if distance_to_zone < 30:
                
                
                direction_normalized = direction_to_obstacle / distance_to_zone
    
                # Calcular a força de desvio com base na distância à zona de segurança
                avoidance_force = 0.5 / (1 - math.log(max(distance_to_zone, 1.05)))
    
                # Calcular o ângulo entre o vetor de direção ao obstáculo e o vetor de direção ao alvo
                angle_between = np.arctan2(direction_to_task[1], direction_to_task[0]) - np.arctan2(direction_to_obstacle[1], direction_to_obstacle[0])
    
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
                        
        self.is_active = is_active        
        self.type = task_type
        self.typeIdx = sceneData.TaskIndex[self.type]
        
        self.task_window = task_window
        self.task_duration = sceneData.getTaskDuration(self.type)
                    
        self.final_quality = -1

#---------- Class Obstacle ----------#

class Obstacle:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.detected_segments = []
        
