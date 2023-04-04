import numpy as np
import math
import Utils as ut

class Drone:
    def __init__(self, drone_id, position, sensors=None ):
        
        self.drone_id = drone_id
        self.position = position               
        
        self.sensors = sensors
        
        self.tasks = []
        self.tasks_done = []
        self.has_capability = True
        
    def avoid_obstacles(self, drone, obstacles, target):
        
        avoid_vector = np.array([0.0, 0.0])
        
        for obstacle in obstacles:
            
            direction_to_target = target - drone.position        
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
                angle_between = np.arctan2(direction_to_target[1], direction_to_target[0]) - np.arctan2(direction_to_obstacle[1], direction_to_obstacle[0])
    
                # Normalizar o ângulo para ficar entre -pi e pi
                angle_between = (angle_between + np.pi) % (2 * np.pi) - np.pi
    
                # Escolher a rotação tangente mais curta com base no ângulo entre os vetores
                if angle_between > 0:
                    rotated_vector = ut.rotate_vector_90_degrees_clockwise(direction_normalized)
                else:
                    rotated_vector = ut.rotate_vector_90_degrees_counterclockwise(direction_normalized)
    
                # Aplicar uma força de desvio proporcional à distância ao obstáculo
                avoid_vector += rotated_vector * avoidance_force
    
        return avoid_vector

                    

class Target:
    def __init__(self, target_id, position, target_type, is_active=True ):
        
        self.target_id = target_id
        self.position = position
        
        self.final_quality = -1
        
        self.is_active = is_active        
        self.target_type = target_type


class Obstacle:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.detected_segments = []
        
