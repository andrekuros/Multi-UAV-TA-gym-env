import numpy as np

class Drone:
    def __init__(self, drone_id, position, target=None):
        self.drone_id = drone_id
        self.position = position       
        self.target = target
        self.tasks = []
        self.tasks_done = []
   


class Target:
    def __init__(self, target_id, position, is_active=True):
        self.target_id = target_id
        self.position = position
        self.is_active = is_active


