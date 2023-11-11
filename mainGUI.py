#%%%
import pygame
from pygame.locals import *
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Task Allocation Algorithms
from mUAV_TA.DroneEnv import MAX_INT, MultiUAVEnv
from mUAV_TA.DroneEnv import env
from TaskAllocation.BehaviourBased.swarm_gap import SwarmGap
from TaskAllocation.MarketBased.CBBA import CBBA

import mUAV_TA.MultiDroneEnvUtils as utils

#RL Models/Policies
from TaskAllocation.RL_Policies.Tianshou_Policy import _get_model
import torch
from tianshou.data import Batch
import torch.nn.functional as F

# Initialize pygame
pygame.init()


# Colors and dimensions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50

# Screen setup
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Drone Simulation Interface')

# # Placeholders for parameters and algorithm
# config = {
#     "param1": None,  # Replace with your actual parameter names
#     "param2": None,
#     # ... [Other parameters] ...
# }

selected_algorithm = "Random"
algorithms = ["Random", "Swarm-GAP"]  # Extend this list as needed


config = utils.agentEnvOptions(     
            render_speed = -1,
            simulation_frame_rate = 0.02 , #Std 0.02
            max_time_steps = 300 ,                    
            random_init_pos = False,
            num_obstacles = 0,
            hidden_obstacles = False,
 )

# Define rects for interface components
play_button_rect = pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT - 75, BUTTON_WIDTH, BUTTON_HEIGHT)
algorithm_buttons = [pygame.Rect(50, 50 + (i * 60), BUTTON_WIDTH, BUTTON_HEIGHT) for i in range(len(algorithms))]

# Render function
def render_interface():
    screen.fill(WHITE)
    
    # Play button
    pygame.draw.rect(screen, GREEN, play_button_rect)
    font = pygame.font.SysFont(None, 36)
    text = font.render('PLAY', True, BLACK)
    screen.blit(text, (SCREEN_WIDTH - 130, SCREEN_HEIGHT - 65))
    
    # Algorithm selection buttons
    for i, algo in enumerate(algorithms):
        color = GREEN if algo == selected_algorithm else RED
        pygame.draw.rect(screen, color, algorithm_buttons[i])
        text = font.render(algo, True, BLACK)
        screen.blit(text, (60, 60 + (i * 60)))
    
    pygame.display.flip()

def run_simulation():
    # Integrate the simulation logic from main.py
    # Use the 'config' dictionary and 'selected_algorithm' variable
    pass

# Main loop
done = False
while not done:
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True
        elif event.type == MOUSEBUTTONDOWN:
            if play_button_rect.collidepoint(event.pos):
                run_simulation()
            for i, rect in enumerate(algorithm_buttons):
                if rect.collidepoint(event.pos):
                    selected_algorithm = algorithms[i]

    render_interface()

pygame.quit()
