# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:40:47 2022

@author: zengy
"""

import random
from mesa import Agent
import math
import numpy as np
from numpy import linalg as LA


class burnCell(Agent):

    '''
    A cell contains intensity

    Attributes:
        x, y: Grid coordinates
        unique_id: (x,y) tuple.
        fire_bar : real-time intensity

    unique_id isn't strictly necessary here,
    but it's good practice to give one to each
    agent anyway.
    '''

    def __init__(self, model, unique_id, pos):
        '''
        Create a new tree.
        Args:
        pos: The tree's coordinates on the grid. Used as the unique_id
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.unique_id = unique_id
        self.trees_claimed = 0
        self.fire_bar = 0
        #self.fire_bar = -10


    def step(self):
        '''
        If the tree is on fire, spread it to fine trees nearby.
        '''
        self.fire_bar -= 1

    def get_pos(self):
        return self.pos


