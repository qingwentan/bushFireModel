# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 13:33:31 2022

@author: zengy
"""

import random
import numpy as np
import math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from numpy import linalg as LA

class TreeCell(Agent):

    '''
    A tree cell.

    Attributes:
        x, y: Grid coordinates
        condition: Can be "Fine", "On Fire", or "Burned Out"
        unique_id: (x,y) tuple.
        life_bar : looks at the life bar of the tree

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
        self.condition = "Fine"
        self.life_bar = 100       # give the tree a life bar
        self.burning_rate = 20  # need to change that as well
        self.trees_claimed = 0
        self.fire_bar = 0

        self.veg_state = 0.4

        # assigning density with the given probability
        #if random.uniform(0, 1) < self.model.sparse_ratio:
        #    self.veg_density = -0.4
        #else:
        #    self.veg_density = 0

        self.fireinitstep = None

        #

    def step(self):
        '''
        If the tree is on fire, spread it to fine trees nearby.
        '''
        self.trees_claimed = 0
        if self.condition == "On Fire":

            if self.fireinitstep != self.model.current_step:
                neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=1)
                for neighbor in neighbors:

                    if isinstance(neighbor, TreeCell) and neighbor.condition == "Fine":
                        # or neighbor.condition == "Is Extinguished" \
                        # and neighbor.life_bar > 0 and neighbor.fireinitstep != self.model.current_step:

                        # probability of spreading
                        #prob_sp = self.prob_of_spreading(neighbor, self.model.wind_dir, self.model.wind_strength)
                        prob_sp = 0.5
                        if random.uniform(0, 1) < prob_sp:
                            neighbor.condition = "On Fire"
                            neighbor.fire_bar = self.model.steps_to_extinguishment
                            neighbor.fireinitstep = self.model.current_step
                            self.model.count_total_fire += 1 / \
                                (self.model.height * self.model.width * self.model.density)

                # if on fire reduce life_bar
                if self.life_bar != 0:
                    self.life_bar -= self.burning_rate
                    if self.life_bar == 0:
                        self.condition = "Burned Out"
                else:
                    self.condition = "Burned Out"

    def get_pos(self):
        return self.pos

    def prob_of_spreading(self, neighbour, wind_dir, wind_strength):

        p_h = 0.58
        p_veg = neighbour.veg_state
        p_den = neighbour.veg_density
        p_s = 1  # no elavation
        c2 = 0.131
        c1 = 0.045
        theta = 0  # in case wind_strength is zero

        # if wind actually exists
        if self.model.wind_strength != 0:
            neighbour_vec = [neighbour.pos[0] - self.pos[0], neighbour.pos[1] - self.pos[1]]
            wind_vec = [wind_dir[0], wind_dir[1]]

            # get the angle theat between wind in the spreading direction
            dot_product = np.dot(neighbour_vec, wind_vec)
            theta = math.acos((dot_product / (LA.norm(neighbour_vec) * LA.norm(wind_vec))))

        p_w = math.exp(c1 * wind_strength) * math.exp(c2 * wind_strength * (math.cos(theta) - 1))

        p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_w * p_s

        return p_burn


class ForestFire(Model):
    def __init__(
            self,
            height,
            width,
            density,
            truck_strategy,
            random_fires,
            num_firetruck,
            vision,
            truck_max_speed,
            steps_to_extinguishment):
        super().__init__()

        # Initializing model parameters
        self.height = height
        self.width = width
        self.density = density

        self.steps_to_extinguishment = steps_to_extinguishment

        self.n_agents = 0

        self.agents = []
        self.firefighters_lists = []
        self.initial_tree = height * width * density 


        # Setting-up model objects
        self.schedule_TreeCell = RandomActivation(self)
        self.schedule_FireTruck = RandomActivation(self)
        self.schedule = RandomActivation(self)
        self.current_step = 0
        


        # Creating the 2D grid
        self.grid = MultiGrid(height, width, torus=False)


        # Create the vegetation agents
        self.init_vegetation(TreeCell, self.initial_tree)

        # add the agents to the vegetation specific schedule and the overal schedule
        for i in range(len(self.agents)):
            self.schedule_TreeCell.add(self.agents[i])
            self.schedule.add(self.agents[i])

        self.random_fires = random_fires

        self.num_firetruck = num_firetruck
        self.truck_strategy = truck_strategy

        # initialize the population of firefighters
        self.init_firefighters(Direct_Attack, num_firetruck, truck_strategy, vision, truck_max_speed)
        
        # initiate the datacollector
        self.dc = DataCollector({
                                    "Fine": lambda m: self.count_type(m, "Fine"),
                                    "On Fire": lambda m: self.count_type(m, "On Fire"),
                                    "Burned Out": lambda m: self.count_type(m, "Burned Out"),
                                    "Extinguished": lambda m: self.count_extinguished_fires(m)
                                })

        # starting the simulation and collecting the data
        self.running = True
        self.dc.collect(self)


        # Initialise fire in the middle if possible otherwise random
        self.agents[0].condition = "On Fire"

        # get initial fire position and define the square
        self.init_fire_pos = self.agents[0].pos

        # count number of fire took fire
        self.count_total_fire = 0
        
        # starting the simulation and collecting the data
        self.running = True
        
    def new_firetruck(self, agent_type, pos, truck_strategy,
                      vision, truck_max_speed):
        '''
        Method that enables us to add a fire agent.
        '''
        self.n_agents += 1

        # Create a new agent of the given type
        new_agent = agent_type(
            self,
            self.n_agents,
            pos,
            truck_strategy,
            vision,
            truck_max_speed)

        # Place the agent on the grid
        self.grid.place_agent(new_agent, pos)

        # And add the agent to the model so we can track it
        self.agents.append(new_agent)

        return new_agent
    
    def new_agent(self, agent_type, pos):
        '''
        Method that enables us to add agents of a given type.
        '''
        self.n_agents += 1

        # Create a new agent of the given type
        new_agent = agent_type(self, self.n_agents, pos)

        # Place the agent on the grid
        self.grid.place_agent(new_agent, pos)

        # And add the agent to the model so we can track it
        self.agents.append(new_agent)

        return new_agent
    
    @staticmethod
    def count_type(model, tree_condition):
        '''
        Helper method to count trees in a given condition in a given model.
        '''
        count = 0
        for tree in model.schedule_TreeCell.agents:
            if tree.condition == tree_condition:
                count += 1
        return count
        
    @staticmethod
    def count_extinguished_fires(model):
        '''
        Helper method to count extinguished fires in a given condition in a given model.
        '''

        count = 0
        for firetruck in model.schedule_FireTruck.agents:
            count += firetruck.extinguished

        return count
    
        
    def init_vegetation(self, agent_type, n):
        '''
        Creating trees
        '''
        x = random.randrange(self.width)
        y = random.randrange(self.height)

        # initiating vegetation in the centre if possible, otherwise random position
        self.new_agent(agent_type, (x, y))

        # Placing all other vegetation
        for i in range(int(n - 1)):
            while not self.grid.is_cell_empty((x, y)):
                x = random.randrange(self.width)
                y = random.randrange(self.height)
            self.new_agent(agent_type, (x, y))



    def init_firefighters(self, agent_type, num_firetruck,
                          truck_strategy, vision, truck_max_speed):
        '''
        Initialises the fire fighters
        placed_on_edges: if True --> places the firetrucks randomly over the grid.
        If False it places the firetrucks equispaced on the rim of the grid.
        '''

        if num_firetruck > 0:


            # Places the firetrucks randomly on the grid
            for i in range(num_firetruck):
                x = random.randrange(self.width)
                y = random.randrange(self.height)


                firetruck = self.new_firetruck(
                        Direct_Attack, (x, y), truck_strategy, vision, truck_max_speed)
                self.schedule_FireTruck.add(firetruck)
                self.schedule.add(firetruck)
                self.firefighters_lists.append(firetruck)
                
    @staticmethod
    def list_tree_by_type(model, tree_condition):
        '''
        Helper method to count trees in a given condition in a given model.
        '''
        tree_list = [tree for tree in model.schedule_TreeCell.agents if tree.condition == tree_condition]
        return tree_list

    def step(self):
        '''
        Advance the model by one step.
        '''

        # progress the fire spread by a step
        self.schedule_TreeCell.step()

        # save all burning trees for easy search
        self.tree_list = self.list_tree_by_type(self, "On Fire")

        # if using optimized method, produce a matrix with the distances between the firetrucks and the burning veg
        if len(self.tree_list) > 0:
            if (self.truck_strategy == "Optimized closest"):
                self.assigned_list = self.assign_closest(
                    self.compute_distances(self.tree_list,
                                           self.firefighters_lists), self.tree_list)


            # progress the firetrucks by one step
            self.schedule_FireTruck.step()

        # collect data
        self.dc.collect(self) # because of modified dc, now the agents need to be specified
        self.current_step += 1

        # if spontaneous fires are turned on, check whether one ignites in this step
        if self.random_fires:
            randtree = int(random.random() * len(self.agents))
            if self.agents[randtree].condition == "Fine":
                self.randomfire(self, randtree)

        # Halt if no more fire
        if self.count_type(self, "On Fire") == 0:
            print(" \n \n Fire is gone ! \n \n")
            self.running = False

    @staticmethod
    def randomfire(self, randtree):
        '''
        Possibly ignites a new fire, chance depending on the temperature
        '''
        #if (random.random() < (math.exp(self.temperature / 15) / 300.0)):
        #    self.agents[randtree].condition = "On Fire"
        if (random.random() < (math.exp(30 / 15) / 300.0)):
            self.agents[randtree].condition = "On Fire"


class Direct_Attack(Agent):
    '''A class specific to a firetruck'''
    def __init__(self, model, unique_id, pos,
                 truck_strategy, vision, truck_max_speed):
        super().__init__(unique_id, model)
        self.pos = pos
        self.unique_id = unique_id
        self.condition = "Full"
        self.extinguished = 0
        self.truck_strategy = truck_strategy
        self.vision = vision
        self.truck_max_speed = truck_max_speed
        self.life_bar = -5

    def firefighters_tree_ratio(self, number_of_firefighters, trees_on_fire):
        '''Calculates the ratio of fire fighters and vegetation'''
        if trees_on_fire > 0:
            return int(math.ceil(number_of_firefighters / trees_on_fire))
        return 1
    
    def get_pos(self):
        '''Returns the position of the firetruck'''
        return self.pos
    
    def step(self):
        # set step according to strategy
        if (self.truck_strategy == 'Goes to the closest fire'):
            self.closestfire_move()

        elif (self.truck_strategy == "Optimized closest"):
            self.optimized_closest_fire()
        self.extinguish()


    def take_step(self, closest_neighbor):
        '''This function takes a step in the direction of a given neighbour'''

        # calculates total places to move in x and y direction
        places_to_move_y = closest_neighbor.pos[1] - self.pos[1]
        places_to_move_x = closest_neighbor.pos[0] - self.pos[0]

        # lowers the max speed of the trucks when destination is closer
        speed_x = min(self.truck_max_speed, abs(places_to_move_x))
        speed_y = min(self.truck_max_speed, abs(places_to_move_y))

        new_x, new_y = self.pos[0], self.pos[1]

        # determine new position of fire fighting agent
        if places_to_move_x > 0:
            new_x += speed_x
        if places_to_move_x < 0:
            new_x -= speed_x
        if places_to_move_y > 0:
            new_y += speed_y
        if places_to_move_y < 0:
            new_y -= speed_y

        self.model.grid.move_agent(self, (new_x, new_y))

    def random_move(self):
        '''
        This method should get the neighbouring cells (Moore's neighbourhood)
        select one, and move the agent to this cell.
        '''

        # get all neighbours within reachable distance
        cell_list = self.model.grid.get_neighborhood(self.pos, moore=True, radius=self.truck_max_speed)

        # choose the new position
        new_pos = cell_list[random.randint(0, len(cell_list) - 1)]

        self.model.grid.move_agent(self, new_pos)


    def closestfire_move(self):
        '''Makes firetrucks move towards closest fire'''

        # calculate fire fighter to burning vegetation ratio
        ratio = self.firefighters_tree_ratio(
            self.model.num_firetruck, self.model.count_type(
                self.model, "On Fire"))
        fire_intheneighborhood = False

        # skip through a percentage of the vision to find the closest fire more efficiently
        limited_vision_list = [i for i in range(2, 100, 2)]
        for i in range(len(limited_vision_list)):
            limited_vision = int(self.vision * limited_vision_list[i] / 100.)

            if i > 0:
                inner_radius = int(
                    self.vision * limited_vision_list[i - 1] / 100.)
            else:
                inner_radius = 0

            # find hot trees in neighborhood
            #neighbors_list = self.model.grid.get_neighbors(
            #    self.pos, moore=True, radius=limited_vision, inner_radius=inner_radius)
            neighbors_list = self.model.grid.get_neighbors(
                self.pos, moore=True, radius=limited_vision)

            # filter for trees that are on fire
            neighbors_list = [
                x for x in neighbors_list if x.condition == "On Fire"]

            # find closest fire
            min_distance = limited_vision ** 2
            for neighbor in neighbors_list:
                if neighbor.trees_claimed < ratio:
                    distance = abs(neighbor.pos[0] ** 2 - self.pos[0] ** 2) + \
                        abs(neighbor.pos[1] ** 2 - self.pos[1] ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_neighbor = neighbor
                        fire_intheneighborhood = True
            if fire_intheneighborhood:
                break



        # move toward fire if it is actually in the neighborhood
        if fire_intheneighborhood:
            self.take_step(closest_neighbor)
            closest_neighbor.trees_claimed += 1

        # if fire not in the neighboorhood, do random move
        else:
            self.random_move()



    def extinguish(self):
        '''This function has firetrucks extinguishing the burning trees in their moore neighbourhood'''
        neighbors_list = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=1, include_center=True)

        # if there is a burning tree in the moore neighbourhood, lower its firebar by 1
        for tree in neighbors_list:
            if tree.condition == "On Fire":
                tree.fire_bar -= 1
                if tree.fire_bar == 0:
                    tree.condition = "Is Extinguished"
                    self.extinguished += 1
                    





# To be used if you want to run the model without the visualiser:
temperature = 20
truck_strategy = 'Goes to the closest fire'
density = 0.6
width = 100
height = 100
num_firetruck = 3
vision = 100
truck_max_speed = 20
steps_to_extinguishment = 2
break_number = 0
river_number = 0
river_width = 0
random_fires = 1
wind_strength = 8
wind_dir = "N"
# wind[0],wind[1]=[direction,speed]
wind = [1, 2]
fire = ForestFire(
    height,
    width,
    density,
    truck_strategy,
    random_fires,
    num_firetruck,
    vision,
    truck_max_speed,
    steps_to_extinguishment
)

fire.run_model()

results = fire.dc.get_model_vars_dataframe()
agent_variable = fire.dc.get_agent_vars_dataframe()
results_firetrucks = fire.dc.get_model_vars_dataframe()

print(results)
#print(agent_variable[0])
#print(agent_variable[1])

