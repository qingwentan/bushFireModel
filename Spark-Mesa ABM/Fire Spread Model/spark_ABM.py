# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:22:28 2022

@author: zengy
"""

"""
Created on Wed Jan  8 15:30:03 2020

This code was implemented by
Robin van den Berg, Beau Furnée, Wiebe Jelsma,
Hildebert Moulié, Philippe Nicolau & Louis Weyland
"""
from burn_cell import burnCell
from mesa import Model, Agent
from mesa.time import RandomActivation
#from mesa.space import MultiGrid
from space_v2 import MultiGrid
from datacollector_v2 import DataCollector
from environment.river import RiverCell
from environment.vegetation import TreeCell
#from agents.firetruck import Walker
#from agents.firetruck import Firetruck
from agents.agents import Walker
from agents.agents import Firetruck
#from environment.rain import Rain
#from environment.firebreak import BreakCell
import numpy as np
import random
import sys
sys.path.append('../')


# Creating and defining the model with all of its parameters
class ForestFire(Model):
    
    '''
    Create a forest fire model with fire fighting agents that try to extinguish the fire

    Args:
        height, width: The size of the grid to model;
        density: What fraction of grid cells have a tree in them;
        temperature: influences the number of spontaneous fires;
        truck_strategy: the tactic that firetrucks will adhere to;
        river_width: in the case that a river is present, what is its width;
        break_width: in the case that a pre-made fire break is present, what is its width;
        random_fires: boolean indicating whether spontaneous fires are present;
        vision: the distance fire fighting agents can look around them;
        truck_max_speed: the max speed with which firetruck agents can move around (grid cells/step);
        wind_strength: the speed with which the wind moves;
        wind_dir: string specifying the direction of the fire;
        sparse_ratio: the fraction of the vegetation that is sparse instead of dense;
        steps_to_extinguishment: number of steps it takes for firetrucks to extinguish a burning cell;
        placed_on_edges: indicates whether the firetrucks are placed randomly over the grid, or equispaced on the rim
    '''

    def __init__(
            self,
            height,
            width,
            truck_strategy,
            num_firetruck,
            vision,
            truck_max_speed,
            steps_to_extinguishment,
            placed_on_edges):
        super().__init__()

        # Initializing model parameters
        self.height = height
        self.width = width

        self.steps_to_extinguishment = steps_to_extinguishment
        self.placed_on_edges = placed_on_edges
        self.n_agents = 0

        self.agents = []
        self.firefighters_lists = []
        #self.initial_tree = height * width * density - \
        #    self.river_length * self.river_width
        #self.initial_tree = self.initial_tree - self.break_length * self.break_width

        # Setting-up model objects

        self.schedule_burncell = RandomActivation(self)
        self.schedule_FireTruck = RandomActivation(self)
        self.schedule = RandomActivation(self)
        self.current_step = 0


        # Creating the 2D grid
        self.grid = MultiGrid(height, width, torus=False)
        
        self.init_intensity()


        self.num_firetruck = num_firetruck
        self.truck_strategy = truck_strategy

        # initialize the population of firefighters
        self.init_firefighters(Firetruck, num_firetruck, truck_strategy, vision, truck_max_speed, placed_on_edges)


        # count number of fire took fire
        self.count_total_fire = 0

        '''  ####data collection part need to change
        
        # initiate the datacollector
        self.dc = DataCollector(self,
                                model_reporters={
                                    "Fine": lambda m: self.count_type(m, "Fine"),
                                    "On Fire": lambda m: self.count_type(m, "On Fire"),
                                    "Burned Out": lambda m: self.count_type(m, "Burned Out"),
                                    "Extinguished": lambda m: self.count_extinguished_fires(m)
                                },

                                # the data collector was modified to extract different data from different agents
                                agent_reporters={TreeCell: {"Life bar": "life_bar", "Burning rate": "burning_rate"},
                                                 Firetruck: {"Condition": "condition"}})

        # starting the simulation and collecting the data
        self.running = True
        self.dc.collect(self, [TreeCell, Firetruck])
        '''

        # presetting the indirect attack boundary
        self.buffer_x_min = int((self.width/2) - 30)
        self.buffer_x_max = int((self.width/2) + 30)
        self.buffer_y_min = int((self.height/2) - 30)
        self.buffer_y_max = int((self.height/2) + 30)
        self.buffer_coordinates = [self.buffer_x_min, self.buffer_x_max, self.buffer_y_min, self.buffer_y_max]
        self.tree_list_on_buffer = self.list_burn_in_buffer(self, self.buffer_coordinates)



    def init_firefighters(self, agent_type, num_firetruck,
                          truck_strategy, vision, truck_max_speed, placed_on_edges):
        '''
        Initialises the fire fighters
        placed_on_edges: if True --> places the firetrucks randomly over the grid.
        If False it places the firetrucks equispaced on the rim of the grid.
        '''

        if num_firetruck > 0:

            # Places the firetrucks on the edge of the grid with equal spacing
            if placed_on_edges:
                init_positions = self.equal_spread()
                for i in range(num_firetruck):
                    my_pos = init_positions.pop()
                    firetruck = self.new_firetruck(
                        Firetruck, my_pos, truck_strategy, vision, truck_max_speed)
                    self.schedule_FireTruck.add(firetruck)
                    self.schedule.add(firetruck)
                    self.firefighters_lists.append(firetruck)

            # Places the firetrucks randomly on the grid
            #else:
            #    for i in range(num_firetruck):
            #        x = random.randrange(self.width)
            #        y = random.randrange(self.height)

                    # make sure fire fighting agents are not placed in a river
            #        while self.grid.get_cell_list_contents((x, y)):
            #            if isinstance(self.grid.get_cell_list_contents(
            #                    (x, y))[0], RiverCell):
            #                x = random.randrange(self.width)
            #                y = random.randrange(self.height)
            #            else:
            #                break

            #        firetruck = self.new_firetruck(
            #            Firetruck, (x, y), truck_strategy, vision, truck_max_speed)
            #        self.schedule_FireTruck.add(firetruck)
            #        self.schedule.add(firetruck)
            #        self.firefighters_lists.append(firetruck)
            ####################################################
            
            # Places all the firetrucks at (0,0)
            else:
                for i in range(num_firetruck):
                    x = 0
                    y = 0

                    # make sure fire fighting agents are not placed in a river
                    while self.grid.get_cell_list_contents((x, y)):
                        if isinstance(self.grid.get_cell_list_contents(
                                (x, y))[0], RiverCell):
                            x = random.randrange(self.width)
                            y = random.randrange(self.height)
                        else:
                            break

                    firetruck = self.new_firetruck(
                        Firetruck, (x, y), truck_strategy, vision, truck_max_speed)
                    self.schedule_FireTruck.add(firetruck)
                    self.schedule.add(firetruck)
                    self.firefighters_lists.append(firetruck)


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
    

    def init_intensity(self):
        '''
        Creating intensity layer
        '''
        for i in range(self.width):
            for j in range(self.height):
                x = i
                y = j
                new_burn = self.new_agent(burnCell, (x, y))
                self.schedule_burncell.add(new_burn)
                self.schedule.add(new_burn)
                
    
    def step_intensity(self, new_input):
        '''
        use arrray data to change intensity
        '''
        x_array = np.where(~np.isnan(new_input))[0]
        y_array = np.where(~np.isnan(new_input))[1]
        for i in range(len(x_array)):
            #position = (x_array[i],y_array[i])
            #print(self.grid[x_array[i]][y_array[i]][0].fire_bar)         
            burn_cell = self.grid.get_cell_list_contents((x_array[i],y_array[i]))
            for content in burn_cell:
                if isinstance(content, burnCell):
                    burn_present = True
                    burn_object = content

            if burn_present:
                burn_object.fire_bar = new_input[x_array[i]][y_array[i]]
                burn_object.trees_claimed = 0
            
            
    def output_array(self):
        #result = [i[0].fire_bar for i in self.grid[:,:] if len(i)!=0]
        #result = [list(self.grid[i][j])[0].fire_bar for i in range(self.width) for j in range(self.height)]
        #result = np.array(result)
        burn_intensity = []
        for i in range(self.width):
            for j in range(self.height):
                burn_cell = self.grid.get_cell_list_contents((i,j))
                for content in burn_cell:
                    if isinstance(content, burnCell):
                        burn_intensity.append(content.fire_bar)
        burn_intensity = np.array(burn_intensity)
        burn_intensity = np.reshape(burn_intensity, (self.width, self.height))
        
        return burn_intensity
        
    @staticmethod
    def list_burn_cell(model):
        '''
        Helper method to count cell that intensity not 0
        '''
        cell_list = [cell for cell in model.schedule_burncell.agents if cell.fire_bar != 0]
        return cell_list    
        
            
    def step(self):
        '''
        Advance the model by one step.
        '''

        # save all burning trees for easy search
        self.burn_list = self.list_burn_cell(self)

        # if using optimized method, produce a matrix with the distances between the firetrucks and the burning veg
        if len(self.burn_list) > 0:
            if (self.truck_strategy == "Optimized closest"):
                self.assigned_list = self.assign_closest(
                    self.compute_distances(self.burn_list,
                                           self.firefighters_lists), self.burn_list)

            elif (self.truck_strategy == "Optimized Parallel attack"):
                self.assigned_list = self.assign_parallel(
                    self.compute_distances(self.burn_list, self.firefighters_lists),
                    self.burn_list)

            elif (self.truck_strategy == "Indirect attack"):
                self.assigned_list = self.assign_parallel(
                    self.compute_distances(self.burn_list, self.firefighters_lists),
                    self.burn_list)

            # progress the firetrucks by one step
            self.schedule_FireTruck.step()

        # collect data
        #self.dc.collect(self, [TreeCell, Firetruck]) # because of modified dc, now the agents need to be specified
        self.current_step += 1

        # Halt if no more fire
        if len(self.list_burn_cell(self)) == 0:
            print(" \n \n Fire is gone ! \n \n")
            self.running = False


    def compute_distances(self, burn_list, truck_list):
        '''
        Computes the distances between the firetrucks and the burning vegetation
        '''
        distances = [[0 for x in range(len(truck_list))] for y in range(len(burn_list))]
        for i in range(len(burn_list)):
            for j in range(len(truck_list)):
                distances[i][j] = (burn_list[i].pos[0] - truck_list[j].pos[0]) ** 2 + \
                    (burn_list[i].pos[1] - truck_list[j].pos[1]) ** 2
        return distances

    def assign_closest(self, matrix, tree_list):
        '''
        Uses the matrix produces by compute_distances() to assign the firetrucks to the closest burning vegetation
        '''
        assigned_trucks = [0 for x in range(self.num_firetruck)]

        ratio = Walker.firefighters_tree_ratio(self, self.num_firetruck, len(tree_list))

        # assign firetrucks to the closest fires
        matrix = np.asarray(matrix, dtype=int)
        while 0 in assigned_trucks:
            curr_smallest_pos = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)

            # if there is a surplus of firetrucks, allow them to go to the same fire
            if assigned_trucks[curr_smallest_pos[1]] == 0 and tree_list[curr_smallest_pos[0]].trees_claimed < ratio:
                assigned_trucks[curr_smallest_pos[1]] = tree_list[curr_smallest_pos[0]]

                tree_list[curr_smallest_pos[0]].trees_claimed += 1
            matrix[curr_smallest_pos] = 100000
        return assigned_trucks

    def assign_parallel(self, matrix, tree_list):
        '''
        Uses compute_distances() to carry out the parallel attack
        '''
        assigned_trucks = [0 for x in range(self.num_firetruck)]

        # if there is a surplus of firetrucks, allow them to go to the same fire
        ratio = Walker.firefighters_tree_ratio(self, self.num_firetruck, len(tree_list))

        matrix = np.asarray(matrix, dtype=int)
        for i in range(len(matrix[0])):
            curr_best = [matrix[0][i], tree_list[0].life_bar, 0]
            indices = [j for j, x in enumerate(matrix[:, i]) if x <= curr_best[0]]
            for m in indices:
                if tree_list[m].trees_claimed >= ratio:
                    indices.remove(m)
            if len(indices) > 1:
                for k in indices:
                    if matrix[k][i] <= curr_best[0] and tree_list[k].fire_bar >= curr_best[1]:
                        curr_best = [matrix[k][i], tree_list[k].fire_bar, k]
                tree_list[curr_best[2]].trees_claimed += 1
            assigned_trucks[i] = tree_list[curr_best[2]]
        return assigned_trucks



    @staticmethod
    def list_burn_in_buffer(model, coordinates):
        '''
        Helper method to count burn cell lying on the buffer
        coordinates = [self.buffer_x_min,self.buffer_x_max,self.buffer_y_min,self.buffer_y_max]
        '''

        cell_list_b = [cell for cell in model.schedule_burncell.agents
                       if((cell.pos[1] == coordinates[2])
                          and (coordinates[0] <= cell.pos[0])
                           and (cell.pos[0] <= coordinates[1]))]
        cell_list_u = [cell for cell in model.schedule_burncell.agents
                       if((cell.pos[1] == coordinates[3])
                          and (coordinates[0] <= cell.pos[0])
                           and (cell.pos[0] <= coordinates[1]))]
        cell_list_l = [cell for cell in model.schedule_burncell.agents
                       if ((cell.pos[0] == coordinates[0])
                           and (coordinates[2] < cell.pos[1])
                           and (cell.pos[1] < coordinates[3]))]
        cell_list_r = [cell for cell in model.schedule_burncell.agents
                       if ((cell.pos[0] == coordinates[1])
                           and (coordinates[2] < cell.pos[1])
                           and (cell.pos[1] < coordinates[3]))]

        cell_list = cell_list_r + cell_list_l + cell_list_b + cell_list_u

        return cell_list

    @staticmethod
    def count_extinguished_fires(model):
        '''
        Helper method to count extinguished fires in a given condition in a given model.
        '''

        count = 0
        for firetruck in model.schedule_FireTruck.agents:
            count += firetruck.extinguished

        return count

    def remove_agent(self, agent):
        '''
        Method that enables us to remove passed agents.
        '''
        self.n_agents -= 1

        # Remove agent from grid
        self.grid.remove_agent(agent)

        # Remove agent from model
        self.agents.remove(agent)

    def equal_spread(self):
        '''
        Function to equally space the firetruck along the edge of the grid
        '''
        edge_len = self.height - 1
        total_edge = 4 * edge_len

        x = 0
        y = 0

        start_pos = [(x, y)]
        spacing = total_edge / self.num_firetruck
        total_edge -= spacing
        step = 0

        while total_edge > 0:
            fill_x = edge_len - x
            fill_y = edge_len - y

            # special cases (<4)
            if spacing > edge_len:
                if x == 0:
                    x += edge_len
                    y += spacing - edge_len
                else:
                    x, y = y, x

            # all other cases
            else:
                # Increasing x
                if y == 0 and x + spacing <= edge_len and step < 2:
                    x += spacing
                    step = 1

                # x maxxed, increasing y
                elif x + spacing > edge_len and y + (spacing - fill_x) < edge_len and step < 3:
                    x += fill_x
                    y += spacing - fill_x
                    step = 2

                # x&y maxxed, decreasing x
                elif x - (spacing - fill_y) >= 0 and y + fill_y >= edge_len and step < 4:
                    x -= (spacing - fill_y)
                    y += fill_y
                    step = 3

                # x emptied, decreasing y
                elif x - spacing < 0 and step < 5:
                    y -= (spacing - x)
                    x = 0
                    step = 4

            start_pos += [(round(x), round(y))]
            total_edge -= spacing

        return start_pos


'''
# To be used if you want to run the model without the visualiser:
temperature = 20
truck_strategy = 'Goes to the closest fire'
density = 0.6
width = 100
height = 100
num_firetruck = 30
vision = 100
max_speed = 1
steps_to_extinguishment = 2
placed_on_edges = True
break_number = 0
river_number = 0
river_width = 0
random_fires = 1
wind_strength = 8
#wind_dir = "N"
wind_dir = "North"
# wind[0],wind[1]=[direction,speed]
wind = [1, 2]
fire = ForestFire(
    height,
    width,
    density,
    temperature,
    truck_strategy,
    river_number,
    river_width,
    break_number,
    random_fires,
    num_firetruck,
    vision,
    max_speed,
    wind_strength,
    wind_dir,
    steps_to_extinguishment,
    placed_on_edges
)
fire.run_model()

#results = fire.dc.get_model_vars_dataframe()
#agent_variable = fire.dc.get_agent_vars_dataframe()
#results_firetrucks = fire.dc.get_model_vars_dataframe()

#print(agent_variable[0])
#print(agent_variable[1])
'''

