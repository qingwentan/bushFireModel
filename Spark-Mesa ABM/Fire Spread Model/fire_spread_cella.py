import mesa
import numpy as np

def compute_state(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B


class FireAgent(mesa.Agent):
    """A fire agent which does nothing."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.stepcount = 0

    def step(self):
        print(self.stepcount)


class FireSpreadModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height, fuelprob, burnprob):
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.fuelprob = fuelprob
        self.burnprob = burnprob
        self.width = width
        self.height = height
        self.totaltime = 10
        self.timeidx = 0

        midwidth = round(width/2)
        midheight = round(height/2)-1

        self.state = np.zeros((height,width,self.totaltime))

        for i in range(height):
            for j in range(width):
                fuelrandomizer = np.random.random()
                if(fuelrandomizer>fuelprob):
                    #state of cell(i,j) at time 0 is not a fuel (1)
                    self.state[i,j,0] = 1
                else:
                    #state of cell(i,j) at time 0 is a fuel (2)
                    self.state[i,j,0] = 2    
     
        self.state[midheight,midwidth,0] = 3

    def step(self):
        #self.datacollector.collect(self)
        self.schedule.step()
        self.timeidx += 1
        if(self.timeidx < self.totaltime):
            for i in range(self.height):
                for j in range(self.width):
                    if(self.state[i,j,self.timeidx-1]==1):
                        #non ignitable cells will remain unignited
                        self.state[i,j,self.timeidx] = 1
                    elif(self.state[i,j,self.timeidx-1]==3):
                        #burning cells will burned down at the next step
                        self.state[i,j,self.timeidx]==4
                        #neighbor ignitable cells of ignited cells will spread fire with a certain probability
                        if(i==0):
                            #top cells
                            if(j==0):
                                #top left corner cell
                                if(np.random.random()<self.burnprob):
                                    if(self.state[0,1,self.timeidx-1]==2):
                                        self.state[0,1,self.timeidx] = 3
                                else:
                                    if(self.state[0,1,self.timeidx-1] != 3):
                                        self.state[0,1,self.timeidx] = self.state[0,1,self.timeidx-1]

                                if(np.random.random()<self.burnprob):
                                    if(self.state[1,0,self.timeidx-1]==2):
                                        self.state[1,0,self.timeidx] = 3
                                else:
                                    if(self.state[1,0,self.timeidx-1] != 3):
                                        self.state[1,0,self.timeidx] = self.state[1,0,self.timeidx-1]

                                if(np.random.random()<self.burnprob):
                                    if(self.state[1,1,self.timeidx-1]==2):
                                        self.state[1,1,self.timeidx] = 3
                                else:
                                    if(self.state[1,1,self.timeidx-1] != 3):
                                        self.state[1,1,self.timeidx] = self.state[1,1,self.timeidx-1]

                            elif(j==self.width-1):
                                #top right corner cell

                                #bottom neighbor of top right corner cell
                                if(np.random.random()<self.burnprob):
                                    if(self.state[1,self.width-1,self.timeidx-1]==2):
                                        self.state[1,self.width-1,self.timeidx] = 3
                                else:
                                    if(self.state[1,self.width-1,self.timeidx-1] != 3):
                                        self.state[1,self.width-1,self.timeidx] = self.state[1,self.width-1,self.timeidx-1]

                                #left neighbor of top right corner cell
                                if(np.random.random()<self.burnprob):
                                    if(self.state[0,self.width-2,self.timeidx-1] == 2):
                                        self.state[0,self.width-2,self.timeidx] = 3
                                else:
                                    if(self.state[0,self.width-2,self.timeidx-1]!=3):
                                        self.state[0,self.width-2,self.timeidx] = self.state[0,self.width-2,self.timeidx-1]

                                #bottom left neighbor of top right corner cell
                                if(np.random.random()<self.burnprob):
                                    if(self.state[1,self.width-2,self.timeidx-1] == 2):
                                        self.state[1,self.width-2,self.timeidx] = 3
                                else:
                                    if(self.state[1,self.width-2,self.timeidx-1]!=3):
                                        self.state[1,self.width-2,self.timeidx] = self.state[1,self.width-2,self.timeidx-1]
                            else:
                                #top middle cells

                                #left neighbor
                                if(np.random.random()<self.burnprob):
                                    if(self.state[0,j-1,self.timeidx-1] == 2):
                                        self.state[0,j-1,self.timeidx] = 3
                                else:
                                    if(self.state[0,j-1,self.timeidx-1]!=3):
                                        self.state[0,j-1,self.timeidx] = self.state[0,j-1,self.timeidx-1]

                                #right neighbor
                                if(np.random.random()<self.burnprob):
                                    if(self.state[0,j+1,self.timeidx-1] == 2):
                                        self.state[0,j+1,self.timeidx] = 3
                                else:
                                    if(self.state[0,j+1,self.timeidx-1]!=3):
                                        self.state[0,j+1,self.timeidx] = self.state[0,j+1,self.timeidx-1]

                                #bottom neighbor
                                if(np.random.random()<self.burnprob):
                                    if(self.state[1,j,self.timeidx-1] == 2):
                                        self.state[1,j,self.timeidx] = 3
                                else:
                                    if(self.state[1,j,self.timeidx-1]!=3):
                                        self.state[1,j,self.timeidx] = self.state[1,j,self.timeidx-1]

                                #bottom left neighbor
                                if(np.random.random()<self.burnprob):
                                    if(self.state[1,j-1,self.timeidx-1] == 2):
                                        self.state[1,j-1,self.timeidx] = 3
                                else:
                                    if(self.state[1,j-1,self.timeidx-1]!=3):
                                        self.state[1,j-1,self.timeidx] = self.state[1,j-1,self.timeidx-1]

                        elif(i==self.height-1):
                            #bottom cells
                            if(j==0):
                                #bottom left corner cell

                                #top neighbor of bottom left corner cell
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-2,0,self.timeidx-1]==2):
                                        self.state[self.height-2,0,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-2,0,self.timeidx-1] != 3):
                                        self.state[self.height-2,0,self.timeidx] = self.state[0,self.height-2,self.timeidx-1]

                                #right neighbor of bottom left corner cell
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-1,1,self.timeidx-1]==2):
                                        self.state[self.height-1,1,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-1,1,self.timeidx-1] != 3):
                                        self.state[self.height-1,1,self.timeidx] = self.state[1,self.height-1,self.timeidx-1]

                                #top right neighbor of bottom left corner cell
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-2,1,self.timeidx-1]==2):
                                        self.state[self.height-2,1,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-2,1,self.timeidx-1] != 3):
                                        self.state[self.height-2,1,self.timeidx] = self.state[self.height-2,1,self.timeidx-1]

                            elif(j==self.width-1):
                                #bottom right corner cell

                                #top neighbor of bottom right corner cell
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-2,self.width-1,self.timeidx-1]==2):
                                        self.state[self.height-2,self.width-1,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-2,self.width-1,self.timeidx-1] != 3):
                                        self.state[self.height-2,self.width-1,self.timeidx] = self.state[self.height-2,self.width-1,self.timeidx-1]

                                #left neighbor of bottom right corner cell
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-1,self.width-2,self.timeidx-1]==2):
                                        self.state[self.height-1,self.width-2,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-1,self.width-2,self.timeidx-1] != 3):
                                        self.state[self.height-1,self.width-2,self.timeidx] = self.state[self.height-1,self.width-2,self.timeidx-1]

                                #top left neighbor of bottom right corner cell
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-2,self.width-2,self.timeidx-1]==2):
                                        self.state[self.height-2,self.width-2,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-2,self.width-2,self.timeidx-1] != 3):
                                        self.state[self.height-2,self.width-2,self.timeidx] = self.state[self.height-2,self.width-2,self.timeidx-1]
                            else:
                                #bottom middle cells

                                #top neighbor of bottom middle cells
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-2,j,self.timeidx-1]==2):
                                        self.state[self.height-2,j,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-2,j,self.timeidx-1] != 3):
                                        self.state[self.height-2,j,self.timeidx] = self.state[self.height-2,j,self.timeidx-1]
                                
                                #top left neighbor of bottom middle cells
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-2,j-1,self.timeidx-1]==2):
                                        self.state[self.height-2,j-1,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-2,j-1,self.timeidx-1] != 3):
                                        self.state[self.height-2,j-1,self.timeidx] = self.state[self.height-2,j-1,self.timeidx-1]

                                #left neighbor of bottom middle cells
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-1,j-1,self.timeidx-1]==2):
                                        self.state[self.height-1,j-1,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-1,j-1,self.timeidx-1] != 3):
                                        self.state[self.height-1,j-1,self.timeidx] = self.state[self.height-1,j-1,self.timeidx-1]

                                #right neighbor of bottom middle cells
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-1,j+1,self.timeidx-1]==2):
                                        self.state[self.height-1,j+1,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-1,j+1,self.timeidx-1] != 3):
                                        self.state[self.height-1,j+1,self.timeidx] = self.state[self.height-1,j+1,self.timeidx-1]

                                #top right neighbor of bottom middle cells
                                if(np.random.random()<self.burnprob):
                                    #burned if ignitable
                                    if(self.state[self.height-2,j+1,self.timeidx-1]==2):
                                        self.state[self.height-2,j+1,self.timeidx] = 3
                                else:
                                    #not burned
                                    if(self.state[self.height-2,j+1,self.timeidx-1] != 3):
                                        self.state[self.height-2,j+1,self.timeidx] = self.state[self.height-2,j+1,self.timeidx-1]

                        elif(j==0):
                            #left middle cells

                            #top neighbor of left middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i-1,0,self.timeidx-1]==2):
                                    self.state[i-1,0,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i-1,0,self.timeidx-1] != 3):
                                    self.state[i-1,0,self.timeidx] = self.state[i-1,0,self.timeidx-1]

                            #bottom neighbor of left middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i+1,0,self.timeidx-1]==2):
                                    self.state[i-1,0,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i+1,0,self.timeidx-1] != 3):
                                    self.state[i+1,0,self.timeidx] = self.state[i+1,0,self.timeidx-1]
                            
                            #top right neighbor of left middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i-1,1,self.timeidx-1]==2):
                                    self.state[i-1,1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i-1,1,self.timeidx-1] != 3):
                                    self.state[i-1,1,self.timeidx] = self.state[i-1,1,self.timeidx-1]

                            #bottom right neighbor of left middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i+1,1,self.timeidx-1]==2):
                                    self.state[i+1,1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i+1,1,self.timeidx-1] != 3):
                                    self.state[i+1,1,self.timeidx] = self.state[i+1,1,self.timeidx-1]

                            #right neighbor of left middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i,1,self.timeidx-1]==2):
                                    self.state[i,1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i,1,self.timeidx-1] != 3):
                                    self.state[i,1,self.timeidx] = self.state[i,1,self.timeidx-1]

                        elif(j==self.width-1):
                            #right middle cells

                            #top neighbor of right middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i-1,self.width-1,self.timeidx-1]==2):
                                    self.state[i-1,self.width-1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i-1,self.width-1,self.timeidx-1] != 3):
                                    self.state[i-1,self.width-1,self.timeidx] = self.state[i-1,self.width-1,self.timeidx-1]

                            #bottom neighbor of right middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i+1,self.width-1,self.timeidx-1]==2):
                                    self.state[i-1,self.width-1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i+1,self.width-1,self.timeidx-1] != 3):
                                    self.state[i+1,self.width-1,self.timeidx] = self.state[i+1,self.width-1,self.timeidx-1]
                            
                            #top left neighbor of right middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i-1,self.width-2,self.timeidx-1]==2):
                                    self.state[i-1,self.width-2,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i-1,self.width-2,self.timeidx-1] != 3):
                                    self.state[i-1,self.width-2,self.timeidx] = self.state[i-1,self.width-2,self.timeidx-1]

                            #bottom left neighbor of right middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i+1,self.width-2,self.timeidx-1]==2):
                                    self.state[i+1,self.width-2,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i+1,self.width-2,self.timeidx-1] != 3):
                                    self.state[i+1,self.width-2,self.timeidx] = self.state[i+1,self.width-2,self.timeidx-1]

                            #left neighbor of right middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i,self.width-2,self.timeidx-1]==2):
                                    self.state[i,self.width-2,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i,self.width-2,self.timeidx-1] != 3):
                                    self.state[i,self.width-2,self.timeidx] = self.state[i,self.width-2,self.timeidx-1]
                        else:
                            #middle cells with 8 neighbors

                            #top neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i-1,j,self.timeidx-1]==2):
                                    self.state[i-1,j,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i-1,j,self.timeidx-1] != 3):
                                    self.state[i-1,j,self.timeidx] = self.state[i-1,j,self.timeidx-1]

                            #bottom neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i+1,j,self.timeidx-1]==2):
                                    self.state[i-1,j,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i+1,j,self.timeidx-1] != 3):
                                    self.state[i+1,j,self.timeidx] = self.state[i+1,j,self.timeidx-1]
                            
                            #top left neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i-1,j-1,self.timeidx-1]==2):
                                    self.state[i-1,j-1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i-1,j-1,self.timeidx-1] != 3):
                                    self.state[i-1,j-1,self.timeidx] = self.state[i-1,j-1,self.timeidx-1]

                            #bottom left neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i+1,j-1,self.timeidx-1]==2):
                                    self.state[i+1,j-1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i+1,j-1,self.timeidx-1] != 3):
                                    self.state[i+1,j-1,self.timeidx] = self.state[i+1,j-1,self.timeidx-1]

                            #left neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i,j-1,self.timeidx-1]==2):
                                    self.state[i,j-1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i,j-1,self.timeidx-1] != 3):
                                    self.state[i,j-1,self.timeidx] = self.state[i,j-1,self.timeidx-1]

                            #top right neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i-1,j+1,self.timeidx-1]==2):
                                    self.state[i-1,j+1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i-1,j+1,self.timeidx-1] != 3):
                                    self.state[i-1,j+1,self.timeidx] = self.state[i-1,j+1,self.timeidx-1]

                            #bottom right neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i+1,j+1,self.timeidx-1]==2):
                                    self.state[i+1,j+1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i+1,j+1,self.timeidx-1] != 3):
                                    self.state[i+1,j+1,self.timeidx] = self.state[i+1,j+1,self.timeidx-1]

                            #right neighbor of middle cells
                            if(np.random.random()<self.burnprob):
                                #burned if ignitable
                                if(self.state[i,j+1,self.timeidx-1]==2):
                                    self.state[i,j+1,self.timeidx] = 3
                            else:
                                #not burned
                                if(self.state[i,j+1,self.timeidx-1] != 3):
                                    self.state[i,j+1,self.timeidx] = self.state[i,j+1,self.timeidx-1]

                    elif(self.state[i,j,self.timeidx-1]==4):
                        self.state[i,j,self.timeidx] = 4