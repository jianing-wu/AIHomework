from game import Directions
from game import Agent
from game import Actions
# from searchAgents import SearchAgent
import sys
import util
import time
import search
import copy


class SuperFoodSearchProblem(search.SearchProblem):
    """
    This search problem finds paths to eat at least two of each type of food.
    
    There are three types of food ',' '.' and ';', and there are always at least two of each on the map.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.foodType = ['.', ',', ';']
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2

        self.foodLocation = dict()
        for foodtype in self.foodType:
            self.foodLocation[foodtype] = []

        for x in range(1, right + 1):
            for y in range(1, top + 1):
                for foodtype in self.foodType:
                    if startingGameState.getCellContent(x,y) == foodtype:
                        self.foodLocation[foodtype].append((x, y))


        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """

        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """

        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """


        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]


        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)



def superFoodHeuristic(state, problem):
    # initialize locations of remaining food of all types

    
    '''
    # THIS CODE PIECE IS A HINT FOR HOW TO FIND ALL REMAINING FOOD
    visitedFoodLoc = ... #get food location you have eaten as a dict()
    allRemainingFood = copy.deepcopy(problem.foodLocation)
    for type in problem.foodType:
        for loc in visitedFoodLoc[type]:
            if loc in allRemainingFood[type]:
                allRemainingFood[type].remove(loc)
    # allRemainingFood is a dict() containing the locations of remaining food of each type; you may want to use this
    '''

    return 0 # Default to trivial solution
