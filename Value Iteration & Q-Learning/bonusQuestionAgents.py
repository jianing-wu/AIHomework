"""Custom agent code for the bonus question."""
import util
from featureExtractors import *
from game import Actions
from qlearningAgents import ApproximateQAgent, PacmanQAgent

####################
# Bonus Question 1 #
####################


class CustomExtractor1(FeatureExtractor):
    """Extracts your custom features from the game state."""

    def getFeatures(self, state, action):

        # Currently, this code is copied from SimpleExtractor in
        # featureExtractors.py. It includes simple features for a basic reflex Pacman:
        # - whether food will be eaten
        # - how far away the next food is
        # - whether a ghost collision is imminent
        # - whether a ghost is one step away

        # Feel free to add or remove features as you see fit.

        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        scaredGhosts = []
        unscaredGhosts = []
        for i, ghost in enumerate(ghosts, 1):
            if state.getGhostScared(i):
                scaredGhosts.append(ghost)
            else:
                unscaredGhosts.append(ghost)


        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-unscared-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls)
            for g in unscaredGhosts)

        features["#-of-scared-ghosts-1-step-away"] = 10*sum(
            (next_x, next_y) in Actions.getLegalNeighbors(sg, walls)
            for sg in scaredGhosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-unscared-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width *
                                                      walls.height)
        features.divideAll(10.0)
        return features


class CustomQAgent1(ApproximateQAgent):
    """Custom Q-learning agent."""

    def __init__(self, extractor='CustomExtractor1', **args):
        self.featExtractor: FeatureExtractor = util.lookup(
            extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

        # Feel free to modify these parameters.
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.2

    def update(self, state, action, nextState, reward):
        # Currently, this just calls the ApproximateQAgent update rule. Feel
        # free to modify it as you see fit.
        ApproximateQAgent.update(self, state, action, nextState, reward)


####################
# Bonus Question 2 #
####################


class CustomExtractor2(FeatureExtractor):
    """Extracts your custom features from the game state."""
    def closestGhost(self, pos, ghostPositions, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if (pos_x, pos_y) in ghostPositions:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def getFeatures(self, state, action):

        # Currently, this code is copied from SimpleExtractor in
        # featureExtractors.py. It includes simple features for a basic reflex Pacman:
        # - whether food will be eaten
        # - how far away the next food is
        # - whether a ghost collision is imminent
        # - whether a ghost is one step away

        # Feel free to add or remove features as you see fit.

        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        scaredGhosts = []
        unscaredGhosts = []
        for i, ghost in enumerate(ghosts, 1):
            if state.getGhostScared(i):
                scaredGhosts.append(ghost)
            else:
                unscaredGhosts.append(ghost)

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-unscared-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls)
            for g in unscaredGhosts)

        # features["#-of-scared-ghosts-1-step-away"] = 10*sum(
        #     (next_x, next_y) in Actions.getLegalNeighbors(sg, walls)
        #     for sg in scaredGhosts)
        # # if not features["#-of-ghosts-1-step-away"]:
        # features["#-of-scared-ghosts-1-step-away"] = sum(
        #     (next_x, next_y) in Actions.getLegalNeighbors(sg, walls)
        #     for sg in scaredGhosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-unscared-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # if not features["#-of-unscared-ghosts-1-step-away"] and features["#-of-scared-ghosts-1-step-away"]:
        #     features["eats-ghost"] = 2.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width *
                                                      walls.height)

        scaredGhostDist = self.closestGhost((next_x, next_y), scaredGhosts, walls)
        if scaredGhostDist is not None:
            features["closest-scared-ghost"] = float(scaredGhostDist) / (walls.width *
                                                      walls.height)
        # else:
        #     features["closest-scared-ghost"] = 1.0

        unscaredGhostDist = self.closestGhost((next_x, next_y), unscaredGhosts, walls)
        if unscaredGhostDist is not None:
            features["closest-unscared-ghost"] = float(unscaredGhostDist) / (walls.width *
                                                      walls.height)
        # else:
        #     features["closest-unscared-ghost"] = 1.0

        features.divideAll(10.0)
        return features


class CustomQAgent2(ApproximateQAgent):
    """Custom Q-learning agent."""

    def __init__(self, extractor='CustomExtractor2', **args):
        self.featExtractor: FeatureExtractor = util.lookup(
            extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

        # Feel free to modify these parameters.
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.3

    def update(self, state, action, nextState, reward):
        # Currently, this just calls the ApproximateQAgent update rule. Feel
        # free to modify it as you see fit.
        ApproximateQAgent.update(self, state, action, nextState, reward)
