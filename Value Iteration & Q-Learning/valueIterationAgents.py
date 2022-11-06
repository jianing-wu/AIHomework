# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import itertools
import math

import util
from learningAgents import ValueEstimationAgent
from mdp import MarkovDecisionProcess


class ValueIterationAgent(ValueEstimationAgent):
    """A ValueIterationAgent takes a Markov decision process (see mdp.py) on
    initialization and runs value iteration for a given number of iterations
    using the supplied discount factor."""

    def __init__(self,
                 mdp: MarkovDecisionProcess,
                 discount: float = 0.9,
                 iterations: int = 100):
        """Your value iteration agent should take an mdp on construction, run
        the indicated number of iterations and then act according to the
        resulting policy.

        Some useful mdp methods you will use:
          - mdp.getStates()
          - mdp.getPossibleActions(state)
          - mdp.getTransitionStatesAndProbs(state, action)
          - mdp.getReward(state, action, nextState)
          - mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here.

        for i in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    newValues[state] = self.values[state]
                else:
                    newValues[state] = max([self.computeQValueFromValues(state, action) for action in actions])
            self.values = newValues

    def getValue(self, state):
        """Return the value of the state (computed in __init__)."""
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """Computes the Q-value of action in state from the value function
        stored in self.values."""
        nextStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        if not nextStatesAndProbs or self.mdp.isTerminal(state):
            return self.values[state]

        Q = 0.0
        for (nextState, prob) in nextStatesAndProbs:
            Q += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState]) 
        return Q

    def computeActionFromValues(self, state):
        """The policy is the best action in the given state according to the
        values currently stored in self.values.

        You may break ties any way you see fit.  Note that if there are no legal
        actions, which is the case at the terminal state, you should return
        None.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None

        bestAtion = None
        bestQ = float("-inf")
        for action in actions:
            Q = self.computeQValueFromValues(state, action)
            if Q > bestQ:
                bestQ = Q
                bestAction = action
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
