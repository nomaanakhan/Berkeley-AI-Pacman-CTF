# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class OffensiveReflexAgent(CaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    self.border = []
    if self.red:
      self.borderX = gameState.data.layout.width/2 - 1
    else:
      self.borderX = gameState.data.layout.width / 2

    for i in range(gameState.data.layout.height):
      if not gameState.data.layout.walls[self.borderX][i]:
        self.border.append(((self.borderX), i))

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    #if gameState.getAgentState(self.index).numCarrying != 0:
    #  bestDist = 9999
    #  for action in actions:
    #    successor = self.getSuccessor(gameState, action)
    #    pos2 = successor.getAgentPosition(self.index)
    #    dist = self.getMazeDistance(self.start, pos2)
    #    if dist < bestDist:
    #      bestAction = action
    #      bestDist = dist
    #  return bestAction

    bestAction = random.choice(bestActions)

    return bestAction

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    features['stop'] = 0
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    features['reverse'] = 0
    if action == rev: features['reverse'] = 1
    features['enemyClose'] = 0
    features['enemyRealClose'] = 0
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    threats = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(threats) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in threats]
      #print dists
      features['threatDistance'] = min(dists)
      for dist in dists:
        if dist <= 1:
          #print dist
          features['enemyClose'] = 1
        #if dist == 1:
        #  features['enemyRealClose'] = 1
    #print features['enemyClose']
    capsuleList = self.getCapsules(successor)
    if len(capsuleList) > 0:
      minCapDistance = min([self.getMazeDistance(myPos, cap) for cap in capsuleList])
      if features['enemyClose'] == 0:
        features['distanceToCapsule'] = minCapDistance
      else:
        features['distanceToCapsule'] = 5 * minCapDistance

    #distance to the nearest score spot
    disttoscore = min([self.getMazeDistance(myPos, scorespot) for scorespot in self.border])
    features['distToScore'] = disttoscore

    return features

  def getWeights(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()
    #function here to compute value for dist to score as a function of pelets carried, 0 if none, 0 if no enemies in x distance
    if gameState.getAgentState(self.index).numCarrying != 0:
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      threats = [a for a in enemies if not a.isPacman and a.getPosition() != None]
      if len(threats) > 0:
        closest = min([self.getMazeDistance(myPos, a.getPosition()) for a in threats])
        if closest >= 10:
            distToScore = 0
        else:
            distToScore = -5*gameState.getAgentState(self.index).numCarrying
    else:
      distToScore = 0


    return {'successorScore': 100, 'distanceToFood': -2, 'threatDistance': 1, 'enemyClose': -100, 'stop': -300,
            'distanceToCapsule': -2, 'enemyRealClose': -100, 'reverse': -20, 'distToScore': distToScore}


class DefensiveReflexAgent(CaptureAgent):
	"""
	A simple reflex agent that takes score-maximizing actions. It's given 
	features and weights that allow it to prioritize defensive actions over any other.
	"""

	def registerInitialState(self, gameState):
		#self.start = gameState.getAgentPosition(self.index)
		CaptureAgent.registerInitialState(self, gameState)

	def getSuccessor(self, gameState, action):
		"""
		Finds the next successor which is a grid position (location tuple).
		"""
		successor = gameState.generateSuccessor(self.index, action)
		pos = successor.getAgentState(self.index).getPosition()
		if pos != nearestPoint(pos):
			# Only half a grid position was covered
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		# Computes whether we're on defense (1) or offense (0)
		features['onDefense'] = 1
		if myState.isPacman: features['onDefense'] = 0

		# Computes distance to invaders we can see
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(dists)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
		
	def evaluate(self, gameState, action):
		"""
		Computes a linear combination of features and feature weights
		"""
		features = self.getFeatures(gameState, action)
		weights = self.getWeights(gameState, action)
		return features * weights

	def chooseAction(self, gameState):
		
		#if gameState.getAgentState(CaptureAgent.getOpponents(self, gameState)[0]).scaredTimer:
		#	print "Hello ========"
			
		actions = gameState.getLegalActions(self.index)

		# You can profile your evaluation time by uncommenting these lines
		# start = time.time()
		values = [self.evaluate(gameState, a) for a in actions]
		# print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		
		return random.choice(bestActions)