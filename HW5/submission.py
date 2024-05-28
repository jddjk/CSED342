from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()
    

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:  # Pacman's turn, maximize
                nextAgent = 1
                bestValue = float('-inf')
                actionResult = None
            else:  # Ghosts' turn, minimize
                nextAgent = agentIndex + 1 if agentIndex < gameState.getNumAgents() - 1 else 0
                bestValue = float('inf')
                actionResult = None

            nextDepth = depth + 1 if nextAgent == 0 else depth
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value, _ = minimax(nextAgent, nextDepth, successor)
                if (agentIndex == 0 and value > bestValue) or (agentIndex != 0 and value < bestValue):
                    bestValue, actionResult = value, action

            return bestValue, actionResult

    _, action = minimax(0, 0, gameState)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)
    def minimax(agentIndex, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        nextAgent = agentIndex + 1 if agentIndex < gameState.getNumAgents() - 1 else 0
        nextDepth = depth + 1 if nextAgent == 0 else depth

        values = []
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = minimax(nextAgent, nextDepth, successor)
            values.append(value)

        return max(values) if agentIndex == 0 else min(values)

    return minimax(1, 0, successorGameState)
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def expectimax(gameState, depth, player):
        if gameState.isWin() or gameState.isLose(): return (gameState.getScore(), Directions.STOP)
        if depth == 0: return (self.evaluationFunction(gameState), Directions.STOP)
        if player == 0: return max([(expectimax(gameState.generateSuccessor(0, action), depth, 1)[0], action) for action in gameState.getLegalActions(0)])
        if player == (gameState.getNumAgents() - 1): depth -= 1
        legalActions = gameState.getLegalActions(player)
        possibleScores = [expectimax(gameState.generateSuccessor(player, action), depth, (player + 1) % gameState.getNumAgents())[0] for action in legalActions]
        return (sum(possibleScores)/len(possibleScores), random.choice(legalActions))
    
    return expectimax(gameState, self.depth, 0)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectimax(agentIndex, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        results = []

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            result = expectimax(nextAgent, nextDepth, successor)
            results.append(result)

        if agentIndex == 0:
            return max(results)
        else:
            return sum(results) / len(results)
    successorGameState = gameState.generateSuccessor(0, action)
    return expectimax(1, 0, successorGameState)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def _biased_expectimax(currentAgent, currentDepth, currentState):
        if currentState.isWin() or currentState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(currentState), None

        legalActions = currentState.getLegalActions(currentAgent)
        if not legalActions:
            return self.evaluationFunction(currentState), None

        nextAgentIndex = (currentAgent + 1) % currentState.getNumAgents()
        increasedDepth = currentDepth + 1 if nextAgentIndex == 0 else currentDepth
        evaluatedActions = []

        stop_action_weight = 0.5 + 0.5 / len(legalActions)
        other_action_weight = 0.5 / len(legalActions)

        for action in legalActions:
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = _biased_expectimax(nextAgentIndex, increasedDepth, successor)
            evaluatedActions.append((result, action))

        if currentAgent == 0:
            return max(evaluatedActions)
        else:
            weightedSum = sum((stop_action_weight if action == Directions.STOP else other_action_weight) * result
                              for result, action in evaluatedActions)
            return weightedSum, None

    _, action = _biased_expectimax(0, 0, gameState)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def _biased_expectimax(currentAgent, currentDepth, currentState):
        if currentState.isWin() or currentState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(currentState), None

        legalActions = currentState.getLegalActions(currentAgent)
        if not legalActions:
            return self.evaluationFunction(currentState), None

        nextAgentIndex = (currentAgent + 1) % currentState.getNumAgents()
        increasedDepth = currentDepth + 1 if nextAgentIndex == 0 else currentDepth
        evaluatedActions = []

        stop_action_weight = 0.5 + 0.5 / len(legalActions)
        other_action_weight = 0.5 / len(legalActions)

        for action in legalActions:
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = _biased_expectimax(nextAgentIndex, increasedDepth, successor)
            evaluatedActions.append((result, action))

        if currentAgent == 0:
            return max(evaluatedActions)
        else:
            weightedSum = sum((stop_action_weight if action is Directions.STOP else other_action_weight) * result
                              for result, action in evaluatedActions)
            return weightedSum, None


    successorState  = gameState.generateSuccessor(0, action)
    result, _ = _biased_expectimax(1, 0, successorState)
    return result
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def evaluateActions(currentAgent, currentDepth, currentState):
        nextAgentIndex = (currentAgent + 1) % currentState.getNumAgents()
        increasedDepth = currentDepth + 1 if nextAgentIndex == 0 else currentDepth
        evaluatedActions = []

        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = expectiminimax(nextAgentIndex, increasedDepth, successor)
            evaluatedActions.append((result, action))

        if currentAgent == 0:
            return max(evaluatedActions)
        elif currentAgent % 2 == 1:
            return min(evaluatedActions)
        else:
            average = sum(result for result, _ in evaluatedActions) / len(evaluatedActions)
            return average, None

    def expectiminimax(currentAgent, currentDepth, currentState):
        if currentState.isWin() or currentState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(currentState), None

        if not currentState.getLegalActions(currentAgent):
            return self.evaluationFunction(currentState), None

        return evaluateActions(currentAgent, currentDepth, currentState)

    _, action = expectiminimax(0, 0, gameState)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)

    def evaluateResults(currentAgent, currentDepth, currentState):
        nextAgentIndex = (currentAgent + 1) % currentState.getNumAgents()
        increasedDepth = currentDepth + 1 if nextAgentIndex == 0 else currentDepth
        results = []

        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result = expectiminimax(nextAgentIndex, increasedDepth, successor)
            results.append(result)

        if currentAgent == 0:
            return max(results)
        elif currentAgent % 2 == 1:
            return min(results)
        else:
            return sum(results) / len(results)

    def expectiminimax(currentAgent, currentDepth, currentState):
        if currentState.isWin() or currentState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(currentState)

        if not currentState.getLegalActions(currentAgent):
            return self.evaluationFunction(currentState)

        return evaluateResults(currentAgent, currentDepth, currentState)

    return expectiminimax(1, 0, successorGameState)
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def handleMaxAgent(currentAgent, currentDepth, currentState, alpha, beta):
        value = float('-inf')
        chosenAction = None
        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = alpha_beta(nextAgent(currentAgent), increaseDepth(currentAgent, currentDepth), successor, alpha, beta)
            if result > value:
                value, chosenAction = result, action
                alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, chosenAction

    def handleMinAgent(currentAgent, currentDepth, currentState, alpha, beta):
        value = float('inf')
        chosenAction = None
        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = alpha_beta(nextAgent(currentAgent), increaseDepth(currentAgent, currentDepth), successor, alpha, beta)
            if result < value:
                value, chosenAction = result, action
                beta = min(beta, value)
            if alpha >= beta:
                break
        return value, chosenAction

    def handleAverageAgent(currentAgent, currentDepth, currentState, alpha, beta):
        results = []
        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = alpha_beta(nextAgent(currentAgent), increaseDepth(currentAgent, currentDepth), successor, alpha, beta)
            results.append(result)
        average = sum(results) / len(results)
        return average, None

    def alpha_beta(currentAgent, currentDepth, currentState, alpha, beta):
        if currentState.isWin() or currentState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(currentState), None
        if not currentState.getLegalActions(currentAgent):
            return self.evaluationFunction(currentState), None

        if currentAgent == 0:
            return handleMaxAgent(currentAgent, currentDepth, currentState, alpha, beta)
        elif currentAgent % 2 == 1:
            return handleMinAgent(currentAgent, currentDepth, currentState, alpha, beta)
        else:
            return handleAverageAgent(currentAgent, currentDepth, currentState, alpha, beta)

    def nextAgent(currentAgent):
        return (currentAgent + 1) % gameState.getNumAgents()

    def increaseDepth(currentAgent, currentDepth):
        return currentDepth + 1 if nextAgent(currentAgent) == 0 else currentDepth

    _, action = alpha_beta(0, 0, gameState, float('-inf'), float('inf'))
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)

    def handleMaxAgent(currentAgent, currentDepth, currentState, alpha, beta):
        value = float('-inf')
        chosenAction = None
        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = alpha_beta(nextAgent(currentAgent), increaseDepth(currentAgent, currentDepth), successor, alpha, beta)
            if result > value:
                value, chosenAction = result, action
                alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, chosenAction

    def handleMinAgent(currentAgent, currentDepth, currentState, alpha, beta):
        value = float('inf')
        chosenAction = None
        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = alpha_beta(nextAgent(currentAgent), increaseDepth(currentAgent, currentDepth), successor, alpha, beta)
            if result < value:
                value, chosenAction = result, action
                beta = min(beta, value)
            if alpha >= beta:
                break
        return value, chosenAction

    def handleAverageAgent(currentAgent, currentDepth, currentState, alpha, beta):
        results = []
        for action in currentState.getLegalActions(currentAgent):
            successor = currentState.generateSuccessor(currentAgent, action)
            result, _ = alpha_beta(nextAgent(currentAgent), increaseDepth(currentAgent, currentDepth), successor, alpha, beta)
            results.append(result)
        average = sum(results) / len(results)
        return average, None

    def alpha_beta(currentAgent, currentDepth, currentState, alpha, beta):
        if currentState.isWin() or currentState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(currentState), None
        if not currentState.getLegalActions(currentAgent):
            return self.evaluationFunction(currentState), None

        if currentAgent == 0:
            return handleMaxAgent(currentAgent, currentDepth, currentState, alpha, beta)
        elif currentAgent % 2 == 1:
            return handleMinAgent(currentAgent, currentDepth, currentState, alpha, beta)
        else:
            return handleAverageAgent(currentAgent, currentDepth, currentState, alpha, beta)

    def nextAgent(currentAgent):
        return (currentAgent + 1) % gameState.getNumAgents()

    def increaseDepth(currentAgent, currentDepth):
        return currentDepth + 1 if nextAgent(currentAgent) == 0 else currentDepth

    value, action = alpha_beta(1, 0, successorGameState, float('-inf'), float('inf'))
    return value
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function
def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  pacmanPos = currentGameState.getPacmanPosition()
  capsules = currentGameState.getCapsules()
  ghostPositions = currentGameState.getGhostPositions()
  food = currentGameState.getFood()
  # walls = currentGameState.getWalls()
  # width, height = walls.width, walls.height

  def closestDistance(position, targets):
    return min(manhattanDistance(position, target) for target in targets)
  def closestDistances(position, targets):
    grid = [
    "%%%%%%%%%%%%%%%%%%%%",
    "%......%G  G%......%",
    "%.%%...%%  %%...%%.%",
    "%.%o.%........%.o%.%",
    "%.%%.%.%%%%%%.%.%%.%",
    "%........P.........%",
    "%%%%%%%%%%%%%%%%%%%%"
    ]

    def heuristic(a, b):
      return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def bfs(start, goal, grid=grid):
      open_set = util.PriorityQueue()
      open_set.push(start, heuristic(start, goal))
      came_from = {}
      g_score = {start: 0}
      f_score = {start: heuristic(start, goal)}

      while not open_set.isEmpty():
          current = open_set.pop()

          if current == goal:
              return g_score[current]

          for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
              neighbor = (current[0] + dx, current[1] + dy)
              if 0 <= neighbor[0] < len(grid[0]) and 0 <= neighbor[1] < len(grid):
                  if grid[neighbor[1]][neighbor[0]] == '%':
                      continue

                  tentative_g_score = g_score[current] + 1
                  if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                      came_from[neighbor] = current
                      g_score[neighbor] = tentative_g_score
                      f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                      if neighbor not in [item[1] for item in open_set.heap]:
                          open_set.push(neighbor, f_score[neighbor])

      return float('inf')

    distances = [bfs(position, target, grid) for target in targets]
    return min(distances) if distances else float('inf')
  # capsule_positions = [(3, 3), (16, 3)]
  # generator_positions = [(8, 5), (11, 5)]
  winningScore = 300 if currentGameState.isWin() else 0

  scaredGhostPos = [ghost.getPosition() for ghost in currentGameState.getGhostStates() if ghost.scaredTimer > 0]
  numScaredGhosts = len(scaredGhostPos)
  distanceToClosestScaredGhost = min((manhattanDistance(pacmanPos, ghost) for ghost in scaredGhostPos), default=float("inf"))

  score = currentGameState.getScore()
  distanceToClosestCapsule = closestDistance(pacmanPos, capsules) if capsules else float('inf')
  distanceToClosestGhost = closestDistances(pacmanPos, ghostPositions)
  distanceToClosestFood = closestDistance(pacmanPos, [(x, y) for x in range(food.width) for y in range(food.height) if food[x][y]])
  if (len(capsules) == 2):
      evaluationScore = score + 10./distanceToClosestCapsule + 150./len(capsules) - 5./distanceToClosestGhost
      return evaluationScore
  elif(len(capsules) == 1):
      if (numScaredGhosts == 2):
        evaluationScore = score + 10./distanceToClosestScaredGhost + 150./len(capsules) + 50*len(capsules)
        return evaluationScore
      elif (numScaredGhosts == 1):
        evaluationScore = score + 10./distanceToClosestScaredGhost + 150./len(capsules) + 50*len(capsules) - 3./distanceToClosestGhost - 50*numScaredGhosts
        return evaluationScore
      else:
        evaluationScore = score + 10./distanceToClosestCapsule + 150./len(capsules)
        return evaluationScore

  else:
      if (numScaredGhosts == 2):
        evaluationScore = score + 10./distanceToClosestScaredGhost + 150./(len(capsules)+1) + 50*len(capsules)
        return evaluationScore
      elif (numScaredGhosts == 1):
        evaluationScore = score + 10./distanceToClosestScaredGhost + 150./(len(capsules)+1) + 50*len(capsules) - 3./distanceToClosestGhost - 50*numScaredGhosts
        return evaluationScore
      else:
        evaluationScore = score + 10./distanceToClosestFood + 150./(len(capsules)+1) - 4./(distanceToClosestGhost+0.001) + winningScore
        return evaluationScore
  # END_YOUR_ANSWER
#len(food.asList())
class MyOwnAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='scoreEvaluationFunction', depth='3'):
        super().__init__(evalFn, depth)
        
    def getAction(self, gameState):

      # BEGIN_YOUR_ANSWER
      def expectimax(gameState, depth, player):
          if gameState.isWin() or gameState.isLose(): return (gameState.getScore(), Directions.STOP)
          if depth == 0: return (self.evaluationFunction(gameState), Directions.STOP)
          if player == 0: return max([(expectimax(gameState.generateSuccessor(0, action), depth, 1)[0], action) for action in gameState.getLegalActions(0)])
          if player == (gameState.getNumAgents() - 1): depth -= 1
          legalActions = gameState.getLegalActions(player)
          possibleScores = [expectimax(gameState.generateSuccessor(player, action), depth, (player + 1) % gameState.getNumAgents())[0] for action in legalActions]
          return (sum(possibleScores)/len(possibleScores), random.choice(legalActions))
      
      return expectimax(gameState, self.depth, 0)[1]
      # END_YOUR_ANSWER
    
    def getQ(self, gameState, action):

      # BEGIN_YOUR_ANSWER
      successorGameState = gameState.generateSuccessor(0, action)

      def expectimax(agentIndex, depth, gameState):
          if gameState.isWin() or gameState.isLose() or depth == self.depth:
              return self.evaluationFunction(gameState)

          actions = gameState.getLegalActions(agentIndex)
          if not actions:
              return self.evaluationFunction(gameState)

          nextAgent = (agentIndex + 1) % gameState.getNumAgents()
          nextDepth = depth + 1 if nextAgent == 0 else depth
          results = []

          for action in actions:
              successor = gameState.generateSuccessor(agentIndex, action)
              result = expectimax(nextAgent, nextDepth, successor)
              results.append(result)

          if agentIndex == 0:
              return max(results)
          else:
              return sum(results) / len(results)

      return expectimax(1, 0, successorGameState)
      # END_YOUR_ANSWER

      
def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER

  return 'MyOwnAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction


# Pacman emerges victorious! Score: 1745
# Pacman emerges victorious! Score: 1755
# Pacman emerges victorious! Score: 1758
# Pacman emerges victorious! Score: 1603
# Pacman emerges victorious! Score: 1683
# Pacman emerges victorious! Score: 1760
# Pacman emerges victorious! Score: 1754
# Pacman emerges victorious! Score: 1751
# Pacman emerges victorious! Score: 1549
# Pacman emerges victorious! Score: 1741
# Pacman emerges victorious! Score: 1558
# Pacman emerges victorious! Score: 1761
# Pacman emerges victorious! Score: 1375
# Pacman emerges victorious! Score: 1767
# Pacman emerges victorious! Score: 1686
# Pacman emerges victorious! Score: 1749
# Pacman emerges victorious! Score: 1703
# Pacman emerges victorious! Score: 1772
# Pacman emerges victorious! Score: 1718
# Pacman emerges victorious! Score: 1677
# Average Score: 1693.25