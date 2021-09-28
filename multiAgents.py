# multiAgents.py
# --------------
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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        "Here we want to define a function to call position of the food to calculate summation of stored food" 
        def Available_food_Summation(cur_pos, food_positions, Flag_variable=False):
            Distance_of_Food = []
            for food in food_positions:
                "To calculate distance, we used mahathan distance"
                Distance_of_Food.append(util.manhattanDistance(food, cur_pos))
                "The initial value of the flag is false "
                "So we normal;ized it to get Finding_Normalized_Valued distance of the food"
            if Flag_variable:
                return Finding_Normalized_Value(sum(Distance_of_Food) if sum(Distance_of_Food) > 0 else 1)
            else:
                return sum(Distance_of_Food) if sum(Distance_of_Food) > 0 else 1
        "We are going to define the score value which can getscore from Successorgamestate"
        score = successorGameState.getScore()
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        "We are defineing position, state and scores of the ghost"
        def Useful_Information_Ghost(cur_pos, ghost_states, radius, scores):
            Ghost_Number_Variable = 0
            "Based on the Manhathan distance the score would be subtracted or added"
            for ghost in ghost_states:
                if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                    scores -= 30
                    Ghost_Number_Variable += 1
            return scores
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        "We want to collect all related variables to the food to calculated scores of them"
        def Useful_Information_Food(cur_pos, food_pos, cur_score):
            Available_Food_New = Available_food_Summation(cur_pos, food_pos)
            Available_Food_Current = Available_food_Summation(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
            Available_Food_New = 1/Available_Food_New
            Available_Food_Current = 1/Available_Food_Current
            "Based on the new available food we increase or decrease the score "
            if Available_Food_New > Available_Food_Current:
                cur_score += (Available_Food_New - Available_Food_Current) * 3
            else:
                cur_score -= 20
                "Here we are trying to find out next available food with shortest distance"
            Available_food_Next_Distance = Available_Nearest_Small_Food(cur_pos, food_pos)
            Available_Food_Current_Distance = Available_Nearest_Small_Food(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
            if Available_food_Next_Distance < Available_Food_Current_Distance:
                cur_score += (Available_food_Next_Distance - Available_Food_Current_Distance) * 3
            else:
                cur_score -= 20
            return cur_score
        
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        "Here closest food distance would be calculated"
        def Available_Nearest_Small_Food(cur_pos, food_pos):
            Distance_of_Food = []
            for food in food_pos:
                Distance_of_Food.append(util.manhattanDistance(food, cur_pos))
                "After Calculation, minimum distance would be returned"
            return min(Distance_of_Food) if len(Distance_of_Food) > 0 else 1

        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        def Finding_Normalized_Value(distance, layout):
            return distance

        return Useful_Information_Food(newPos, newFood.asList(), Useful_Information_Ghost(newPos, newGhostStates, 2, score))
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

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
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """ "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        PACMAN = 0
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        def Information_About_Agent_MaxValue(state, depth):
            if state.isWin() or state.isLose():
                return state.getScore()
            "Number of actions would be recorded in the action variable"
            actions = state.getLegalActions(PACMAN)
            "Best score would be initialized by the infinite value at first"
            Recorded_Best_Scored = float("-inf")
            score = Recorded_Best_Scored
            Happend_Action_Best_One = Directions.STOP
            for action in actions:
                score = Useful_Information_Agent(state.generateSuccessor(PACMAN, action), depth, 1)
                "We will consider Score to find out the best recorded score"
                if score > Recorded_Best_Scored:
                    Recorded_Best_Scored = score
                    Happend_Action_Best_One = action
                    "We would find best score in all depth until we reach to the first depth"
            if depth == 0:
                return Happend_Action_Best_One
            else:
                return Recorded_Best_Scored
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        "We are considering Information of the agent such as depth , ghost and state"
        def Useful_Information_Agent(state, depth, ghost):
            "based on the status we will return score"
            if state.isLose() or state.isWin():
                return state.getScore()
            Available_Ghost_Next_One = ghost + 1
            '''
            Based on the state of the ghost we are trying to find out 
            the number of available ghosts
            then we can find out which ghost would be closest one
            '''
            if ghost == state.getNumAgents() - 1:
                Available_Ghost_Next_One = PACMAN
            actions = state.getLegalActions(ghost)
            Recorded_Best_Scored = float("inf")
            score = Recorded_Best_Scored
            '''
            Based on the action that we recorded, we wil call evaluate function
            to evaluate and give us a score
            '''
            for action in actions:
                if Available_Ghost_Next_One == PACMAN:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = Information_About_Agent_MaxValue(state.generateSuccessor(ghost, action), depth + 1)
                else:
                    score = Useful_Information_Agent(state.generateSuccessor(ghost, action), depth, Available_Ghost_Next_One)
                if score < Recorded_Best_Scored:
                    Recorded_Best_Scored = score
            return Recorded_Best_Scored
        return Information_About_Agent_MaxValue(gameState, 0)
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        '''
        We are using implemented parts in previous question to find out
        value of max and min here also
        '''
        def Information_About_Agent_MaxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            "Number of actions would be recorded in the action variable"
            actions = state.getLegalActions(PACMAN)
            "Best score would be initialized by the infinite value at first"
            Recorded_Best_Scored = float("-inf")
            score = Recorded_Best_Scored
            Happend_Action_Best_One = Directions.STOP
            for action in actions:
                score = Information_About_Agent_MinValue(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > Recorded_Best_Scored:
                    Recorded_Best_Scored = score
                    Happend_Action_Best_One = action
                    '''
                    Finding value of apha
                    Which is max value between alpha and recorded best scores
                    '''
                alpha = max(alpha, Recorded_Best_Scored)
                if Recorded_Best_Scored > beta:
                    return Recorded_Best_Scored
                "We would find best score in all depth until we reach to the first depth"

            if depth == 0:
                return Happend_Action_Best_One
            else:
                return Recorded_Best_Scored
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        def Information_About_Agent_MinValue(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            Available_Ghost_Next_One = ghost + 1
            '''
            Based on the state of the ghost we are trying to find out 
            the number of available ghosts
            then we can find out which ghost would be closest one
            '''
            if ghost == state.getNumAgents() - 1:
                Available_Ghost_Next_One = PACMAN
            actions = state.getLegalActions(ghost)
            Recorded_Best_Scored = float("inf")
            score = Recorded_Best_Scored
            '''
            Based on the action that we recorded, we wil call evaluate function
            to evaluate and give us a score
            '''
            for action in actions:
                if Available_Ghost_Next_One == PACMAN: 
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = Information_About_Agent_MaxValue(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = Information_About_Agent_MinValue(state.generateSuccessor(ghost, action), depth, Available_Ghost_Next_One, alpha, beta)
                if score < Recorded_Best_Scored:
                    Recorded_Best_Scored = score
                    '''
                    Same as finding value of Alpha
                    Finding value of Betta
                    Which is max value between Betta and recorded best scores
                    '''
                beta = min(beta, Recorded_Best_Scored)
                if Recorded_Best_Scored < alpha:
                    return Recorded_Best_Scored
            return Recorded_Best_Scored
        return Information_About_Agent_MaxValue(gameState, 0, float("-inf"), float("inf"))
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        PACMAN = 0
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        def Information_About_Agent_MaxValue(state, depth):
            if state.isWin() or state.isLose():
                return state.getScore()
            "We are trying to find the action which is getting back from the Pacman"
            actions = state.getLegalActions(PACMAN)
            "At first the initial value would be infinite"
            Recorded_Best_Scored = float("-inf")
            score = Recorded_Best_Scored
            "We are finding best happend action"
            Happend_Action_Best_One = Directions.STOP
            '''
            In this loop we are trying to do :
                Compare value of the score variable with recorded best record
                If it is less than that
                swap them
                continue until depth=0
            '''   
            for action in actions:
                score = Information_About_Agent_MinValue(state.generateSuccessor(PACMAN, action), depth, 1)
                if score > Recorded_Best_Scored:
                    Recorded_Best_Scored = score
                    Happend_Action_Best_One = action
            if depth == 0:
                return Happend_Action_Best_One
            else:
                return Recorded_Best_Scored
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        def Information_About_Agent_MinValue(state, depth, ghost):
            if state.isLose():
                return state.getScore()
            Available_Ghost_Next_One = ghost + 1
            "Based on the ghost value we will initialize available next ghost"
            if ghost == state.getNumAgents() - 1:
                Available_Ghost_Next_One = PACMAN
                "based on the legal action function we will find out actions"
            actions = state.getLegalActions(ghost)
            "Same as before, the initial value of recorded scores would be initialized with infiniti"
            Recorded_Best_Scored = float("inf")
            score = Recorded_Best_Scored
            '''
            In this loop we are trying to do :
                Compare value of the score variable with recorded best record
                If it is less than that
                swap them
                continue until depth=0
            '''
            for action in actions:
                prob = 1.0/len(actions)
                if Available_Ghost_Next_One == PACMAN:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                        score += prob * score
                    else:
                        score = Information_About_Agent_MaxValue(state.generateSuccessor(ghost, action), depth + 1)
                        score += prob * score
                else:
                    score = Information_About_Agent_MinValue(state.generateSuccessor(ghost, action), depth, Available_Ghost_Next_One)
                    score += prob * score
            return score
        return Information_About_Agent_MaxValue(gameState, 0)
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    def Available_Nearest_Small_Food(cur_pos, food_pos):
        Distance_of_Food = []
        for food in food_pos:
            Distance_of_Food.append(util.manhattanDistance(food, cur_pos))
        return min(Distance_of_Food) if len(Distance_of_Food) > 0 else 1
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"


    def closest_ghost(cur_pos, ghosts):
        Distance_of_Food = []
        for food in ghosts:
            Distance_of_Food.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(Distance_of_Food) if len(Distance_of_Food) > 0 else 1
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"


    def Useful_Information_Ghost(cur_pos, ghost_states, radius, scores):
        Ghost_Number_Variable = 0
        for ghost in ghost_states:
            if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                scores -= 30
                Ghost_Number_Variable += 1
        return scores
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    def Useful_Information_Food(cur_pos, food_positions):
        Distance_of_Food = []
        for food in food_positions:
            Distance_of_Food.append(util.manhattanDistance(food, cur_pos))
        return sum(Distance_of_Food)
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    def Variable_Of_Food_Number(cur_pos, food):
        return len(food)
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    Position_of_Ghost = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    "Based on the available nearest small foods or dots"
    "we calculates current score"
    score = score * 2 if Available_Nearest_Small_Food(Position_of_Ghost, food) < closest_ghost(Position_of_Ghost, ghosts) + 3 else score
    "We call position of ghost and food to count new score"
    score -= .35 * Useful_Information_Food(Position_of_Ghost, food)
    return score

# Abbreviation
better = betterEvaluationFunction
"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
class ContestAgent(MultiAgentSearchAgent):
    """
    "I used this code from GitHub to calculate multi agent search agent"
      Your agent for the mini-contest
    """
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        PACMAN = 0
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        def maxi_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            Recorded_Best_Scored = float("-inf")
            score = Recorded_Best_Scored
            Happend_Action_Best_One = Directions.STOP
            for action in actions:
                score = expecti_agent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > Recorded_Best_Scored:
                    Recorded_Best_Scored = score
                    Happend_Action_Best_One = action
            if depth == 0:
                return Happend_Action_Best_One
            else:
                return Recorded_Best_Scored
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        def expecti_agent(state, depth, ghost, alpha, beta):
            if state.isLose():
                return state.getScore()
            Available_Ghost_Next_One = ghost + 1
            if ghost == state.getNumAgents() - 1:
                Available_Ghost_Next_One = PACMAN
            actions = state.getLegalActions(ghost)
            Recorded_Best_Scored = float("inf")
            score = Recorded_Best_Scored
            for action in actions:
                prob = .8
                if Available_Ghost_Next_One == PACMAN: 
                    if depth == 3:
                        score = contestEvaluationFunc(state.generateSuccessor(ghost, action))
                        score += prob * score
                    else:
                        score = Information_About_Agent_MaxValue(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                        score += prob * score
                else:
                    score = expecti_agent(state.generateSuccessor(ghost, action), depth, Available_Ghost_Next_One, alpha, beta)
                    score += (1-prob) * score
            return score
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        def Information_About_Agent_MaxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            Recorded_Best_Scored = float("-inf")
            score = Recorded_Best_Scored
            Happend_Action_Best_One = Directions.STOP
            for action in actions:
                score = Information_About_Agent_MinValue(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > Recorded_Best_Scored:
                    Recorded_Best_Scored = score
                    Happend_Action_Best_One = action
                alpha = max(alpha, Recorded_Best_Scored)
                if Recorded_Best_Scored > beta:
                    return Recorded_Best_Scored
            if depth == 0:
                return Happend_Action_Best_One
            else:
                return Recorded_Best_Scored
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        def Information_About_Agent_MinValue(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            Available_Ghost_Next_One = ghost + 1
            if ghost == state.getNumAgents() - 1:
                Available_Ghost_Next_One = PACMAN
            actions = state.getLegalActions(ghost)
            Recorded_Best_Scored = float("inf")
            score = Recorded_Best_Scored
            for action in actions:
                if Available_Ghost_Next_One == PACMAN:
                    if depth == 3:
                        score = contestEvaluationFunc(state.generateSuccessor(ghost, action))
                    else:
                        score = Information_About_Agent_MaxValue(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = Information_About_Agent_MinValue(state.generateSuccessor(ghost, action), depth, Available_Ghost_Next_One, alpha, beta)
                if score < Recorded_Best_Scored:
                    Recorded_Best_Scored = score
                beta = min(beta, Recorded_Best_Scored)
                if Recorded_Best_Scored < alpha:
                    return Recorded_Best_Scored
            return Recorded_Best_Scored
        return maxi_agent(gameState, 0, float("-inf"), float("inf"))
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
def contestEvaluationFunc(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def Available_Nearest_Small_Food(cur_pos, food_pos):
        Distance_of_Food = []
        for food in food_pos:
            Distance_of_Food.append(util.manhattanDistance(food, cur_pos))
        return min(Distance_of_Food) if len(Distance_of_Food) > 0 else 1
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def closest_ghost(cur_pos, ghosts):
        Distance_of_Food = []
        for food in ghosts:
            Distance_of_Food.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(Distance_of_Food) if len(Distance_of_Food) > 0 else 1

    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def Useful_Information_Ghost(cur_pos, ghost_states, radius, scores):
        Ghost_Number_Variable = 0
        for ghost in ghost_states:
            if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                scores -= 30
                Ghost_Number_Variable += 1
        return scores
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def Useful_Information_Food(cur_pos, food_positions):
        Distance_of_Food = []
        for food in food_positions:
            Distance_of_Food.append(util.manhattanDistance(food, cur_pos))
        return sum(Distance_of_Food)
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def Variable_Of_Food_Number(cur_pos, food):
        return len(food)
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def Available_Nearest_Capsules(cur_pos, caps_pos):
        The_Distance_Of_Capsule = []
        for caps in caps_pos:
            The_Distance_Of_Capsule.append(util.manhattanDistance(caps, cur_pos))
        return min(The_Distance_Of_Capsule) if len(The_Distance_Of_Capsule) > 0 else 9999999
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def Information_About_Ghost_Scared_white(ghost_states, cur_pos, scores):
        The_List_of_scores = []
        for ghost in ghost_states:
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 4:
                The_List_of_scores.append(scores + 50)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 3:
                The_List_of_scores.append(scores + 60)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 2:
                The_List_of_scores.append(scores + 70)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 1:
                The_List_of_scores.append(scores + 90)
        return max(The_List_of_scores) if len(The_List_of_scores) > 0 else scores
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def Variable_of_Attacks_Ghost(ghost_states, cur_pos, scores):
        The_List_of_scores = []
        for ghost in ghost_states:
            if ghost.scaredTimer == 0:
                The_List_of_scores.append(scores - util.manhattanDistance(ghost.getPosition(), cur_pos) - 10)
        return max(The_List_of_scores) if len(The_List_of_scores) > 0 else scores
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    def The_Agent_Score(cur_pos, food_pos, ghost_states, caps_pos, score):
        if Available_Nearest_Capsules(cur_pos, caps_pos) < closest_ghost(cur_pos, ghost_states):
            return score + 40
        if Available_Nearest_Small_Food(cur_pos, food_pos) < closest_ghost(cur_pos, ghost_states) + 3:
            return score + 20
        if Available_Nearest_Capsules(cur_pos, caps_pos) < Available_Nearest_Small_Food(cur_pos, food_pos) + 3:
            return score + 30
        else:
            return score
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "Calculating Position of capsules and updating state of ghost"
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    Position_of_Capsules = currentGameState.getCapsules()
    Position_of_Ghost = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "Calsulating final scores"
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"    
    score = The_Agent_Score(Position_of_Ghost, food, ghosts, Position_of_Capsules, score)
    score = Information_About_Ghost_Scared_white(ghosts, Position_of_Ghost, score)
    score = Variable_of_Attacks_Ghost(ghosts, Position_of_Ghost, score)
    score -= .35 * Useful_Information_Food(Position_of_Ghost, food)
    return score
"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
