from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1a: Solve the segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 1b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE


def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 1c: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE


def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 2a: Solve the maze search problem with uniform cost search

class MazeProblem(util.SearchProblem):
    def __init__(self, start: tuple, goal: tuple, moveCost: Callable[[tuple, tuple], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
        self.start = start
        self.goal = goal
        self.moveCost = moveCost
        self.possibleMoves = possibleMoves

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE
            

def UCSMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, tuple], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves))
    
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the maze search problem with A* search

def consistentHeuristic(goal: tuple):
    def _consistentHeuristic(state: tuple) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE
    return _consistentHeuristic

def AStarMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, tuple], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves), heuristic=consistentHeuristic(goal))
    
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_CODE

############################################################


if __name__ == '__main__':
    shell.main()
