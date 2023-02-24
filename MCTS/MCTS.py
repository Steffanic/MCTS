
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
import math


class Node(ABC):
    @property
    @abstractmethod
    def IS_TWO_PLAYER():
        '''Returns True if the game is two player, False otherwise'''
        return False
    
    '''Base class for nodes in the search tree. Should be subclassed for specific games. Represents the current state'''
    @abstractmethod
    def get_children(self):
        '''Returns a set of all child states of the current state by taking all actions available from the current state'''
        return set()

    @abstractmethod
    def get_random_child(self):
        '''Returns a random child state of the current state. Useful for efficient roll-outs'''
        return None

    @abstractmethod
    def is_terminal(self):
        '''Returns True if the current state is a terminal state'''
        return False

    @abstractmethod
    def get_reward(self):
        '''Returns the reward of the current state. Should only be called if the current state is a terminal state.'''
        return 0

    @abstractmethod
    def __hash__(self):
        '''Nodes must be hashable'''
        return 1234565789

    @abstractmethod
    def __eq__(self, other):
        '''Nodes must be comparable'''
        return True

class MCTS:
    '''Monte-Carlo tree search implementation.'''
    def __init__(self, exploration_weight=1):
        self.total_rewards = defaultdict(int) #This will store the total reward at each node, tbh, Q is not a great name
        self.number_of_visits = defaultdict(int) # This will store the number of visits to each node
        self.children = dict() # This stores the children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node:Node):
        ''' Choose the best action available at the current node. PUCT, UCT, MCTS choice criterion'''
        if node.is_terminal():
            raise RuntimeError("The game is over, no further actions can be taken.")
        if node not in self.children:
            return node.get_random_child()
        
        def score(n:Node):
            '''
            This is the most basic implementation of MCTS that just exploits and doesn't balance exploitation and exploration(e.g. UCT).
            score = $\frac{total_rewards[n]}{number_of_visits[n]}$
            '''
            if self.number_of_visits[n] == 0:
                return float("-inf") # unseen state
            return self.total_rewards[n]/self.number_of_visits[n]
        
        return max(self.children[node], key=score)
    
    def do_rollout(self, node:Node):
        '''MCTS rollout algorithm'''
        path = self._select(node) # first select an unexplored node
        unexplored_node = path[-1]
        self._expand(unexplored_node)
        reward = self._simulate(unexplored_node)
        self._backpropagate(path, reward)

    def _select(self, node:Node):
        '''Find an unexplored child of node'''
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path # node will be in self.children if it has been explored, and if self.children[node] is empty, not self.children[node] is True
            unexplored = self.children[node] - self.children.keys() # children available to explore minus the children that have already been explored
            if unexplored:
                next_node = unexplored.pop()
                path.append(next_node)
                return path
            node = self._uct_select(node)
        
    def _expand(self, node:Node):
        '''Add node to the self.children dictionary'''
        if node in self.children:
            return
        self.children[node] = node.get_children()

    def _simulate(self, node:Node):
        '''Returns the reward for a random simulation of the game from node'''
        reward = 0 # we are always simulating from a reset state
        while True:
            reward += node.get_reward()
            if node.is_terminal():
                return reward
            node = node.get_random_child()

    def _backpropagate(self, path, reward):
        '''Propagates reward backwards through ancestor nodes'''
        for node in reversed(path):
            self.number_of_visits[node] +=1
            self.total_rewards[node] += reward
            if node.IS_TWO_PLAYER:
                reward = 1-reward

    def _uct_select(self, node:Node):
        # Every child of this node must be expanded
        assert all(n in self.children for n in self.children[node])

        log_number_of_visits_vertex = math.log(self.number_of_visits[node])

        def uct(n:Node):
            return self.total_rewards[n]/self.number_of_visits[n] + self.exploration_weight * math.sqrt(log_number_of_visits_vertex/self.number_of_visits[n])
        
        return max(self.children[node], key=uct)
