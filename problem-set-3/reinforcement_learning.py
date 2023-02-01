from typing import Callable, DefaultDict, Dict, Generic, List, Optional, Union
from agents import Agent
from environment import Environment, S, A
from helpers.mt19937 import RandomGenerator
import math

import json
from collections import defaultdict

from helpers.utils import NotImplemented

# The base class for all Reinforcement Learning Agents required for this problem set
class RLAgent(Agent[S, A]):
    rng: RandomGenerator # A random number generator used for exploration
    actions: List[A] # A list of all actions that the environment accepts
    discount_factor: float # The discount factor "gamma"
    epsilon: float # The exploration probability for epsilon-greedy
    learning_rate: float # The learning rate "alpha"

    def __init__(self,
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5, 
            learning_rate: float = 0.01, 
            seed: Optional[int] = None) -> None:
        super().__init__()
        self.rng = RandomGenerator(seed) # initialize the random generator with a seed for reproducability
        self.actions = actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    # A virtual function that returns the Q-value for a specific state and action
    # This should be overriden by the derived RL agents
    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        return 0
    
    # Returns true if we should explore (rather than exploit)
    def should_explore(self) -> bool:
        return self.rng.float() < self.epsilon

    def act(self, env: Environment[S, A], observation: S, training: bool = False) -> A:
        actions = env.actions()
        if training and self.should_explore():
            #TODO: Return a random action whose index is "self.rng.int(0, len(actions)-1)"
            return actions[self.rng.int(0, len(actions)-1)]
        else:
            #TODO: return the action with the maximum q-value as calculated by "compute_q" above
            # if more than one action has the maximum q-value, return the one that appears first in the "actions" list

            # Initialize an empty list to store the computed Q-values and actions
            q_act = []

            # Iterate over the available actions
            for action in actions:
                # Compute the Q-value for the current action
                q_val = self.compute_q(env, observation, action)

                # Append the Q-value and action to the list
                q_act.append((q_val, action))

            # Sort the list in descending order based on the Q-values
            for i in range(len(q_act)):
                for j in range(i + 1, len(q_act)):
                    if q_act[i][0] < q_act[j][0]:
                        q_act[i], q_act[j] = q_act[j], q_act[i]
            
            # Return the action with the highest Q-value
            return q_act[0][1]


#############################
#######     SARSA      ######
#############################

# This is a class for a generic SARSA agent
class SARSALearningAgent(RLAgent[S, A]):
    Q: DefaultDict[S, DefaultDict[A, float]] # The table of the Q values
                                                 # The first key is the string representation of the state
                                                 # The second key is the string representation of the action
                                                 # The value is the Q-value of the given state and action
    
    def __init__(self, 
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5, 
            learning_rate: float = 0.01, 
            seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        self.Q = defaultdict(lambda:defaultdict(lambda:0)) # The default Q value is 0

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        return self.Q[state][action] # Return the Q-value of the given state and action
        # NOTE: we cast the state and the action to a string before querying the dictionaries
    
    # Update the value of Q(state, action) using this transition via the SARSA update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, next_action: Optional[A]):
        #TODO: Complete this function to update Q-table using the SARSA update rule
        # If next_action is None, then next_state is a terminal state in which case, we consider the Q-value of next_state to be 0
        
        # Get the current Q-value for the state and action
        q = self.compute_q(env, state, action)

        # Get the next Q-value for the next state and action, or 0 if the next action is None
        if next_action is not None:
            next_q = self.compute_q(env, next_state, next_action)
        else:
             next_q = 0
        
        # Compute the target Q-value
        q_target = reward + self.discount_factor * next_q

        # Update the Q-value for the state and action using the SARSA equation
        self.Q[state][action] += self.learning_rate * (q_target - q)

    # Save the Q-table to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            Q = {
                env.format_state(state) : {
                    env.format_action(action): value for action, value in state_q.items()
                } for state, state_q in self.Q.items()
            }
            json.dump(Q, f, indent=2, sort_keys=True)
    
    # load the Q-table from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            Q = json.load(f)
            self.Q = {
                env.parse_state(state) : {
                    env.parse_action(action): value for action, value in state_q.items()
                } for state, state_q in Q.items()
            }

#############################
#####   Q-Learning     ######
#############################

# This is a class for a generic Q-learning agent
class QLearningAgent(RLAgent[S, A]):
    Q: DefaultDict[str, DefaultDict[str, float]] # The table of the Q values
                                                 # The first key is the string representation of the state
                                                 # The second key is the string representation of the action
                                                 # The value is the Q-value of the given state and action
    
    def __init__(self, 
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5, 
            learning_rate: float = 0.01, 
            seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        self.Q = defaultdict(lambda:defaultdict(lambda:0)) # The default Q value is 0

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        return self.Q[state][action] # Return the Q-value of the given state and action
        # NOTE: we cast the state and the action to a string before querying the dictionaries
    
    # Given a state, compute and return the utility of the state using the function "compute_q"
    def compute_utility(self, env: Environment[S, A], state: S) -> float:
        #TODO: Complete this function.
        # Initialize a variable to store the maximum Q-value
        max_q = -1 * math.inf

        # Iterate over the available actions
        for action in env.actions():
            # Compute the Q-value for the current action
            q = self.compute_q(env, state, action)
            
            # Update the maximum Q-value if necessary
            if q > max_q:
                max_q = q

        # Return the maximum Q-value
        return max_q


    # Update the value of Q(state, action) using this transition via the Q-Learning update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, done: bool):
        #TODO: Complete this function to update Q-table using the Q-Learning update rule
        # If done is True, then next_state is a terminal state in which case, we consider the Q-value of next_state to be 0

        # Get the current Q-value for the state and action
        q = self.compute_q(env, state, action)

        # Compute the estimated return for the current state and action
        q_target = reward + self.discount_factor * self.compute_utility(env, next_state)
        
        # Update the Q-value for the current state and action using the Q-learning update rule
        self.Q[state][action] += self.learning_rate * (q_target - q)

    
    # Save the Q-table to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            Q = {
                env.format_state(state) : {
                    env.format_action(action): value for action, value in state_q.items()
                } for state, state_q in self.Q.items()
            }
            json.dump(Q, f, indent=2, sort_keys=True)
    
    # load the Q-table from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            Q = json.load(f)
            self.Q = {
                env.parse_state(state) : {
                    env.parse_action(action): value for action, value in state_q.items()
                } for state, state_q in Q.items()
            }

#########################################
#####   Approximate Q-Learning     ######
#########################################

# The type definition for a set of features representing a state
# The key is the feature name and the value is the feature value
Features = Dict[str, float]

# This class takes a state and returns the a set of features
class FeatureExtractor(Generic[S, A]):

    # Returns a list of feature names.
    # This will be used by the Approximate Q-Learning agent to initialize its weights dictionary.
    @property
    def feature_names(self) -> List[str]:
        return []
    
    # Given an enviroment and an observation (a state), return a set of features that represent the given state
    def extract_features(self, env: Environment[S, A], state: S) -> Features:
        return {}

# This is a class for a generic Q-learning agent
class ApproximateQLearningAgent(RLAgent[S, A]):
    weights: Dict[A, Features]    # The weights dictionary for this agent.
                                    # The first key is action and the second key is the feature name
                                    # The value is the weight 
    feature_extractor: FeatureExtractor[S, A]   # The feature extractor used to extract the features corresponding to a state

    def __init__(self, 
            feature_extractor: FeatureExtractor[S, A],
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5, 
            learning_rate: float = 0.01,
            seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        feature_names = feature_extractor.feature_names
        self.weights = {action:{feature: 0 for feature in feature_names} for action in actions} # we initialize the weights to 0
        self.feature_extractor = feature_extractor

    # Given the features of state and an action, compute and return the Q value
    def __compute_q_from_features(self, features: Dict[str, float], action: A) -> float:
        #TODO: Complete this function
        # NOTE: Remember to cast the action to string before quering self.weights

        #  Initialize a variable to store the sum of weighted feature values
        sum = 0

        # Iterate over the features and their values
        for feature_name, feature_value in features.items():
            # Get the weight for the current feature
            w = self.weights[action][feature_name]
            # Multiply the weight by the feature value and add the result to the total sum
            sum += w * feature_value

        # Return the total sum of weighted feature values (q value approximation)
        return sum
    
    # Given the features of a state, compute and return the utility of the state using the function "__compute_q_from_features"
    def __compute_utility_from_features(self, features: Dict[str, float]) -> float:
        #TODO: Complete this function
        
        # Initialize a variable to store the maximum Q-value
        max_q = -1 * math.inf

        # Iterate over the actions
        for action in self.actions:
            # Compute the Q-value for the current action using the features and weights
            q = self.__compute_q_from_features(features, action)
            # Update the maximum Q-value if necessary
            if q > max_q:
                max_q = q

        # Return the maximum Q-value
        return max_q

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        features = self.feature_extractor.extract_features(env, state)
        return self.__compute_q_from_features(features, action)

    # Update the value of Q(state, action) using this transition via the Q-Learning update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, done: bool):
        #TODO: Complete this function to update weights using the Q-Learning update rule
        # If done is True, then next_state is a terminal state in which case, we consider the Q-value of next_state to be 0

        # check if the episode is done
        if done:
            next_q_val = 0
        else:
            # compute the next Q-value using the features of the next state
            next_q_val = self.__compute_utility_from_features(self.feature_extractor.extract_features(env, next_state))
        
        # Compute the target Q-value using the reward and the next Q-value and the current Q-value
        q_target = reward + self.discount_factor * next_q_val - self.compute_q(env, state, action)
        
        # Iterate over the features and their values for the current state
        for it in self.feature_extractor.extract_features(env, state).items():

            # Update the weight for the current feature using the Q-learning update rule
            self.weights[action][it[0]] += self.learning_rate * q_target * it[1]
    
    # Save the weights to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            weights = {env.format_action(action): w for action, w in self.weights.items()}
            json.dump(weights, f, indent=2, sort_keys=True)
    
    # load the weights from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            weights = json.load(f)
            self.weights = {env.parse_action(action): w for action, w in weights.items()}