from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import math

from helpers.utils import NotImplemented

# This is a class for a generic Value Iteration agent
class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training 
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given a state, compute its utility using the bellman equation
    # if the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        #TODO: Complete this function
        max_utility = -1 * math.inf

       # if the current state is terminal, we cannot take any further actions, so return 0
        if self.mdp.is_terminal(state):
            return 0

        # if the current state is not terminal, we need to find the action with the highest expected utility
        # get the list of available actions in the current state
        actions_list = self.mdp.get_actions(state)

        # iterate through each action and compute the expected utility
        for act in actions_list:
            # get all possible next states and their probabilities for the current action
            states_list = self.mdp.get_successor(state, act)
            
            # initialize act_utility to 0
            act_utility = 0
            # iterate through each next state and its corresponding probability
            for it in states_list.items():
                reward = self.mdp.get_reward(state, act, it[0])
                next_util = self.utilities[it[0]]
                # add the probability * (reward + discounted utility of next state) to act_utility
                act_utility += it[1] * (reward + self.discount_factor * next_util)
            # update the maximum utility if this action has higher expected utility
            if act_utility > max_utility:
                max_utility = act_utility
        
        # return the maximum utility of taking any action from the current state
        return max_utility
    
    # Applies a single utility update
    # then returns True if the utilities has converged (the maximum utility change is less or equal the tolerance)
    # and False otherwise
    def update(self, tolerance: float = 0) -> bool:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #TODO: Complete this function

        # make a copy of the current utilities
        computed_utils = self.utilities.copy()

        # initialize maximum_utility_change to negative infinity
        maximum_utility_change = -1 * math.inf
        
        # update the utilities for each state
        for s in computed_utils:
            computed_utils[s] = self.compute_bellman(s)

        # calculate the difference between the old and new utilities for each state
        states_list = self.mdp.get_states()
        for s in states_list:
            # calculate the absolute difference between the old and new utilities for the current state
            utility_change = computed_utils[s] - self.utilities[s]

            if utility_change < 0:
                utility_change = - utility_change

            # update maximum_utility_change if the difference is larger than the current value
            maximum_utility_change = max(maximum_utility_change, utility_change)
            
        # update the utilities to the new values
        self.utilities = computed_utils
        
        # if the maximum difference is within the tolerance, return True
        return maximum_utility_change <=  tolerance

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update(tolerance):
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #TODO: Complete this function
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if the current state is terminal, return None
        if self.mdp.is_terminal(state):
            return None

        # initialize maximum probability to -inf
        max_pro = -1 * math.inf
        
        pro_list = list()

        actions_list = self.mdp.get_actions(state)
        # iterate through each action and find the action with the highest expected utility
        for action in actions_list:
            states_list = self.mdp.get_successor(state, action)
           
           # get all possible next states and their probabilities for the current action
            # initialize action_utility to 0
            action_utility = 0
            # iterate through each next state and probability
            for it in states_list.items():
                # get the reward for taking the current action in the current state and ending up in the next state
                reward = self.mdp.get_reward(state, action, it[0])

                # get the utility of the next state
                next_util = self.utilities[it[0]]

                # add the probability * (reward + discounted utility of next state) to action_utility
                action_utility += it[1] * (reward + self.discount_factor * next_util)
            
            pro_list.append(action_utility)

        # Iterate over the indices of the actions and their probabilities
        for i in range(len(pro_list)):
            # Get the probability for the current index
            pro = pro_list[i]

            # Update the maximum probability and index if necessary
            if pro > max_pro:
                max_pro = pro
                max_pro_index = i
        

        return actions_list[max_pro_index]
       
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in utilities.items()}
