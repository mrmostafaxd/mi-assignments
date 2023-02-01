from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import numpy as np
import math

from helpers.utils import NotImplemented

# This is a class for a generic Policy Iteration agent
class PolicyIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training
    policy: Dict[S, A]
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        # This initial policy will contain the first available action for each state,
        # except for terminal states where the policy should return None.
        self.policy = {
            state: (None if self.mdp.is_terminal(state) else self.mdp.get_actions(state)[0])
            for state in self.mdp.get_states()
        }
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given the utilities for the current policy, compute the new policy
    def update_policy(self):
        # Iterate over the states in the MDP
        for state in self.mdp.get_states():
            # Get the actions available in the current state
            action_list = self.mdp.get_actions(state)

            # Initialize variables to store the maximum utility and corresponding action
            max_utility = -1 * math.inf
            optimal_action = None
            
            # Iterate over the actions
            for act in action_list:
                # Initialize a variable to store the utility for the current action
                act_utility = 0
                # Get the successors of the current state for the current action
                state_list = self.mdp.get_successor(state, act)
                # Iterate over the successors
                for s in state_list:
                    # Compute the contribution of the current successor to the utility of the current action
                    succ_cont = state_list[s] * (self.discount_factor * self.utilities[s] + self.mdp.get_reward(state, act, s))
                    # Add the contribution to the total utility for the current action
                    act_utility += succ_cont
                # Check if the current action has a higher utility than the previous best action
                if act_utility > max_utility:
                    # Update the maximum utility and the best action if necessary
                    max_utility = act_utility
                    optimal_action = act
            # Update the policy for the current state to the action with the highest utility
            self.policy[state] = optimal_action
    
    # Given the current policy, compute the utilities for this policy
    # Hint: you can use numpy to solve the linear equations. We recommend that you use numpy.linalg.lstsq
    def update_utilities(self):
        # Count the number of terminal states
        num_terminal_states = sum(self.mdp.is_terminal(state) for state in self.mdp.get_states())

        # Create a dictionary mapping states to their indices in the matrices
        state_indices = {}
        for state in self.mdp.get_states():
            if not self.mdp.is_terminal(state):
                state_indices[state] = len(state_indices)

        # Calculate the number of non-terminal states
        num_states = len(self.mdp.get_states()) - num_terminal_states

        # Create the matrices for the linear system
        coefficient_matrix = np.zeros((num_states, num_states))
        constant_vector = np.zeros((num_states, 1))

        # Populate the matrices
        for state in self.mdp.get_states():
            if not self.mdp.is_terminal(state):
                action = self.policy[state]
                next_states = self.mdp.get_successor(state, action)

                # Get the row index for this state
                row = state_indices[state]
                coefficient_matrix[row, row] = 1
                for successor in next_states:
                    constant_vector[row] += next_states[successor] * self.mdp.get_reward(state, action, successor)
                    if not self.mdp.is_terminal(successor):
                        coefficient_matrix[row, state_indices[successor]] -= next_states[successor] * self.discount_factor

        # Solve the linear system
        solution = np.linalg.lstsq(coefficient_matrix,constant_vector)

        x_iterator = 0
        # Iterate over the states in the MDP
        for i in range(len(self.mdp.get_states())):
            # Skip terminal states
            if not self.mdp.is_terminal(self.mdp.get_states()[i]): 
                # Update the utility of the current non-terminal state
                self.utilities[self.mdp.get_states()[i]] = solution[0][x_iterator, 0]
                x_iterator += 1
    
    # Applies a single utility update followed by a single policy update
    # then returns True if the policy has converged and False otherwise
    def update(self) -> bool:
        #TODO: Complete this function
        # Save a copy of the current policy
        curr_policy = self.policy.copy()

        # Update the utilities
        self.update_utilities()

        # Update the policy
        self.update_policy()

        # Check if the policy has changed
        return curr_policy == self.policy

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update():
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #TODO: Complete this function
        # if the current state is terminal, return None
        if self.mdp.is_terminal(state):
            return None

        # initialize max_utility to the current utility of the state
        max_utility = -1 * math.inf

        # initialize best_action to None
        optimal_action = None

        action_list = self.mdp.get_actions(state)

        # iterate through each action and find the action with the highest expected utility
        for action in action_list:
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
            
             # if the expected utility of the current action is greater than or equal to the max_utility, update max_utility and optimal_action
            if action_utility >= max_utility:
                max_utility = action_utility
                optimal_action = action

        # return the best action
        return optimal_action
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            policy = {
                self.mdp.format_state(state): (None if action is None else self.mdp.format_action(action)) 
                for state, action in self.policy.items()
            }
            json.dump({
                "utilities": utilities,
                "policy": policy
            }, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in data['utilities'].items()}
            self.policy = {
                self.mdp.parse_state(state): (None if action is None else self.mdp.parse_action(action)) 
                for state, action in data['policy'].items()
            }
