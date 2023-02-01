from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

#TODO: Import any modules you want to use
import math

# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state) 

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.
def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)
    
    terminal, values = game.is_terminal(state)
    if terminal: return values[agent], None

    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    value, _, action = max((heuristic(game, state, agent), -index, action) for index, (action , state) in enumerate(actions_states))
    return value, action

# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that 
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].
def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    #TODO: Write this function
    def max_value(state: S, max_depth: int):
        # check if it is the terminal state then return the value of player at that state
        is_terminal, player_values = game.is_terminal(state)
        if is_terminal:
            return player_values[game.get_turn(state)], None
        
        # if the tree has a depth limit, return the heuristic of that depth
        if max_depth == 0:
            return heuristic(game, state, 0), None
        
        # generate a list with all the states
        state_list  = [game.get_successor(state, action) for action in  game.get_actions(state)]
        
        # get the next player turn
        player = game.get_turn(state_list[0])

        max_val = -math.inf
        max_action = None

        # if the next player is player is my player then keep maximizing my player's gain
        if player == 0:
             # max(max_value)
            for s in state_list:
                val,action = max_value(s, max_depth-1)
                if val > max_val:
                    max_val = val
                    max_action = action
                
        # the next player is another player
        else:
            # max(min_value)
            for s in state_list:
                val,action = min_value(s, max_depth-1)
                if val > max_val:
                    max_val = val
                    max_action = action

        return max_val, max_action

    def min_value(state: S, max_depth: int):
        # get the current player
        player = game.get_turn(state)
        
        # check if it is the terminal state then return the value of player at that state
        is_terminal, player_values = game.is_terminal(state)
        if is_terminal:
            return player_values[player], None
        
        # if the tree has a depth limit, return the heuristic of that depth
        if max_depth == 0:
            return heuristic(game, state, 0), None
        
        # generate a list with all the actions
        state_list  = [game.get_successor(state, action) for action in  game.get_actions(state)]
        
        # get the next player turn
        player = game.get_turn(state_list[0])

        min_val = math.inf
        min_action = None
        # if the next player is player is my player then keep minimizing my player's gain
        #    min(max_value)
        if player == 0:
            # min(max_value)
            for s in state_list:
                val,action = max_value(s, max_depth-1)
                if val < min_val:
                    min_val = val
                    min_action = action
        
        # the next player is another player
        else:
            # min(min_value)
            for s in state_list:
                val,action = min_value(s, max_depth-1)
                if val < min_val:
                    min_val = val
                    min_action = action
    
        return min_val, min_action

    # generate a list with all the actions
    state_action_list  = [(game.get_successor(state, action), action) for action in  game.get_actions(state)]

    # check if it is the terminal state then return the value of player at that state
    is_terminal, player_values = game.is_terminal(state)
    if is_terminal:
        return player_values[game.get_turn(state)], None
    
    action = None
    value = -math.inf
    # check if the game has only one player
    if game.agent_count == 1:
        # max(max_value)
        for s,a in state_action_list:
            val,act = max_value(s, max_depth-1)
            if val > value:
                value = val
                action = a
    else:
        # max(min_value)
        for s,a in state_action_list:
            val,act = min_value(s, max_depth-1)
            if val > value:
                value = val
                action = a
    
    return value, action

# Apply Alpha Beta pruning and return the tree value and the best action
# Hint: Read the hint for minimax.
#  , alpha: float = -math.inf, beta: float = math.inf
def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    def alphabeta_internal(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1, alpha: float = -math.inf, beta: float = math.inf) -> Tuple[float, A]:
        # check if it is the terminal state then return the value of player at that state
        is_terminal, player_values = game.is_terminal(state)
        if is_terminal:
            return player_values[game.get_turn(state)], None
        
        # if the tree has a depth limit, return the heuristic of that depth
        if max_depth == 0:
            return heuristic(game, state, 0), None
        
        # generate a list with all the states and actions
        state_action_list  = [(game.get_successor(state, action), action) for action in  game.get_actions(state)]

        final_value = None
        final_action = None

        # if it is our player (maximize value):
        if game.get_turn(state) == 0:
            # we will maximize the value
            final_value = -math.inf

            for state_iter,action_iter in state_action_list:
                # call the function to get the resulted value and action
                current_value, current_action = alphabeta_internal(game, state_iter, heuristic, max_depth-1, alpha, beta)

                # check if the current value is maximum
                if current_value > final_value:
                    final_value = current_value
                    final_action = action_iter
                

                # check if the other player have a better move (beta < final value) (!!!!!!)
                if final_value >= beta:
                    return final_value, final_action
                
                # update alpha if the final value is bigger
                if final_value > alpha:
                    alpha = final_value
                
                # # check if the other player have a better move (beta < alpha)
                # if beta <= alpha:
                #     return final_value, final_action
        else:
            # we will minimize the value
            final_value = math.inf

            for state_iter,action_iter in state_action_list:
                # call the function to get the resulted value and action
                current_value, current_action = alphabeta_internal(game, state_iter, heuristic, max_depth-1, alpha, beta)

                # check if the current value is minimum
                if current_value < final_value:
                    final_value = current_value
                    final_action = action_iter
                
                # check if the other player have a better move (value < alpha)
                if final_value <= alpha:
                    return final_value, final_action

                # update beta if the current value is smalleer
                if final_value < beta:
                    beta = final_value
                
                # # check if the other player have a better move (beta < alpha)
                # if beta <= alpha:
                #     return final_value, final_action
        return final_value, final_action
    
    return alphabeta_internal(game, state, heuristic, max_depth)
            
            
    

# Apply Alpha Beta pruning with move ordering and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    #TODO: Write this function
    #NotImplemented()
    def alphabeta_internal(paragent, game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1, alpha: float = -math.inf, beta: float = math.inf, ) -> Tuple[float, A]:
        # check if it is the terminal state then return the value of player at that state
        is_terminal, player_values = game.is_terminal(state)
        if is_terminal:
            return player_values[paragent], None
        
        # if the tree has a depth limit, return the heuristic of that depth
        if max_depth == 0:
            return heuristic(game, state, paragent), None
        
        # generate a list with all the states and actions
        state_action_list  = [(game.get_successor(state, action), action) for action in  game.get_actions(state)]

        final_value = None
        final_action = None

        # if it is our player (maximize value):
        if game.get_turn(state) == 0:
            # we will maximize the value
            final_value = -math.inf
            
            # maximize my value
            state_action_list.sort(key = lambda x: heuristic(game, x[0], paragent), reverse=True)

            for state_iter,action_iter in state_action_list:
                # call the function to get the resulted value and action
                current_value, current_action = alphabeta_internal(paragent, game, state_iter, heuristic, max_depth-1, alpha, beta)

                # check if the current value is maximum
                if current_value > final_value:
                    final_value = current_value
                    final_action = action_iter

                # update alpha if the final value is bigger
                if final_value > alpha:
                    alpha = final_value
                
                # check if the other player have a better move (beta < alpha)
                if beta <= alpha:
                    return final_value, final_action
        else:
            # we will minimize the value
            final_value = math.inf

            state_action_list.sort(key = lambda x: heuristic(game, x[0], game.get_turn(state)), reverse=True)

            for state_iter,action_iter in state_action_list:
                # call the function to get the resulted value and action
                current_value, current_action = alphabeta_internal(paragent, game, state_iter, heuristic, max_depth-1, alpha, beta)

                # check if the current value is minimum
                if current_value < final_value:
                    final_value = current_value
                    final_action = action_iter
                
                # update beta if the current value is smalleer
                if final_value < beta:
                    beta = final_value
                
                # check if the other player have a better move (beta < alpha)
                if beta <= alpha:
                    return final_value, final_action
        return final_value, final_action
    
    return alphabeta_internal(game.get_turn(state), game, state, heuristic, max_depth)


# Apply Expectimax search and return the tree value and the best action
# Hint: Read the hint for minimax, but note that the monsters (turn > 0) do not act as min nodes anymore,
# they now act as chance nodes (they act randomly).
def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    #TODO: Write this function
    is_terminal, player_values = game.is_terminal(state)
    if is_terminal:
        # return the values of player 0 (the only real player in the game)
        return player_values[0], None
    
    # if the tree has a depth limit, return the heuristic of that depth
    if max_depth == 0:
        return heuristic(game, state, 0), None
    
    # generate a list with all the states and actions
    state_action_list  = [(game.get_successor(state, action), action) for action in  game.get_actions(state)]

    final_value = None
    final_action = None

    # if it is our player (maximize value):
    if game.get_turn(state) == 0:
        # we will maximize the value
        final_value = -math.inf

        for state_iter,action_iter in state_action_list:
            # call the function to get the resulted value and action
            current_value, current_action = expectimax(game, state_iter, heuristic, max_depth-1)

            # change final_value if current_value is maximum
            if current_value > final_value:
                final_value = current_value
                final_action = action_iter
            
    else:
        final_value = math.inf

        # will be used to sum all the node values and get their expectation
        total_sum = 0

        for state_iter,action_iter in state_action_list:
            # call the function to get the resulted value and action
            current_value, current_action = expectimax(game, state_iter, heuristic, max_depth-1)

            # add value to the total sum
            total_sum += current_value

            # change final_value if current_value is minimum
            if current_value < final_value:
                final_value = current_value
                final_action = action_iter
        
        # the value will be the expectation of this node
        final_value = total_sum / len(state_action_list)

    return final_value, final_action

    