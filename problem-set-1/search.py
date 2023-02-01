from problem import HeuristicFunction, Problem, S, A, Solution
from collections import deque
from helpers import utils
from heapq import heappush, heappop

#TODO: Import any modules you want to use

# All search functions take a problem and a state
# If it is an informed search function, it will also receive a heuristic function
# S and A are used for generic typing where S represents the state type and A represents the action type

# All the search functions should return one of two possible type:
# 1. A list of actions which represent the path from the initial state to the final state
# 2. None if there is no solution

def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #utils.NotImplemented()
    node = initial_state

    # check if the initial state is the goal
    # First In First Out queue for frontier state
    frontier = list()

    # add the initial state to the frontier
    frontier.append(node)
    
    # Set for the explored states
    explored = set()

    # Add the initial states to the list of the explored
    explored.add(node)
    
    # create a dictionary for saving parent and child pairs for each child for tracing
    #  the path to the solution
    parent_list = dict()

    # do loop until all the frontier is explored
    while frontier:
        # choose the shallowest node in the list
        node = frontier.pop(0)

        # check if the current state is the goal
        if problem.is_goal(node):

            # create a list to store the solution path
            sol_path = list()
            
            # loop on the dictionary to fill the solution path
            while node is not initial_state:
                node, action = parent_list[node]

                # to print the solution in Top Down style
                sol_path.insert(0, action)

            # return the path
            return sol_path
        

        # loop on the actions of the current state
        for action in problem.get_actions(node):
            # get a child
            child = problem.get_successor(node, action)

            # if the child is not explored, add it to the frontier and the explored set
            if child not in explored:
                # add child to the frontier and the explored node
                frontier.append(child)
                explored.add(child)

                # add the child and parent pair to parent_list dictionary
                #  to facilitate getting the solution path
                parent_list[child] = (node, action)

    return None


def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #utils.NotImplemented()
    node = initial_state
    
    # First In First Out queue for frontier nodes
    frontier = list()

    # add the initial state to the frontier
    frontier.append(node)
    
    # Set for the explored states
    explored = set()
    
    # create a dictionary for saving parent and child pairs for each child for tracing
    #  the path to the solution
    parent_list = dict()

    # loop on the actions of the current node state
    while frontier:

        # choose the shallowest node in the list
        node = frontier.pop()

        # skip if the state is explored before
        if node in explored:
            continue
        
        # Add the state to the list of the explored nodes
        explored.add(node)

        # check if the current node is the goal
        if problem.is_goal(node):
            # create a list to store the solution path
            sol_path = list()
            
            # loop on the dictionary to fill the solution path
            while node is not initial_state:
                node, action = parent_list[node]

                # to print the solution in Top Down style
                sol_path.insert(0, action)

            # return the path
            return sol_path
        
        # loop on the actions of the current state
        for action in problem.get_actions(node):
            # get a child
            child = problem.get_successor(node, action)

            # if the child is not explored, add it to the frontier and the explored set
            if child not in explored:
                # add child to the frontier
                frontier.append(child)

                # add the child and parent pair to parent_list dictionary
                #  to facilitate getting the solution path
                parent_list[child] = (node, action)
    
    return None


def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #utils.NotImplemented()
    
    # An increasing counter that will be put beside the cost to break the tie
    # in the arrangement of the nodes in the priority queue
    counter = 0

    node = initial_state

    # Priority Queue for the frontier states
    frontier = list()

    # add the initial state to the frontier priority queue
    heappush(frontier, (0.0,counter, node))
    counter += 1
    
    # Set for the explored states
    explored = set()

    # Add the initial states to the list of the explored
    explored.add(node)
    
    # create a dictionary for saving parent and child pairs for each child for tracing
    #  the path to the solution
    parent_list = dict()

    # create a dictionary for storing the action costs so that 
    #  the children could access it
    cost_list = dict()
    cost_list[node] = 0

    # do loop until all the frontier is explored
    while frontier:
        # choose the node with the lowest cost
        node = heappop(frontier)[2]

        # check if the current state is the goal
        if problem.is_goal(node):

            # create a list to store the solution path
            sol_path = list()
            
            # loop on the dictionary to fill the solution path
            while node is not initial_state:
                node, action = parent_list[node]

                # to print the solution in Top Down style
                sol_path.insert(0, action)

            # return the path
            return sol_path
        
        # add the current state to the explored set
        explored.add(node)
        

        # loop on the possible actions of the current state
        for action in problem.get_actions(node):
            # get a child
            child = problem.get_successor(node, action)

            # check if the child state is not in the frontier
            is_frontier = False
            for front in frontier:
                if child is front[2]:
                    is_frontier = True
                    break

             # if the child is not explored 
            if child not in explored:
                # if current path to the child is better than the old path
                if is_frontier:
                    if problem.get_cost(node, action) + cost_list[node] < cost_list[child]:
                        # get the child index in the frontier
                        j = None
                        for i in range(len(frontier)):
                            if child == (frontier[i])[2]:
                                j = i
                                break
                        #update child state inside the frontier 
                        del frontier[j]
                        heappush(frontier, (problem.get_cost(node, action) + cost_list[node], counter,child))
                        counter += 1

                        # add the child and parent pair to parent_list dictionary
                        #  to facilitate getting the solution path
                        parent_list[child] = (node, action)
                else:
                    # calculate the path cost to the child and push it in the queue
                    cost_list[child] = cost_list[node] + problem.get_cost(node, action)
                    heappush(frontier, (cost_list[child], counter,child))
                    counter += 1

                    # add the child and parent pair to parent_list dictionary
                    #  to facilitate getting the solution path
                    parent_list[child] = (node, action)
                
                # add the state to the explored set
                explored.add(child)
                # here
                # here 2

    return None
    

def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #utils.NotImplemented()

    # An increasing counter that will be put beside the cost to break the tie
    # in the arrangement of the nodes in the priority queue
    counter = 0

    node = initial_state

    # Priority Queue for the frontier states
    frontier = list()

    # add the initial state to the frontier priority queue
    heappush(frontier, (0.0,counter, node))
    counter += 1

    # Set for the explored states
    explored = set()

    # Add the initial state to the explored set
    explored.add(node)

    # create a dictionary for saving parent and child pairs for each child for tracing
    #  the path to the solution
    parent_list = dict()

    # create a dictionary for storing the action costs so that 
    #  the children could access it
    cost_list = dict()
    cost_list[node] = 0

    # create a dictionary for storing the total cost = g(n) + h(n)
    total_cost_list = dict()
    total_cost_list[node] = heuristic(problem, node)

    # do loop until all the frontier is explored
    while frontier:
        # choose the node with the lowest cost
        node = heappop(frontier)[2]

        # check if the current state is the goal
        if problem.is_goal(node):

            # create a list to store the solution path
            sol_path = list()
            
            # loop on the dictionary to fill the solution path
            while node is not initial_state:
                node, action = parent_list[node]

                # to print the solution in Top Down style
                sol_path.insert(0, action)

            # return the path
            return sol_path
        
        # loop on the possible actions of the current state
        for action in problem.get_actions(node):
            # get a child
            child = problem.get_successor(node, action)
        
            # if the child is not explored and not in frontier
            is_frontier = False
            for front in frontier:
                if child is front[2]:
                    is_frontier = True
                    break
            if child not in explored and not is_frontier:
                # calculate the new path cost g(n) of the child
                cost = cost_list[node] + problem.get_cost(node, action)

                # if the new path cost g(n) is less than the old stored cost
                if cost_list.get(child, None) is None or cost_list[child] > cost:

                    # change the path cost and the total path cost of the child
                    cost_list[child] = cost
                    total_cost_list[child] = cost + heuristic(problem, child)
                    
                    # push the child inside the priority queue
                    heappush(frontier, (total_cost_list[child],counter, child))
                    counter += 1

                    # add the child and parent pair to parent_list dictionary
                    #  to facilitate getting the solution path
                    parent_list[child] = (node, action)
            elif is_frontier:
                # get the child index in the frontier and its total cost
                child_cost = None
                j = None
                for i in range(len(frontier)):
                    if child == (frontier[i])[2]:
                        child_cost = frontier[i][0]
                        j = i
                        break
                
                # check if the new total cost is less than the old total cost
                if (total_cost_list[child] < child_cost):
                    # remove the it from the frontier and push it inside
                    # the frontier with the new path cost
                    del frontier[j]
                    parent_list[child] = (node, action)
                    heappush(frontier, (cost_list[child] + heuristic(problem, child),counter, child))
                    counter += 1
            
            # here
            # add the child to the explored set
            explored.add(child)
    return None


def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #utils.NotImplemented()

    # An increasing counter that will be put beside the cost to break the tie
    # in the arrangement of the nodes in the priority queue
    counter = 0

    node = initial_state

    # Priority Queue for the frontier states
    frontier = list()

    # add the initial state to the frontier priority queue
    heappush(frontier, (heuristic(problem, node),counter, node))
    counter += 1

    # Set for the explored states
    explored = set()

    

    # create a dictionary for saving parent and child pairs for each child for tracing
    #  the path to the solution
    parent_list = dict()

    # do loop until all the frontier is explored
    while frontier:
        # choose the node with the lowest cost
        node = heappop(frontier)[2]

        # check if the current state is the goal
        if problem.is_goal(node):

            # create a list to store the solution path
            sol_path = list()
            
            # loop on the dictionary to fill the solution path
            while node is not initial_state:
                node, action = parent_list[node]

                # to print the solution in Top Down style
                sol_path.insert(0, action)

            # return the path
            return sol_path

        # Add the state to the explored set
        explored.add(node)

        # loop on the possible actions of the current state
        for action in problem.get_actions(node):
            # get a child
            child = problem.get_successor(node, action)

            # check if the child in the frontier 
            is_frontier = False
            for front in frontier:
                if child is front[2]:
                    is_frontier = True
                    break
            
            # if the child is not explored
            if child not in explored:
                # get the child index in the frontier
                j = None
                for i in range(len(frontier)):
                    if child == (frontier[i])[2]:
                        j = i
                        break

                # if the child in the frontier and the current path
                #  is than the stored path
                if is_frontier and heuristic(problem, child) < frontier[j][0]:
                    # update the child heuristic cost inside the frontier
                    del frontier[j]
                    parent_list[child] = (node, action)
                    heappush(frontier, (heuristic(problem, child),counter, child))
                    counter += 1
                else:
                    #push the child inside the frontier queue
                    parent_list[child] = (node, action)
                    heappush(frontier, (heuristic(problem, child),counter, child))
                    counter += 1
            # add the child to the explored set
            explored.add(child)
    
    return None
