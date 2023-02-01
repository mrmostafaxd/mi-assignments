from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented
import copy
from copy import deepcopy

# This function should apply 1-Consistency to the problem.
# In other words, it should modify the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints should be removed from the problem (they are no longer needed).
# The function should return False if any domain becomes empty. Otherwise, it should return True.
def one_consistency(problem: Problem) -> bool:
    #TODO: Write this function
    assignment = dict()
    # loop over each variable
    for variable in problem.variables:
        # get the variables domain (deep copy to avoid iterator error when length changes)
        domain = copy.deepcopy(problem.domains[variable])

        # loop over each constraint
        for constraint in problem.constraints:
            # check if the constraint is a unary constraint and
            #     the variable it should apply to is the current variable
            if isinstance(constraint, BinaryConstraint) or not constraint.variable == variable:
                continue

            # loop over each domain value 
            for domain_val in problem.domains[variable]:
                # check if the domain value breaks the current Unary constraint
                assignment[variable] = domain_val
                if domain_val in domain and not constraint.is_satisfied(assignment):
                    domain.remove(domain_val)

        # update domain values with the new domain
        problem.domains[variable] = domain

    # remove all unary constraints
    for constraint in problem.constraints.copy():
        if isinstance(constraint, UnaryConstraint):
            problem.constraints.remove(constraint)
    
    # return if the domain is empty
    for variable in problem.variables:
        if len(problem.domains[variable]) == 0:
            return False
    return True
                    

# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    #TODO: Write this function
    for constraint in problem.constraints:
        # check if the current binary constraint is related to the assigned variable
        variable1, variable2 = constraint.variables
        if not assigned_variable == variable1 and not assigned_variable == variable2:
            continue
        
        # get the other variable
        other_variable = constraint.get_other(assigned_variable)
        
        # check if the other variable has no domain
        if not other_variable in domains:
            continue

        # update the other variable's domain
        if assigned_value in domains[other_variable]:
            domains[other_variable].remove(assigned_value)
        
        # check if the other variable's domain became empty 
        if len(domains[other_variable]) == 0:
            return False
    
    return True


# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic, 
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    #TODO: Write this function
    
    # create a dictionary for counting the domain values for every possible assignment
    val_dict = dict()

    # initialize each value of the domain inside val_dict
    for value in domains[variable_to_assign]:
        val_dict[value] = 0
    
    for constraint in problem.constraints:
        # check if the current binary constraint is related to the assigned variable
        if variable_to_assign not in constraint.variables:
            continue
        
        # get the other variable
        other_variable = constraint.get_other(variable_to_assign)

        # check if the other variable has no domain
        if not other_variable in domains:
            continue
        
        # increase the number of occurences of value 
        #     if the value exists in the other variable's domain
        for value in domains[variable_to_assign]:
            if value in domains[other_variable]:
                val_dict[value] += 1
    
    # return the values of the dictionary 
    #     sorted by the number of occurences then by the value itself
    return [x for x,y in sorted(val_dict.items(), key=lambda x: (x[1],x[0]))]


# This function should return the variable that should be picked based on the MRV heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
# IMPORTANT: If multiple variables have the same priority given the MRV heuristic, 
#            order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    #TODO: Write this function

    # create a dictionary for counting the domain values for every possible assignment
    val_dict = dict()

    # store the domain size of each vairable (if domain exists) inside val_dict
    for variable in problem.variables:
        if variable in domains:
            val_dict[variable] = len(domains[variable])
    
    # generate a sorted list of the variables
    val_list = [x for x,y in sorted(val_dict.items(), key=lambda x: (x[1],x[0]))]
    # return the variable of the Minimum remaining variables
    return val_list[0]


# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#            Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.
def solve(problem: Problem) -> Optional[Assignment]:

    # recursive internal solve
    def solve_internal(problem: Problem, variable_to_assign: str, domains: Dict[str, set], para_assignment: Assignment) -> Optional[Assignment]:
        # check the length of the domain of the variable to assign
        if len(domains[variable_to_assign]) == 0:
            return None
        
        # sort the domain of the variable to assign
        variable_domain = least_restraining_values(problem, variable_to_assign, domains)

        # delete the domain of the variable so that the MRV can pick the next variable
        del domains[variable_to_assign]

        for value in variable_domain:
            # assign the value to the variable to assign
            para_assignment[variable_to_assign] = value

            # check forward checking
            if not forward_checking(problem, variable_to_assign, value, domains):
                continue
            
            # check if the assignment complete and satisfies the constraints
            if problem.is_complete(para_assignment) and problem.satisfies_constraints(para_assignment):
                return para_assignment
            
            # get the next variable to assign
            next_variable_to_assign = minimum_remaining_values(problem, domains)
            
            # perform the recursion to the next variable to assign
            return solve_internal(problem, next_variable_to_assign, domains, para_assignment)

    
    # check 1 consistency
    if not one_consistency(problem):
        return None
    
    # create empty assignment for value assignments
    final_assignment = dict()

    # check if the assignment complete and satisfies the constraints
    if problem.is_complete(final_assignment) and problem.is_consistent(final_assignment):
        return final_assignment
    
    # get first variable to assign
    variable_to_assign = minimum_remaining_values(problem, problem.domains)
    
    # call the recursive function
    return solve_internal(problem, variable_to_assign, problem.domains, final_assignment)


