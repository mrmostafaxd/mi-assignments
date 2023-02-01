def backtracking_search(problem: Problem, assignment: Assignment, domains: Dict[str, set]) -> Optional[Assignment]:
    # if assignment is complete
    if problem.is_complete(assignment):
        # return assignment
        return assignment
    # get variable to assign
    variable_to_assign = minimum_remaining_values(problem, domains)
    # get values to assign
    values_to_assign = least_restraining_values(problem, variable_to_assign, domains)
    # for each value in values_to_assign
    for value in values_to_assign:
        # create copy of assignment
        assignment_copy = assignment.copy()
        # create copy of domains
        domains_copy = domains.copy()
        # add variable to assignment with value as value
        assignment_copy[variable_to_assign] = value
        # if forward_checking(problem, variable_to_assign, value, domains_copy)
        if forward_checking(problem, variable_to_assign, value, domains_copy):
            # result = backtracking_search(problem, assignment_copy, domains_copy)
            result = backtracking_search(problem, assignment_copy, domains_copy)
            # if result is not None
            if result is not None:
                # return result
                return result
    # return None
    return None
def solve(problem: Problem) -> Optional[Assignment]:
    # if problem is not one consistent
    if not one_consistency(problem):
        # return None
        return None
    # create empty assignment
    assignment = {}
    # create empty domains
    domains = {}
    # for each variable in problem.variables
    for variable in problem.variables:
        # add variable to domains with domain as value
        domains[variable] = problem.domains[variable]
    # return backtracking_search(problem, assignment, domains)
    return backtracking_search(problem, assignment, domains)