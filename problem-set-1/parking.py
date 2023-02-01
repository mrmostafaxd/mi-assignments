from typing import Any, Dict, Set, Tuple, List
from problem import Problem
from mathutils import Direction, Point
from helpers import utils

#TODO: (Optional) Instead of Any, you can define a type for the parking state
ParkingState = Tuple[Point]
# An action of the parking problem is a tuple containing an index 'i' and a direction 'd' where car 'i' should move in the direction 'd'.
ParkingAction = Tuple[int, Direction]

# This is the implementation of the parking problem
class ParkingProblem(Problem[ParkingState, ParkingAction]):
    passages: Set[Point]    # A set of points which indicate where a car can be (in other words, every position except walls).
    cars: Tuple[Point]      # A tuple of points where state[i] is the position of car 'i'. 
    slots: Dict[Point, int] # A dictionary which indicate the index of the parking slot (if it is 'i' then it is the lot of car 'i') for every position.
                            # if a position does not contain a parking slot, it will not be in this dictionary.
    width: int              # The width of the parking lot.
    height: int             # The height of the parking lot.

    # This function should return the initial state
    def get_initial_state(self) -> ParkingState:
        #TODO: ADD YOUR CODE HERE
        #utils.NotImplemented()

        # return the state of the cars
        return self.cars
    
    # This function should return True if the given state is a goal. Otherwise, it should return False.
    def is_goal(self, state: ParkingState) -> bool:
        #TODO: ADD YOUR CODE HERE
        #utils.NotImplemented()

        # check if all cars are in parking positions and each car is in its position
        for car_index in range(len(state)):
            parked_car = self.slots.get(state[car_index], None)
            if parked_car is None or not parked_car is car_index:
                return False
        
        return True

        

    
    # This function returns a list of all the possible actions that can be applied to the given state
    def get_actions(self, state: ParkingState) -> List[ParkingAction]:
        #TODO: ADD YOUR CODE HERE
        #utils.NotImplemented()

        # create unit points for addition and subtraction
        point_X = Point(1, 0)
        point_Y = Point(0, 1)

        # create a list for storing the possible actions for each car
        action_list = list()

        # loop on the car position points
        for car_index in range(len(state)):

            # check if moving RIGHT is a valid move
             #  and there is no car in that position
            new_s = state[car_index] + point_X
            if new_s in self.passages and not new_s in state:
                action_list.append((car_index, Direction.RIGHT))
            
            # check if moving LEFT is a valid move
             #  and there is no car in that position
            new_s = state[car_index] - point_X
            if new_s in self.passages and not new_s in state:
                action_list.append((car_index, Direction.LEFT))

            # check if moving DOWN is a valid move
            #  and there is no car in that position
            new_s = state[car_index] + point_Y
            if new_s in self.passages and not new_s in state:
                action_list.append((car_index, Direction.DOWN))

            # check if moving UP is a valid move
             #  and there is no car in that position
            new_s = state[car_index] - point_Y
            if new_s in self.passages and not new_s in state:
                action_list.append((car_index, Direction.UP))

        return action_list

            


    # This function returns a new state which is the result of applying the given action to the given state
    def get_successor(self, state: ParkingState, action: ParkingAction) -> ParkingState:
        #TODO: ADD YOUR CODE HERE

        # get an mutable copy from the state
        temp_state = list(state)

        # get the number of the car and the intended move from the action
        car_index = action[0]
        direction = action[1]
        
        # create a new point for storing the new state for the car
        carpoint = None

        # create unit points for addition and subtraction
        point_X = Point(1, 0)
        point_Y = Point(0, 1)

        # check if the moving direction is RIGHT
        if direction is Direction.RIGHT:
            carpoint = state[car_index] + point_X
        
        # check if the moving direction is LEFT 
        if direction is Direction.LEFT:
            carpoint = state[car_index] - point_X
        
        # check if the moving direction is DOWN
        if direction is Direction.DOWN:
            carpoint = state[car_index] + point_Y
        
        #check if the moving direction is UP
        if direction is Direction.UP:
            carpoint = state[car_index] - point_Y

        # store the new point in the new state tuple and return it
        temp_state[car_index] = carpoint
        return tuple(temp_state)

    
    # This function returns the cost of applying the given action to the given state
    def get_cost(self, state: ParkingState, action: ParkingAction) -> float:
        #TODO: ADD YOUR CODE HER

        # get the number of the car and the intended move from the action
        car_index = action[0] 
        direction = action[1]

        # create a new point for storing the new state for the car
        carpoint = None

        # create unit points for addition and subtraction
        point_X = Point(1, 0)
        point_Y = Point(0, 1)

        # check if the movement was RIGHT
        if direction is Direction.RIGHT:
            carpoint = state[car_index] + point_X
        
        # check if the movement was LEFT
        if direction is Direction.LEFT:
            carpoint = state[car_index] - point_X
        
        # check if the movement was DOWN
        if direction is Direction.DOWN:
            carpoint = state[car_index] + point_Y
        
        # check if the movement was UP
        if direction is Direction.UP:
            carpoint = state[car_index] - point_Y
        
        # check if the movement is in the car's parking space if it moved inside a parking slot
        if carpoint in self.slots and not self.slots[carpoint] is car_index:
            return 101
        else:
            return 1

     # Read a parking problem from text containing a grid of tiles
    @staticmethod
    def from_text(text: str) -> 'ParkingProblem':
        passages =  set()
        cars, slots = {}, {}
        lines = [line for line in (line.strip() for line in text.splitlines()) if line]
        width, height = max(len(line) for line in lines), len(lines)
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != "#":
                    passages.add(Point(x, y))
                    if char == '.':
                        pass
                    elif char in "ABCDEFGHIJ":
                        cars[ord(char) - ord('A')] = Point(x, y)
                    elif char in "0123456789":
                        slots[int(char)] = Point(x, y)
        problem = ParkingProblem()
        problem.passages = passages
        problem.cars = tuple(cars[i] for i in range(len(cars)))
        problem.slots = {position:index for index, position in slots.items()}
        problem.width = width
        problem.height = height
        return problem

    # Read a parking problem from file containing a grid of tiles
    @staticmethod
    def from_file(path: str) -> 'ParkingProblem':
        with open(path, 'r') as f:
            return ParkingProblem.from_text(f.read())
    
