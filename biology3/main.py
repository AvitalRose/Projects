# install the packages if they are not installed

import random
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import NearestNeighbors

FIRST_EFFECT = 0.4


####CLASSES####
class Hexagon:
    def __init__(self, index, center, size):
        self.index = index
        self.center = center
        self.size = size
        self.corners = self.get_all_vertices()
        self.edges = self.find_edges()
        self.vector = None
        self.cities = []
        self.neighbours = {}
        self.average_social_economic_state = None

    def get_all_vertices(self):
        """
        Function to get all vertices of the hexagon
        :return:
        """
        corners_list = [self.find_hex_corner(corner) for corner in range(6)]
        return corners_list

    def find_hex_corner(self, i):
        """
        Function to calculate a specific edge, based on the distance and angle from center to corner
        :param i: number of the corner to find
        :return: Point(x,y) of edge
        """
        angle_deg = 60 * i - 30
        angle_rad = math.pi / 180 * angle_deg
        x_corner = self.center.x + self.size * math.cos(angle_rad)
        y_corner = self.center.y + self.size * math.sin(angle_rad)
        return Point(x_corner, y_corner)

    def find_edges(self):
        """
        Function to calculate the edges of the hexagon
        :return: all edges of the hexagon
        """
        edges = {}
        directions = ["EAST", "NORTH_EAST", "NORTH_WEST", "WEST", "SOUTH_WEST", "SOUTH_EAST"]
        for i in range(5):
            edges[directions[i]] = Line(self.corners[i], self.corners[i + 1])
        edges[directions[5]] = Line(self.corners[5], self.corners[0])
        return edges

    def find_center_distance_to_east_or_west(self):
        """
        FUnction to find the distance of the center to the east or west edge
        :return:
        """
        # the distance is the distance of a point from an edge
        x_east = self.edges["EAST"].point_a.x
        distance = abs(self.center.x - x_east)
        return distance

    def generate_neighbour_hexagon(self, index, direction, num=1):
        """
        Function to generate neighbour hexagon in a given direction
        :param direction: direction to place new hexagon
        :param num: distance in which to place hexagon (only for right or left), hexagon unit away
        :return: new hexagon
        """
        # for generating hexagon to the east or west, center of hexagon is twice the distance to east/west edge
        if direction == "EAST":
            new_center_x = self.center.x + 2 * self.find_center_distance_to_east_or_west() * num
            new_center_y = self.center.y
            return Hexagon(index, Point(new_center_x, new_center_y), self.size)
        elif direction == "WEST":
            new_center_x = self.center.x - 2 * self.find_center_distance_to_east_or_west() * num
            new_center_y = self.center.y
            return Hexagon(index, Point(new_center_x, new_center_y), self.size)
        elif direction == "NORTH_EAST":
            new_center_x = self.edges["EAST"].point_a.x
            new_center_y = self.center.y + 0.5 * self.edges["EAST"].length() + self.size
            return Hexagon(index, Point(new_center_x, new_center_y), self.size)
        elif direction == "NORTH_WEST":
            new_center_x = self.edges["WEST"].point_a.x
            new_center_y = self.center.y + 0.5 * self.edges["WEST"].length() + self.size
            return Hexagon(index, Point(new_center_x, new_center_y), self.size)
        elif direction == "SOUTH_WEST":
            new_center_x = self.edges["WEST"].point_a.x
            new_center_y = self.center.y - (0.5 * self.edges["WEST"].length() + self.size)
            return Hexagon(index, Point(new_center_x, new_center_y), self.size)
        elif direction == "SOUTH_EAST":
            new_center_x = self.edges["EAST"].point_a.x
            new_center_y = self.center.y - (0.5 * self.edges["EAST"].length() + self.size)
            return Hexagon(index, Point(new_center_x, new_center_y), self.size)

    def get_closed_sides(self):
        """
        Function used for plotting hexagon, calculates all x's and y's of hexagon vertices, resulting in a closed loop
        :return: list of x's and y's of vertices
        """
        x_points = [i.x for i in self.corners]
        x_points.append(self.corners[0].x)
        y_points = [i.y for i in self.corners]
        y_points.append(self.corners[0].y)
        return x_points, y_points

    def calculate_average_social_economic_state(self, voting_dictionary):
        """
        Function to calculate average social economic state of cities which are assigned to hexagon
        :param voting_dictionary: dictionary of all cities, to extract social economic state of city
        :return:
        """
        if self.cities:
            cities_sum = 0
            for city in self.cities:
                cities_sum += voting_dictionary[city][0]
            self.average_social_economic_state = cities_sum / len(self.cities)
        else:
            self.average_social_economic_state = 0

    def update_vector(self, ring, vector):
        """
        Function to update hexagon vector
        :param ring: how far is this hexagon from the originally updated hexagon, this affects the level of change
        :param vector:
        :return:
        """
        # update_effect = FIRST_EFFECT - ring * DECREASE_RATE
        # if update_effect > 0:
        #     self.vector = (1 - update_effect) * self.vector + update_effect * vector
        # error = np.linalg.norm(vector - self.vector)
        self.vector = self.vector + FIRST_EFFECT * (4 - ring) * (vector - self.vector)

    def find_first_ring(self, grid):
        """
        Find the first neighbours ring
        :param grid: hexagons dictionary
        :return:
        """
        self.neighbours[1] = []
        for key, value in grid.items():
            if 1 < self.find_neighbour_dist(value) < 2:
                self.neighbours[1].append(key)

    def find_second_ring(self, grid):
        """
        Find the second neighbours ring, by getting first neighbours of first neighbours
        :param grid: hexagons dictionary
        :return:
        """
        self.neighbours[2] = []
        for first_neighbour in self.neighbours[1]:
            self.neighbours[2].extend(grid[first_neighbour].neighbours[1])
            self.neighbours[2].remove(self.index)
        self.neighbours[2] = list(set(self.neighbours[2]))

    def find_third_ring(self, grid):
        """
        Find the third neighbours ring by getting first neighbours of second neighbours
        :param grid: hexagons dictionary
        :return:
        """
        self.neighbours[3] = []
        for second_neighbour in self.neighbours[2]:
            self.neighbours[3].extend(grid[second_neighbour].neighbours[1])
            if self.index in self.neighbours[3]:
                self.neighbours[3].remove(self.index)
        self.neighbours[3] = list(set(self.neighbours[3]))

    def find_neighbour_dist(self, other_hex):
        """
        Given another hexagon, find the distance to center
        :param other_hex: another hexagon
        :return:
        """
        distance = np.sqrt(np.square(self.center.x - other_hex.center.x) +
                           np.square(self.center.y - other_hex.center.y))
        return distance

    def print_hexagon(self):
        """
        Function to print hexagon
        :return:
        """
        for key, value in self.sides.items():
            print(key, "(", value.point_a.x, value.point_a.y, "),(", value.point_b.x, value.point_b.y, ")")


class Line:
    def __init__(self, point_a, point_b):
        """
        Line class is made up of two points, the line goes from one point to another
        :param point_a: Point
        :param point_b: Point
        """
        self.point_a = point_a
        self.point_b = point_b

    def length(self):
        """
        Function to find lines length
        :return: length
        """
        length = np.sqrt(np.square(self.point_a.x - self.point_b.x) + np.square(self.point_a.y - self.point_b.y))
        return length

    def line_equation(self):
        """
        Function to find the line equation of function
        :return:
        """
        if self.point_a.x == self.point_b.x:
            m = 0
            b = 0
        else:
            m = (self.point_b.y - self.point_a.y) / (self.point_b.x - self.point_a.x)
            b = self.point_b.y - m * self.point_b.x
        return m, b

    def print_line(self):
        """
        Function to print line
        :return:
        """
        print(f'({self.point_a.x},{self.point_a.y}), ({self.point_b.x},{self.point_b.y})')


class Point:
    def __init__(self, x, y):
        """
        Point class is made out of x and y coordinates
        :param x: x coordinate
        :param y: y coordinate
        """
        self.x = x
        self.y = y

    def distance_to_other_point(self, point_a):
        """
        Function to calculate distance to another point
        :param point_a:
        :return:
        """
        x_distance = self.x - point_a.x
        y_distance = self.y - point_a.y
        distance = np.sqrt(np.square(x_distance) + np.square(y_distance))
        return distance

    def print_point(self):
        """
        Function to print point
        :return:
        """
        print(f'({self.x},{self.y})')


####FUNCTIONS####
def read_file_into_dictionary(file_name, order=None):
    """
    Function to read file into dictionary
    :param file_name:
    :param order:
    :return:
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    elected_parties = [party for party in lines[0].split(',')]

    # order lines
    random.shuffle(elected_parties)

    # the voting dictionary has the form of city_name: [social_economic_state, list of city votes, hex (initially None)]
    voting_dictionary = {}
    for line in lines[1:]:
        line = line.split("\n")[0]
        # dropping the total votes per city, that why starting from 3
        voting_dictionary[line.split(",")[0]] = [int(line.split(",")[1]), [int(i) for i in line.split(",")[3:]],
                                                 [None, None], int(line.split(",")[2])]

    return elected_parties, voting_dictionary


def normalize_vectors(voting_dictionary):
    """
    Function to normalize voting vectors of each city / make percentage
    :param voting_dictionary:
    :return:
    """
    for key, value in voting_dictionary.items():
        # changing to numpy array
        voting_dictionary[key][1] = np.array(voting_dictionary[key][1])
        # normalize vector by min-max normalization
        # value[1] = (value[1] - value[1].min()) / (value[1].max() - value[1].min())
        # convert to percentage
        value[1] = value[1] / np.sum(value[1])
        # z core normalize
        # value[1] = (value[1] - np.mean(value[1])) / np.std(value[1])


def create_grid():
    """
    Function to create grid.
    The first cell of the first row is created, then all cells on that row are generated from that cell.
    Next, the first cell of second row is created from first cell of first row, and then the rest of the second row is
    generated using the first cell of the second row, and so on.
    The layout is: 5, 6, 7, 8, 9, 8, 7, 6, 5
    :return:
    """
    # row number 1 - length 5 - 0,1,2,3,4
    grid = {0: Hexagon(0, Point(0, 0), size=1)}
    for i in range(1, 5):
        grid[i] = grid[0].generate_neighbour_hexagon(i, "EAST", i)
    # row number 2 - length 6 -  5,6,7,8,9,10
    grid[5] = grid[0].generate_neighbour_hexagon(5, "SOUTH_WEST")
    for i in range(1, 6):
        grid[i + 5] = grid[5].generate_neighbour_hexagon(i + 5, "EAST", i)
    # row number 3 - length 7 - 11,12,13,14,15,16,17
    grid[11] = grid[5].generate_neighbour_hexagon(11, "SOUTH_WEST")
    for i in range(1, 7):
        grid[i + 11] = grid[11].generate_neighbour_hexagon(i + 11, "EAST", i)
    # row number 4 - length 8 - 18,19,20,21,22,23,24,25
    grid[18] = grid[11].generate_neighbour_hexagon(18, "SOUTH_WEST")
    for i in range(1, 8):
        grid[i + 18] = grid[18].generate_neighbour_hexagon(i + 18, "EAST", i)
    # row number 5 - length 9 - 26,27,28,29,30,31,32,33,34
    grid[26] = grid[18].generate_neighbour_hexagon(26, "SOUTH_WEST")
    for i in range(1, 9):
        grid[i + 26] = grid[26].generate_neighbour_hexagon(i + 26, "EAST", i)
    # row number 6 - length 8 - 35, 36, 37, 38, 39, 40, 41, 42
    grid[35] = grid[26].generate_neighbour_hexagon(35, "SOUTH_EAST")
    for i in range(1, 8):
        grid[i + 35] = grid[35].generate_neighbour_hexagon(i + 35, "EAST", i)
    # row number 7 - length 7 - 43, 44, 45, 46, 47, 48, 49
    grid[43] = grid[35].generate_neighbour_hexagon(43, "SOUTH_EAST")
    for i in range(1, 7):
        grid[i + 43] = grid[43].generate_neighbour_hexagon(i + 43, "EAST", i)
    # row number 8 - length 6 - 50, 51, 52, 53, 54, 55
    grid[50] = grid[43].generate_neighbour_hexagon(50, "SOUTH_EAST")
    for i in range(1, 6):
        grid[i + 50] = grid[50].generate_neighbour_hexagon(i + 50, "EAST", i)
    # row number 9 - length 5 - 56, 57, 58, 59, 60
    grid[56] = grid[50].generate_neighbour_hexagon(56, "SOUTH_EAST")
    for i in range(1, 5):
        grid[i + 56] = grid[56].generate_neighbour_hexagon(i + 56, "EAST", i)

    return grid


def print_grid(voting_dictionary, grid):
    """
    Function to print the grid
    :param voting_dictionary:
    :param grid:
    :return:
    """
    # define min and max value
    # calculate average social
    average_list = []
    for hexagon in grid.values():
        hexagon.calculate_average_social_economic_state(voting_dictionary)
        average_list.append(hexagon.average_social_economic_state)

    min_val = min(average_list)
    max_val = max(average_list)
    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val, clip=True)
    for key, hexagon in grid.items():
        x, y = hexagon.get_closed_sides()
        if len(hexagon.cities) == 0:  # no cities in hex
            # color = (255, 255, 255)  # print white
            plt.fill(x, y, color='lightgray')
        elif hexagon.average_social_economic_state <= 2:
            # plt.fill(x, y, color='r')
            plt.fill(x, y, color='lightcoral')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 3:
            # plt.fill(x, y, color='darkorange')
            plt.fill(x, y, color='moccasin')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 4:
            # plt.fill(x, y, color='y')
            plt.fill(x, y, color='gold')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 5:
            # plt.fill(x, y, color='g')
            plt.fill(x, y, color='yellowgreen')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 6:
            plt.fill(x, y, color='c')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 7:
            # plt.fill(x, y, color='b')
            plt.fill(x, y, color='turquoise')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 8:
            # plt.fill(x, y, color='purple')
            plt.fill(x, y, color='lightpink')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 9:
            plt.fill(x, y, color='m')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
        elif hexagon.average_social_economic_state <= 10:
            plt.fill(x, y, color='pink')
            plt.text(hexagon.center.x, hexagon.center.y, str(math.ceil(hexagon.average_social_economic_state)))
    plt.title("SOM GRID")
    plt.show()


def coloring_grid(voting_dictionary, grid):
    # define min and max value
    # calculate average social
    average_list = []
    for hexagon in grid.values():
        hexagon.calculate_average_social_economic_state(voting_dictionary)
        average_list.append(hexagon.average_social_economic_state)

    if len(hexagon.cities) == 0:  # no cities in hex
        color = (255, 255, 255)  # print white

    min_val = min(average_list)
    max_val = max(average_list)
    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

    for v in average_list:
        print(v, mapper.to_rgba(v))

    # really simple grayscale answer
    algebra_list = [(x - min_val) / (max_val - min_val) for x in average_list]
    # let's compare the mapper and the algebra
    mapper_list = [mapper.to_rgba(x)[0] for x in average_list]
    matplotlib.pyplot.plot(average_list, mapper_list, color='red', label='ScalarMappable')
    matplotlib.pyplot.plot(average_list, algebra_list, color='blue', label='Algebra')
    plt.show()


def find_neighbours(grid):
    """
    For hex in grid, iterate over all grid and find ur neighbors
    :param grid:
    :return:
    """
    # ring one
    for key, value in grid.items():
        grid[key].find_first_ring(grid)
    # ring two
    for key, value in grid.items():
        grid[key].find_second_ring(grid)
    # ring three
    for key, value in grid.items():
        grid[key].find_third_ring(grid)


def loss_function_a(voting_dict, grid):
    loss = 0
    for key, hexagon in grid.items():
        if hexagon.cities:
            for city in hexagon.cities:
                loss += np.linalg.norm(hexagon.vector - voting_dict[city][1])
    return loss


def loss_function_b(voting_dict, grid):
    distance = 0
    for city, value in voting_dict.items():
        # find distance between first best and second best
        center_a = grid[value[2][0]].center
        center_b = grid[value[2][1]].center
        distance += np.sqrt(np.square(center_a.x - center_b.x) + np.square(center_a.y - center_b.y))
    return distance


def train(voting_dictionary, grid, matrix):
    """
    Function to train the grid
    :param voting_dictionary: dictionary of all cities and there votes
    :param grid: game grid
    :param matrix: matrix which rows correspond to matching hexagon (row x -> index x)
    :return:
    """
    # restart city assignment
    for hexagon in grid.values():
        hexagon.cities = []

    # iterate over all cities
    for i, (city, votes) in enumerate(voting_dictionary.items()):
        # iterate over all vectors and find most matching one
        city_votes = votes[1]
        nbrs = NearestNeighbors(n_neighbors=2).fit(matrix)
        index = nbrs.kneighbors(np.atleast_2d(city_votes))[1][0, 0]
        index_second = nbrs.kneighbors(np.atleast_2d(city_votes))[1][0, 1]  # used for type b loss calculation

        # go to matching hexagon and update city list
        grid[index].cities.append(city)
        votes[2] = [index, index_second]

        # update vectors
        grid[index].update_vector(0, city_votes)
        for key, value in grid[index].neighbours.items():
            for neighbour in value:
                grid[neighbour].update_vector(key, city_votes)


def game_logic():
    """
    Main logic of the program
    :return:
    """
    # read file
    parties, voting_dict = read_file_into_dictionary(file_name="Elec_24.csv")

    # normalize
    normalize_vectors(voting_dictionary=voting_dict)

    # sorting options
    # sort by social economic state
    # voting_dict = {k: v for k, v in sorted(voting_dict.items(), key=lambda item: item[1][0])}
    # sort by amount of voters
    voting_dict = {k: v for k, v in sorted(voting_dict.items(), key=lambda item: item[1][3])}

    # create grid
    game_grid = create_grid()

    # find closest neighbours for all grid
    find_neighbours(game_grid)

    # make randomized matrix, each row corresponds to hexagon of same index
    som_matrix = np.random.rand(61, 13) / 20

    # give each hexagon a row vector from the matrix
    for i, (_key, _value) in enumerate(game_grid.items()):
        _value.vector = som_matrix[i]

    for i in range(5):
        # train
        train(voting_dictionary=voting_dict, grid=game_grid, matrix=som_matrix)
        l1 = [i[2][0] for i in voting_dict.values()]
        loss_a = loss_function_a(voting_dict, game_grid)
        loss_b = loss_function_b(voting_dict, game_grid)
        # print("loss a is: ", loss_a, "loss b is:", loss_b, "overall loss is: ", 0.7 * loss_a + 0.3 * loss_b)

    # calculate average social
    # count how many hexagons
    cnt_hex = 0
    for hexagon in game_grid.values():
        if hexagon.cities:
            cnt_hex += 1
        hexagon.calculate_average_social_economic_state(voting_dict)

    print("Cities locations:")
    for city in voting_dict.keys():
        hexagon_matched = voting_dict[city][2][0]
        print(f"city: {city}, x: {round(game_grid[hexagon_matched].center.x,2)}, y: {game_grid[hexagon_matched].center.y}")

    print_grid(voting_dictionary=voting_dict, grid=game_grid)


if __name__ == "__main__":
    game_logic()
