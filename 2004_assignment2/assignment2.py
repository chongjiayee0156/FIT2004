from typing import List
from collections import deque
import math

"""
FIT2004 ALGORITHMS AND DATA STRUCTURE 
ASSIGNMENT 2 FINAL SUBMISSION
Student name: CHONG JIA YEE
Student ID: 33563888
Campus: MALAYSIA
Date: 21/10/2023
"""

"""
QUESTION 1:  Customized Auto-complete
EXPLANATION: FIND WORD OF THE SAME PREFIX WITH HIGHEST FREQUENCY AND RETURN ITS DEFINITION, KEEP COUNT OF TOTAL WORDS 
WITH SAME PREFIX
GENRE: TRIE
"""


### DO NOT CHANGE THIS FUNCTION
def load_dictionary(filename):
    infile = open(filename)
    word, frequency = "", 0
    aList = []
    for line in infile:
        line.strip()
        if line[0:4] == "word":
            line = line.replace("word: ", "")
            line = line.strip()
            word = line
        elif line[0:4] == "freq":
            line = line.replace("frequency: ", "")
            frequency = int(line)
        elif line[0:4] == "defi":
            index = len(aList)
            line = line.replace("definition: ", "")
            definition = line.replace("\n", "")
            aList.append([word, definition, frequency])

    return aList


class Node:
    """
    Node class for Trie
    each node can store 27 characters (a-z and $)
    each node will keep track of prefix count
    each node will keep track of max frequency of its children

    if the node is a terminal node
        it will keep track of the word's index in the dictionary
        it will keep track of the word's own frequency
    """

    def __init__(self, size=27):
        """
        Function Description:
            - Initialize a node with a link to 27 other nodes, prefix count, max frequency, index in dictionary and
            own frequency
        """
        self.link = [None] * size
        self.prefix_count = 0
        self.max_freq = -1
        self.index_in_dict = -1
        self.own_frequency = -1


class Trie:

    def __init__(self, Dictionary):
        """
        Function Description:
            - Initialize the trie with a root node and insert all words in dictionary to trie

        Precondition:
            - Dictionary is a list of list of strings
            - Each list of strings in Dictionary is in the format [word, definition, frequency]
            - word is a string of characters from a-z and $
            - definition is a string of characters
            - frequency is an integer

        Input:
            - Dictionary: A list of list of strings representing the dictionary

        Output:
            - None

        Postcondition:
            - The trie is initialized with a root node.
            - All words in dictionary are inserted into the trie

        Time Complexity:
            - best & worst: O(T) where T is the number of character in dictionary
                - copying dictionary to a class list: O(T) where T is the number of characters in dictionary
                - creating a trie for each word: O(NX) where N is the number of lists in dictionary and X is the length
                of the longest word
                - However, O(NX) is dominated by O(T) where T is the number of characters in dictionary
                - Hence, time complexity for this function is O(T) where T is the number of characters in dictionary


        Space Complexity:
            Input:
                - Dictionary: O(T) where T is the number of characters in dictionary
            Auxiliary:
                - self.dictionary: O(T) where T is the number of characters in dictionary
                - creating a trie for each word: O(NX) where N is the number of lists in dictionary and X is the length
                of the longest word
                - However, O(NX) is dominated by O(T) where T is the number of characters in dictionary
                - Hence, space complexity for auxiliary is O(T) where T is the number of characters in dictionary
            Total:
                - O(T) where T is the number of characters in dictionary
        """
        # create root node
        self.root = Node()

        # time & space comp = O(T) where T is the number of characters in dictionary
        # copy dictionary to a list
        self.dictionary = Dictionary

        # iterate through dictionary
        # O(N) where N is the number of lists in dictionary
        # insert all words in dictionary to trie
        for index_in_dict, (word, definition, frequency) in enumerate(Dictionary):
            # O(X) where X is the length of the word
            self.insert_recur(word, index_in_dict, frequency)

        # total time and space complexity for this loop:  O(NX) where N is the number of lists in dictionary and X is
        # the length of the longest word

    def insert_recur(self, word, index_in_dict, frequency):
        """
        format input to be used in insert_recur_aux
        """
        # pass in root node as current
        current = self.root
        # initialize i as first index and j as last index of the word
        i = 0
        j = len(word) - 1
        # call recursive function to insert word into trie with its own frequency and index
        self.insert_recur_aux(current, word, i, j, index_in_dict, frequency)

    def insert_recur_aux(self, current, word, i, j, index_in_dict, frequency):
        """
    Function Description:
      - Insert a word into the trie using recursive manner

    Precondition:
        -The trie is initialized with a root node.
        -The word is a string of characters from a-z and $.

    Input:
        -current: A node object representing the current node in the trie
        -word: A string representing the word to be inserted into the trie
        -i: An integer representing the starting index of the word
        -j: An integer representing the ending index of the word
        -index_in_dict: An integer representing the index of the word in the dictionary
        -frequency: An integer representing the frequency of the word in the dictionary

    Time Complexity:
    Best Case: O(1)
        - If the word is length 1 and is already in the trie, the function will only traverse 2 levels down the trie
    Worst Case: O(X) where X is the length of the input word
        - If the word is not in the trie, the function will traverse X levels down the trie
    Space Complexity:
    Input:
        -word: O(X) where X is the length of the input word

    Aux space complexity:
        - O(X) where X is the length of the input word, since the function is recursive,
        the function will be called X times, and each time the function is called, a new stack frame is created
        - other than that, in worst case where the entire prefix has not been included in Trie, X nodes will be created

    Total:
        - O(X) where X is the length of the input word
        """

        # increase prefix count of current node by 1 representing a new word with same prefix
        current.prefix_count += 1
        # update max frequency of current node
        current.max_freq = max(current.max_freq, frequency)

        # base case when i>j
        if i > j:
            # means current is last character of the word
            # move to $ node
            # if $ node does not exist, create a new node
            if current.link[0] is None:
                current.link[0] = Node()
                current = current.link[0]
                # store the word index in dictionary in the $ node
                current.index_in_dict = index_in_dict
                # store the word own frequency in the $ node
                current.own_frequency = frequency
                current.prefix_count += 1
                current.max_freq = max(current.max_freq, frequency)
            else:
                # else, terminal exists, move to $ node
                current = current.link[0]
                current.prefix_count += 1
                current.own_frequency = frequency
                current.max_freq = max(current.max_freq, frequency)
                current.index_in_dict = index_in_dict
        else:
            # else, move to next character
            index = ord(word[i]) - ord('a') + 1
            # if next character node does not exist, create a new node
            if current.link[index] is None:
                current.link[index] = Node()
                current = current.link[index]
            else:
                current = current.link[index]

            # recursive call, move to next character by incrementing i by 1
            self.insert_recur_aux(current, word, i + 1, j, index_in_dict, frequency)

    def prefix_search(self, prefix):
        """
        Function Description:
            - Search for the word with the highest frequency with the given prefix as input

        Precondition:
            - The trie is initialized with a root node.
            - The prefix is a string of characters from a-z and $.

        Input:
            - prefix: A string representing the prefix to be searched in the trie

        Output:
            - A list containing three elements where the first element "word" is the prefix matched word with
            the highest frequency, "definition" is its definition from the dictionary and num_matches is the
            number of words in the dictionary that have the input prefix as their prefix.

        Postcondition:
            - Output is based on lexicographical order of words in dictionary if there are multiple words with similar
            highest frequency with the given prefix

        Time Complexity:
            Best Case: O(1)
                - If the word with highest frequency with the given prefix is the first word in the dictionary,
                with the length of 1, the function will only traverse 2 levels down the trie
            Worst Case: O(M+N) where M is the length of input prefix, N is the total number of characters in the word
                and its definition with the highest frequency
                - after traversing till the end of the prefix, it still needs to traverse down to the terminal node of
                word with highest frequency to get its index in dictionary
                - then, returning the word and its definition will take O(N) time

        Space Complexity:
            Input:
                - prefix: O(M) where M is the length of the input prefix
            Aux space complexity:
                - O(N) where N is the total number of characters in the word with the highest frequency and its
                definition
                -this is because in this case where every string length is different, returning word and its definition
                will take O(N) space
            Total:
                - O(M+N) where M is the length of the input prefix, N is the total number of characters in the word and
                definition

        """

        # first, traverse to the last character of prefix
        current = self.root

        for char in prefix:
            index = ord(char) - ord('a') + 1
            # if next character node does not exist in Trie, return [None, None, 0]
            if current.link[index] is None:
                return [None, None, 0]
            else:
                current = current.link[index]

        # after traversing to last character of prefix, current is now the last character of prefix
        # get prefix count in this node
        num_matches = current.prefix_count

        # get max frequency
        max_frequency = current.max_freq

        # traverse from current to $ node with the highest frequency
        while True:
            # if current.link[0] is not None, means current node has a terminal as child
            # means this root to current itself is a word
            # check if this word is the word with maximum frequency by confirming if
            # terminal.own_frequency == max_frequency of the last prefix character
            if current.link[0] is not None and current.link[0].own_frequency == max_frequency:
                current = current.link[0]
                # if yes, return [word, definition, num_matches]
                return [self.dictionary[current.index_in_dict][0], self.dictionary[current.index_in_dict][1],
                        num_matches]

            else:
                # else, if current.link[0] is None, means it does not have a terminal as child
                # traverse current.link from left to right
                for each_char in current.link:
                    # if element is not None
                    # check if its max frequency is the same as the max frequency of the last prefix character
                    # if it is, means this child is one of the links to words with the highest frequency
                    # make it as the new current to trace down the terminal of word with max frequency and
                    # repeat the entire thing again
                    if each_char is not None and each_char.max_freq == max_frequency:
                        current = each_char
                        break


# ------------------------------------------------------------- #

"""
QUESTION 2: A WEEKEND GETAWAY
EXPLANATION: ALLOCATE PEOPLE TO CARS USING MAXIMUM FLOW ALGORITHM WITH CERTAIN CONSTRAINTS
GENRE: CIRCULATION WITH DEMAND, LOWER BOUNDARY, FORD FULKERSON, ADJACENCY GRAPH, BACKTRACKING, BFS
CLASSES: Edge, Vertex, Graph, ResidualGraph
"""


class Graph:
    """
    Create a graph with a list of vertices and a list of edges
    """

    def __init__(self, number_of_vertices, total_people, total_cars):
        """
        Function Description:
            - Initialize a graph with a list of vertices
            - intialize all required class variables

        input:
            - number_of_vertices: an integer representing the number of vertices in the graph
            - total_people: an integer representing the total number of people
            - total_cars: an integer representing the total number of cars/destinations

        output:
            - None

        precondition:
            - number_of_vertices is calculated involving all required vertices for circulation with demand
            - total_people is calculated correctly based on input
            - total_cars is calculated correctly based on math.ceil(total_people/5)

        postcondition:
            - self.vertices is a list of vertices with length number_of_vertices
            - self.edges is a list of edges but the edges have yet to be created in this constructor
            - self.destinations_allocation is a list of lists with length total_cars/total_destinations,
            since cars and destinations have the same total amount
            - self.residual_graph is a ResidualGraph object with the same number of vertices as self.vertices
            - edges of residual graph have yet to be created in this constructor

        time complexity:
            - total: O(n) where n is the number of person
            - breakdown:
                - O(v) to create a list of vertices
                - O(n) to create a list of lists for self.destinations_allocation
                - total: O(v+n) where v is the number of vertices and n is the number of people
                - however, v can be expressed in terms of n
                - v = 1 + total_car + (total_car * 2) + total_person + 1
                - where total_car = math.ceil(total_person/5)
                - where total_person = n
                - therefore, v = 1 + math.ceil(n/5) + (math.ceil(n/5) * 2) + n + 1
                - v = O(n)
                - therefore, time complexity is O(v+n) = O(n+n) = O(n)

        space complexity:
            - total: O(n) where n is the number of person
            - breakdown:
                - O(v) to create a list of vertices
                - O(n) to create a list of lists for self.destinations_allocation
                - total: O(v+n) where v is the number of vertices and n is the number of people
                - however, v = O(n)
                - therefore, space complexity is O(v+n) = O(n+n) = O(n)
        """
        # initialize class variables
        self.residual_graph = None
        # keep track of sink vertex for ford fulkerson
        self.sink_index = None
        # keep track of source vertec for ford fulkerson
        self.source_index = None

        # create a list of vertices
        # o(v) where v is the number of vertices
        self.vertices: List[Vertex] = [None] * number_of_vertices
        # create a list for edges
        self.edges: List[Edge] = []
        for i in range(number_of_vertices):
            self.vertices[i] = Vertex(i)

        # keep track of number of people and cars/destinations
        self.people = total_people
        self.cars = total_cars

        # o(n) where n is the number of people
        # create a list of lists to store allocation function output
        self.destinations_allocation = [None] * total_cars
        for i in range(total_cars):
            self.destinations_allocation[i] = []

    def create_residual_graph(self, number_of_vertices):
        """
        Function Description:
            a function to create residual graph
            when all edges are being added to original graph, the residual graph will be created based on the edges
        input: an integer representing the number of vertices in the graph
        output: None

        time complexity: O(V+E) where V is the number of vertices and E is the number of edges
        space complexity: O(V+E) where V is the number of vertices and E is the number of edges
        """

        # create a residual graph with the same number of vertices as the original graph
        self.residual_graph = ResidualGraph(number_of_vertices)

        # make residual graph with the finalized edge capacity
        # keep a pointer to its original edge in original graph so that any update on residual graph during ford
        # fulkerson, the original graph will be updated as well
        for edge in self.edges:
            u = edge.u
            v = edge.v
            c = edge.c
            self.residual_graph.add_edge(u, v, c, orginal_graph_edge=edge)

    def __str__(self):
        """
        a function to print the graph by printing each of its vertex and edges
        """
        ret = ""
        for vertex in self.vertices:
            ret += "Vertex " + str(vertex) + "\n"
        return ret

    def remove_lower_bound(self):
        """
        Function Description:
            a function to remove lower bound from the graph and updating the demand of each vertex
        input: None
        output: None
        time complexity: O(E) where E is the number of edges
        space complexity: O(1)
        """
        # loop through each edge
        for edge in self.edges:
            # if edge has lower-bound, reduce lower-bound to 0
            if edge.lower_bound > 0:
                # reduce its capacity with lower-bound
                edge.c -= edge.lower_bound
                # minus demand of vertex with incoming edge with lower bound
                self.vertices[edge.v].demand -= edge.lower_bound
                # plus vertex with outgoing edge with lower bound
                self.vertices[edge.u].demand += edge.lower_bound
                edge.lower_bound = 0

    def remove_demand(self):
        """
        Function Description:
            a function to remove demand from the graph by introducing source and sink vertex

        input: None
        output: None

        time complexity: O(V) where V is the number of vertices
        space complexity: O(1)
        """

        # O(1)
        # create a new source and sink for ford fulkerson
        self.source_index = self.add_source()
        self.sink_index = self.add_sink()

        # O(V)
        # loop through each vertex
        for vertex in self.vertices:

            # if demand > 0, add edge from vertex to sink with capacity = demand
            # then demand is reduced to 0
            if vertex.demand > 0:
                self.add_edge(vertex.id, self.sink_index, vertex.demand, 0)
                vertex.demand = 0

            # if demand < 0, add edge from source to vertex with capacity = abs(demand)
            # then demand is updated to 0
            elif vertex.demand < 0:
                self.add_edge(self.source_index, vertex.id, -1 * vertex.demand, 0)
                vertex.demand = 0

    def add_edge(self, u, v, c, lowerbound):
        """
        Function Description:
            a function to add edge to the graph
        input:
            u: an integer representing the id of the vertex where the edge is coming from
            v: an integer representing the id of the vertex where the edge is going to
            c: an integer representing the capacity of the edge
            lowerbound: an integer representing the lowerbound of the edge
        output: None
        time complexity: O(1)
        space complexity: O(1)
        """
        edge = Edge(u, v, c, lowerbound)
        self.vertices[u].add_edge(edge)
        self.edges.append(edge)

    def add_source(self):
        """
        Function Description:
            a function to add source vertex to the graph
        input: None
        output: an integer representing the id of the source vertex
        time complexity: O(1)
        space complexity: O(1)
        """
        # get the length of total vertices indicating the index of the new vertex will be added
        source_index = len(self.vertices)
        # add the new vertex to the graph vertices list
        self.vertices.append(Vertex(source_index))
        # return the index of the new vertex
        return source_index

    def add_sink(self):
        """
        Function Description:
            a function to add sink vertex to the graph
        input: None
        output: an integer representing the id of the sink vertex
        time complexity: O(1)
        space complexity: O(1)
        """
        sink_index = len(self.vertices)
        self.vertices.append(Vertex(sink_index))
        return sink_index

    def find_bottleneck(self, path):
        """
        Function Description:
            a function to find the bottleneck of a path (maximum flow that can be pushed through the path)
        input:
            path: a list of edges representing the path
        output:
            bottleneck: an integer representing the bottleneck of the path
        time complexity:
            O(E) where E is the number of edges in the path
        space complexity:
            O(1)
        """
        bottleneck = float('inf')

        for edge in path:
            # find the available capacity left in the edge by decreasing capacity with flow
            # then, find the MINIMUM of all the available capacity left in the path by comparing the minimum
            # available capacity of each edge
            bottleneck = min(bottleneck, edge.c - edge.f)
        return bottleneck

    def ford_fulkerson(self, source, sink):
        """
        Function Description:
            a function to find the maximum flow of the graph using ford fulkerson algorithm
        input:
            source: an integer representing the id of the source vertex
            sink: an integer representing the id of the sink vertex
        output:
            max_flow: an integer representing the maximum flow of the graph
        time complexity:
            O(n^3) where n is the number of vertices
            breaking down:
                O(V+E) for BFS
                BFS is called at most max flow times
                O(F(V+E)) = O(FE) where F is the max flow of the graph
                in this graph, we know that max flow can only be up to number of person
                so, O(F) = O(n) where F is the number of person
                O(n(V+E)) = O(nE)
                worst case is when each person has preference to ceil(n/5) destinations. E = n*(n/5) + some other edges
                to other vertices > n^2/5 = o(n^2)
                so O(nE) = O(n^3)

        space complexity:
            O(V+E) where V is the number of vertices, E is the number of edges
        """

        # initialize flow to 0
        flow = 0

        # when running bfs and it returns true, indicating there exists an augmenting path from source to sink
        # this bfs will be run at most max flow times
        # since we know in this graph th emaximum possible value for max flow is n, we can run bfs n times

        # O(F) = O(n) where F is the max flow of the graph, n is the number of person
        while self.residual_graph.has_augmenting_path(source, sink):  # O(V + E)
            # get a list of path made up of edges from source to sink
            path_forward, path_backward = self.residual_graph.get_augmenting_path(sink, self)  # O(E)

            # find bottleneck from the path
            bottleneck = self.find_bottleneck(path_forward)  # O(E)

            # update flow
            flow += bottleneck

            # update residual graph and original graph
            self.residual_graph.augment_flow(path_forward, path_backward, bottleneck)  # O(E)

        # return max flow when there is no more augmenting path
        return flow

    def backtrack(self):
        """
        Function Description:
            a function to backtrack the flow of the graph to find the allocation of each person to a car
        input: None
        output: a list of list of integers representing the allocation of each person to a car
        time complexity:
            O(n) where n is the number of people
        space complexity:
            O(n) where n is the number of people
        """

        # we wanna find previous.previous of each person vertex
        # since we know our graph connects (cars -> sub car vertices -> person vertices)
        # we can just find the previous.previous of each person vertex to know which car that person vertex it is
        # connected to

        # O(n) where n is the number of people
        for i in range(self.people):
            # get the previous vertex of the person vertex
            sub_car_vertex = self.vertices[i + 1].previous

            # format sub car vertex to start from 0
            sub_car_vertex = sub_car_vertex - (self.people + 2 + self.cars)

            # now that our sub car vertex starts from 0, we can know which car this sub car vertex belongs to by
            # dividing by 2
            car_vertex = sub_car_vertex // 2

            # add the person vertex to the car
            self.destinations_allocation[car_vertex].append(i)

        return self.destinations_allocation


class ResidualGraph:

    def __init__(self, number_of_vertices):
        """
        Function Description:
            a function to initialize the residual graph
        input:
            number_of_vertices: an integer representing the number of vertices in the graph
        output: None
        time complexity: O(V) where V is the number of vertices in the graph
        space complexity: O(V) where V is the number of vertices in the graph
        """
        self.vertices: List[Vertex] = [None] * number_of_vertices
        for i in range(number_of_vertices):
            self.vertices[i] = Vertex(i)

    def reset(self):
        """
        Function Description:
            a function to reset the residual graph
        input: None
        output: None
        time complexity: O(V) where V is the number of vertices in the graph
        space complexity: O(1)
        """
        for vertex in self.vertices:
            vertex.reset()

    def augment_flow(self, path_forward, path_backward, bottleneck):
        """
        Function Description:
            a function to augment flow in the residual graph and original graph
        input:
            path_forward: a list of edges representing the path from source to sink
            path_backward: a list of edges representing the path from sink to source
            bottleneck: an integer representing the bottleneck of the path
        output: None
        time complexity: O(E) where E is the number of edges in the path
        space complexity: O(1)
        """

        # path forwards
        for edge in path_forward:

            # update residual graph
            # each forward edge along the path of residual graph will have its capacity reduced by the bottleneck
            edge.c -= bottleneck
            # update original graph
            # each forward edge along the path of original graph will have its flow increased by the bottleneck
            if edge.orginal_edge.u is edge.u:
                edge.orginal_edge.f += bottleneck
            else:
                # if this forward edge is a reverse edge from original graph,
                # we need to subtract the flow in original edge by the bottleneck
                edge.orginal_edge.f -= bottleneck

        # update residual graph
        # each backward edge along the path of residual graph will have its capacity increased by the bottleneck
        for edge in path_backward:
            edge.c += bottleneck

    def has_augmenting_path(self, source, target):
        """
        Function Description:
            a function to check if there exists an augmenting path from source to target
        input:
            source: an integer representing the source vertex
            target: an integer representing the target vertex
        output:
            a boolean indicating if there exists an augmenting path from source to target
        time complexity:
            O(V+E) where V is the number of vertices in the graph, E is the number of edges in the graph
        space complexity:
            aux: O(V) where V is the number of vertices in the graph
        """

        # reset all vertices to undiscovered
        # time complexity: o(v)
        self.reset()

        if source == target:
            return True

        # Use a deque for efficient pop from the left
        discovered = deque()
        # Add the first vertex to the deque
        discovered.append(source)
        self.vertices[source].mark_visited()
        self.vertices[source].mark_discovered()

        while discovered:
            # Pop from the left, mark visited, add to result
            # time complexity: o(1)
            vertex_u = discovered.popleft()
            self.vertices[vertex_u].mark_visited()

            # add all adjacent vertices to queue, mark as discovered
            for edge in self.vertices[vertex_u].edges:
                # dont add vertex to queue if edge has no residual capacity
                if edge.c == 0:
                    continue
                vertex_v = edge.v

                # dont add to queue if it is discovered (it may still be in queue,
                # doesn't matter anyhow as it is discovered)
                if not self.vertices[vertex_v].is_discovered():
                    discovered.append(vertex_v)
                    self.vertices[vertex_v].mark_discovered()
                    self.vertices[vertex_v].previous = vertex_u
                    # if one of the edges being visited link to target, tht means we have confirmed there exist a path
                    # from source to target, return true
                    if vertex_v == target:
                        return True

        # if there isn't anymore vertex which this source can link to, but we havent found a path to target,
        # return false
        return False

    def get_augmenting_path(self, target, original_graph):
        """
        Function Description:
            a function to get the augmenting path from source to target after running bfs
            backtrack from target to source using vertex.previous to get the path
        input:
            target: an integer representing the target vertex
            original_graph: the original graph
        output: None
        time complexity: O(E) where E is the number of edges in the path
        space complexity: O(E) where E is the number of edges in the path
        """
        current = self.vertices[target]
        path_forward = []
        path_backward = []

        while current.previous is not None:

            # this line is for backtracking when ford fulkerson is done
            # we need to find a way to keep track of the path we found from each bfs run
            # however, since bfs will mess up vertex.previous of each vertex everytime it runs,
            # we cant rely on vertex.previous of our residual graph as it will be messed up after each bfs run
            # hence, we store the vertex.previous for each vertex in our PATH FOUND in the original graph
            original_graph.vertices[current.id].previous = current.previous

            # get the previous vertex
            previous = self.vertices[current.previous]

            # get forward edges by searching edges of previous vertex
            for edge in previous.edges:
                if edge.v == current.id:
                    path_forward.append(edge)
                    break

            # get backward edges by searching edges of current vertex
            has_backward_edge = False
            for edge in current.edges:
                if edge.v == previous.id:
                    has_backward_edge = True
                    path_backward.append(edge)
                    break

            # if there is no backward edge, means no backward edge for original
            # edge has been added in the residual graph
            # hence, add a new edge with capacity 0
            # pass in the last element collected in path_forward as the original edge
            if not has_backward_edge:
                edge = Edge(current.id, previous.id, 0, orginal_graph_edge=path_forward[-1].orginal_edge)
                # add the edge to the residual graph
                self.vertices[current.id].add_edge(edge)
                # add the edge to the path_backward
                path_backward.append(edge)

            # update current to previous, repeat until we reach source
            current = self.vertices[current.previous]

        # reverse the path forward as we are backtracking from target to source
        path_forward.reverse()
        return path_forward, path_backward

    def __str__(self):
        """
        Function Description:
            a function to print the graph
        input:
            None
        output:
            a string representing the graph
        """
        ret = ""
        for vertex in self.vertices:
            ret += "Vertex " + str(vertex) + "\n"
        return ret

    def add_edge(self, u, v, c, orginal_graph_edge=None):
        edge = Edge(u, v, c, orginal_graph_edge=orginal_graph_edge)
        self.vertices[u].add_edge(edge)


class Vertex:
    """
    class to represent a vertex in a graph
    """

    def __init__(self, id):
        """
        Function Description:
            constructor for Vertex class
        input:
            id: an integer representing the id of the vertex
        """
        self.id = id
        self.edges = []
        # for traversal
        self.visited = False
        self.discovered = False
        # for backtracking
        self.previous = None
        self.demand = 0

    def reset(self):
        """
        Function Description:
            a function to reset the vertex to undiscovered, univisted and has no previous vertex
        """
        self.visited = False
        self.discovered = False
        self.previous = None

    def add_edge(self, edge):
        """
        Function Description:
            a function to add an edge to the vertex
        input:
            edge: an Edge object
        """
        self.edges.append(edge)

    def __str__(self):
        """
        Function Description:
            print the vertex with its edges and demand
        """
        ret = ""
        ret += str(self.id)
        ret += " demand: "
        ret += str(self.demand)
        ret += " edges: "
        for edge in self.edges:
            ret += str(edge) + ","
        return ret

    def is_visited(self):
        return self.visited

    def is_discovered(self):
        return self.discovered

    def mark_visited(self):
        self.visited = True

    def mark_discovered(self):
        self.discovered = True


class Edge:
    """
    class to represent an edge in a graph
    """

    def __init__(self, u, v, c, lower_bound=0, orginal_graph_edge=None):
        """
        Function Description:
            constructor for Edge class
        input:
            u: an integer representing the id of the source vertex
            v: an integer representing the id of the target vertex
            c: an integer representing the capacity of the edge
            lower_bound: an integer representing the lower bound of the edge
            orginal_graph_edge: an Edge object representing the edge in the original graph (if any)
        """
        self.u = u
        self.v = v
        self.c = c
        self.f = 0
        self.orginal_edge = orginal_graph_edge
        self.lower_bound = lower_bound

    def __str__(self):
        """
        Function Description:
            print the edge with its outgoing vertex, incoming vertex, capacity, flow and lower bound
        """
        ret = ""
        ret += str(self.u)
        ret += "->"
        ret += str(self.v)

        ret += " c:"
        ret += str(self.c)
        ret += " f:"
        ret += str(self.f)
        ret += " lower_bound:"
        ret += str(self.lower_bound)
        return ret


def allocate(preferences, licenses):
    """
    Graph explanation:
        - let n = number of people
        - First, I have a circulation with demand graph with lower bound, capacity, no demand
        - One vertex named start will connect to ceil(n/5) vertices named cars
            - each edge connecting start to car vertices has capacity 5, lower bound 2
            - this is because each car can carry up to 5 people, and we need at least 2 people to drive the car
        - Each car vertex will connect to 2 sub car vertices, 1 for drivers and 1 for non-drivers
            - each edge connecting car to sub car DRIVER vertices has capacity 5, lower bound 2
            - each edge connecting car to sub car NON-DRIVER vertices has capacity 3, lower bound 0
            - this is because each car can carry up to 5 people, and we need at least 2 people to drive the car
            - in order to have 2 person driving the car, we can only have maximum of 3 non-drivers in a car
        - Each sub car vertex will then connect to each person vertex based on their preference and license
            - Ex: if person 1 is a driver has preference 1, 3, then we will have 2 edges connecting person 1 to
                DRIVER vertex of car 1 and 3
            - each edge will have capacity 1, lower bound 0
            - this is because each person can only go to 1 destination, they can choose not to go to that destination
        - Each person vertex will then connect to the end vertex
            - each edge will have capacity 1, lower bound 1
            - this is because each person can only go to 1 and only 1 destination, they cannot have no destination or
            more than 1 destination
        - End vertex will connect to start vertex
            - this one edge will have capacity n, lower bound n
            - this is because we need to allocate all n people to 1 destination each, hence end vertex must have n flow
            to start vertex

    Function Description:
        a function to allocate cars to people based on their preferences and licenses
        I created a circulation with demand
        then, I removed the lower bound and demand to get a network flow with capacity only
        then, I used Ford Fulkerson to find the maximum flow
        if the maximum flow is equal to n, then we have a solution
        then, I used backtracking to find the allocation of cars to people

    input:
        preferences: a list of list of integers representing the preferences of each person
        licenses: a list of boolean representing whether each person has a driver license or not
    output:
        a list of list of integers representing the allocation of cars to people

    Time complexity:
        O(n^3) from Ford Fulkerson and also creating the graph, where n is the number of people

    Space complexity:
        O(V+E) from creating the graph, where V is the number of vertices and E is the number of edges
    """

    total_person = len(preferences)

    if total_person == 1 and len(licenses) == 1:
        # means there is only 1 person and he is a driver
        # a car need to have 2 drivers at minimum
        return None

    # no. of car = no. of destinations = ceil(len(person)/5)
    total_destination = math.ceil(total_person / 5)
    total_car = total_destination

    # format the input so that we have driver_preferences and passenger_preferences
    # store index of that person and their preferences
    driver_preferences: List[tuple(int, List[int])] = []
    passenger_preferences: List[tuple(int, List[int])] = []

    # O(n^2) time complexity
    for i in range(total_person):
        if i not in licenses:
            passenger_preferences.append((i, preferences[i]))
        else:
            driver_preferences.append((i, preferences[i]))

    # ------------------------------ create a circulation with demand ------------------------------------------------

    # total number of vertices = 1 start vertex, total number of cars, total number of cars`*2,
    # ,total number of person and 1 end vertex

    total_no_of_vertices = 1 + total_car + (total_car * 2) + total_person + 1

    # create a graph
    graph = Graph(total_no_of_vertices, total_person, total_car)

    # let end vertex.id = 0
    end_vertex_index = 0

    # let person vertices.id = 1 ~ total_person
    first_person_index = 1
    last_person_index = total_person

    # let start vertex.id = total_person+1
    start_vertex_index = total_person + 1

    # let car vertices.id = total_person+2 ~ total_person+2+total_car-1
    first_car_index = total_person + 2
    last_car_index = total_person + 2 + total_car - 1

    # let sub car vertices.id = last_car_index + 1 ~ last_car_index + 2*total_car
    first_sub_car_index = last_car_index + 1
    last_sub_car_index = last_car_index + 2 * total_car

    # start adding edges to the graph

    # loop through each person vertex, link them to end vertex with capacity 1 lower bound 1
    # person > end
    # o(n)
    for i in range(first_person_index, last_person_index + 1):
        # create edge from person vertex to end vertex with capacity 1, lower bound 1
        graph.add_edge(i, end_vertex_index, 1, 1)

    # add edge from end vertex to start vertex, lower bound = n, capacity = n
    # end>start
    graph.add_edge(end_vertex_index, start_vertex_index, total_person, total_person)

    # pointer to count the sub vertex
    # increases everytime a sub vertex is visited
    pointer_for_sub_vertex = 0

    # loop through each car
    # o(n/5)=o(n)
    for i in range(first_car_index, last_car_index + 1):

        # add edge from start vertex to each car vertices, lower bound = 2, capacity = 5
        # start>car, lower bound = 2, capacity = 5
        graph.add_edge(start_vertex_index, i, 5, 2)

        #   for each car, it will have 2 sub vertices, 1 for drivers and 1 for non-drivers
        #   each car will have an edge to 2 sub vertices
        # loop twice for each car to get to its sub vertices
        # o(2)
        for j in range(2):

            # if j == 0: 1st sub vertex is for drivers
            if j == 0:
                #  car > car's driver vertex, lower bound = 2, capacity = 5
                car_driver_vertex = first_sub_car_index + pointer_for_sub_vertex
                graph.add_edge(i, car_driver_vertex, 5, 2)

                # each sub-vertex will have an edge to each person vertex whom that person have interest in with
                # capacity of 1, lower boundary = 0
                # hence, if it is d_vertex, we will loop through (index, list of preference) from driver_preference,
                # if current car is in list of preference, means an edge w capacity = 1 will be created from the
                # d_vertex to that person vertex
                # o(n^2) time complexity
                # car's driver vertex > person vertex

                # loop through each driver's preference list
                for index, preference in driver_preferences:
                    # format current car index to start from 0
                    if i - first_car_index in preference:
                        # if current car is in driver's preference list, create an edge w capacity of 1, lower bound 0
                        # from car's driver vertex > person vertex
                        graph.add_edge(car_driver_vertex, index + 1, 1, 0)

                pointer_for_sub_vertex += 1

            # if j == 1: 2nd sub vertex is for non-drivers
            else:
                car_non_driver_vertex = first_sub_car_index + pointer_for_sub_vertex
                # car > car's non-driver vertex lower bound = 0, capacity = 3
                graph.add_edge(i, car_non_driver_vertex, 3, 0)

                # if it is non-driver vertex, we will repeat the same thing but
                # loop through non-drivers preference list instead
                for index, preference in passenger_preferences:
                    # if current car is in this person's preference list, create an edge w capacity of 1 , lower bound 0
                    if i - first_car_index in preference:
                        # from car's non-driver vertex > person vertex
                        graph.add_edge(car_non_driver_vertex, index + 1, 1, 0)

                pointer_for_sub_vertex += 1

    # and we're done creating the flow network
    # now, remove lower bound from all edges
    graph.remove_lower_bound()
    # now, we will remove the demand
    graph.remove_demand()

    # after getting a network flow with only capacity and our source and sink vertex,
    # we create a copy of it called residual graph
    # pass in total number of vertex, which is sink index + 1
    graph.create_residual_graph(graph.sink_index + 1)

    # run ford fulkerson algorithm to get max flow
    max_flow = graph.ford_fulkerson(graph.source_index, graph.sink_index)

    # if max_flow == total_person, means all person have been allocated to a car while satisfying all constraints
    if max_flow != total_person:
        return None

    # now, we know there's a solution, we just need to backtrack from each person vertex to find their respective car
    return graph.backtrack()







