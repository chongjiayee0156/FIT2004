from typing import List

"""
FIT2004 ALGORITHMS AND DATA STRUCTURE 
ASSIGNMENT 1 FINAL SUBMISSION
Student name: CHONG JIA YEE
Student ID: 33563888
Campus: MALAYSIA
Date: 15/9/2023
"""

"""
QUESTION 1: Fast Food Chain
EXPLANATION: FIND MAXIMUM PROFIT WITH CONSTRAINT THAT NO 2 RESTAURANTS CAN BE WITHIN D KILOMETERS OF EACH OTHER
GENRE: DYNAMIC PROGRAMMING, BACKTRACKING
"""


def restaurantFinder(d: int, site_list: List[int]):
    """
    Function Description:
      -Solves the restaurant finder problem of finding sites from a list of sites to build restaurant so that maximum
       revenue is generated with a constraint that no two restaurants can be within 'd' kilometers of each other.

    Precondition:
      -site_list contains the sites in the order they appear along the freeway, e.g., for each i, the distance between i
       th and (i + x)th site is x kms.
      -d is always a non-negative integer
      -site_list always contains at least 1 site.

    Postcondition:
      -The list selected_sites contain the site numbers in ascending order.

    Input:
      -d: An integer representing the maximum distance between two restaurants.
      -site_list: A list of integers representing the profit of annual revenue of each restaurant if it is opened at the site.

    Return:
      -(total_revenue, selected_sites)
      -total_revenue: An integer representing the e total annual revenue if the company opens restaurants at the sites in
      selected_sites
      -selected_sites: A list of integers containing the site numbers where the company should open their restaurants to maximise the revenue

      Time Complexity:
        Best: O(N), n is length of site_list
        Worst: O(N), n is length of site_list
            - Given n is the number of elements in the input list, site_list
            - Initializing memo and copying input list to a new site_list with placeholder at first index will
            be o(n) best/worst case in time
            - Looping through each site to calculate the max profit will be o(n) best/worst case in time,
            - Backtracking will be o(n) worst case in time, ie: when d == 0.
            - o(1) best case in time, ie: when d>len(site_list)

    Space Complexity:
      Input: O(N), n is length of site_list
      Aux: O(N), n is length of site_list
         - New site_list with 1 extra element and memo array all require O(n) aux space
         - O(n) + O(n)  = O(n).

    """
    # copy all sites to increase all index by 1 (by introducing a placeholder in the first index)
    # o(n)
    site_list = [0] + [site for site in site_list]

    # memo to store the maximum profit from [1..i], including site i
    # o(n)
    memo = [0] * len(site_list)

    for index in range(1, len(site_list)):  # o(n)
        if index - d <= 0:
            # current site is within first d sites of site list.
            # due to d constraint, only 1 site with maximum annual revenue will be chosen among these first d sites
            # hence, site with max profit among first d sites will be chosen
            # by comparing profit 1 by 1, we will always get memo[index-1] = max profit possible

            # hence, if current site has more profit than all previous sites, store current site as max profit in the
            # memo, else, previous site's memo (index-1), which represents the max profit among all
            # site_list[1..index-1] will be the max profit
            memo[index] = max(memo[index - 1], site_list[index])
        else:
            # current site has the ability to have previous sites
            # now check whether or nt to include current site to selected sites

            # if include current site, then we can only include max profit of site_list[1...index-d-1] since
            # we cant consider selecting previous d sites
            # if we do not include current site, then the max profit will be the same as previous site in memo array,
            # since memo array will always have the maximum profit stored in index-1

            # we decide whether or nt to include current site by comparing which of the 2 discussed methods will
            # generate higher profit
            if memo[index - 1] > memo[index - d - 1] + site_list[index]:
                # higher profit if current site is nt included
                memo[index] = memo[index - 1]
            else:
                # higher profit if current site is included
                memo[index] = memo[index - d - 1] + site_list[index]

    # since we hv processed all the sits, last element of memo should store the maximum possible profit
    total_revenue = memo[-1]
    selected_sites = []
    # now we backtrack
    while index > 0:  # o(n) worst case, o(1) best case
        # if current max profit is diff from previous site max profit, tht means we have included the current site
        if memo[index] != memo[index - 1]:
            selected_sites.append(index)
            # since this site is included, nxt site which can be considered shd be d distance away
            index = index - d - 1
        else:
            # else, current site is nt included, rpt the same process for index-1 site
            index -= 1

    # reverse the sites from descending to ascending order
    selected_sites.reverse()  # o(n)

    # return max revenue and all selected sites for tht revenue
    return total_revenue, selected_sites


"""
QUESTION 2: CLIMB KING
EXPLANATION: FIND SHORTEST PATH FROM 1 START TO MULTIPLE EXITS WHICH PASSES THROUGH AT LEAST 1 KEY
GENRE: ADJACENCY GRAPH, DIJKSTRA, BACKTRACKING
CLASSES: Edge, Vertex, Graph, FloorGraph, MinHeap
"""

class Edge:

    def __init__(self, u: int, v: int, w: int):
        """
        Function description: Initialize the edge with the given source vertex id, target vertex id and weight.
        :Input:
            argv1: u, an integer representing the source vertex id.
            argv2: v, an integer representing the target vertex id.
            argv3: w, an integer representing the weight of the edge.
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        return f"(weight: {self.w}, vertex: {self.v})"

class Vertex:
    def __init__(self, id):
        """
        Function description: Initialize the vertex with the given id.
        :Input:
            argv1: id, an integer representing the vertex id.

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.id = id
        self.edges = []
        # for traversal
        self.visited = False
        self.discovered = False
        # for backtracking
        self.previous = None
        self.time = 0
        self.index = id
        # for keys identification
        self.is_key = False

    def reset(self):
        """
        Function description: Reset the attributes of vertex so
                            when dijkstra is run again,
                            the attributes are reset to default value
        :Input: None
        :Output, return : None

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # for traversal
        self.visited = False
        self.discovered = False
        # for backtracking
        self.previous = None
        self.time = 0

    def update_index(self, new_index: int):
        """
        Function description: Update the index of the vertex in minheap.
        :Input:
            argv1: new_index, an integer representing the new index of the vertex.
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.index = new_index

    def add_edge(self, edge):
        """
        Function description: Add an edge to the vertex.
        :Input:
            argv1: edge, an Edge object representing the edge to be added.
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.edges.append(edge)

    def __str__(self):
        ret = str(self.id) + ": "
        for edge in self.edges:
            ret += str(edge)
        return ret

    def is_visited(self):
        """
        Function description: Check if the vertex is visited.
        :Input: None
        :Output, return or postcondition: Boolean value representing whether the vertex is visited.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return self.visited

    def is_discovered(self):
        """
        Function description: Check if the vertex is discovered.
        :Input: None
        :Output, return or postcondition: Boolean value representing whether the vertex is discovered.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return self.discovered

    def mark_visited(self):
        """
        Function description: Mark the vertex as visited.
        :Input: None
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.visited = True

    def mark_discovered(self):
        """
        Function description: Mark the vertex as discovered.
        :Input: None
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.discovered = True

    def check_is_key(self):
        """
        Function description: Check if the vertex is a key.
        :Input: None
        :Output, return or postcondition: Boolean value representing whether the vertex is a key.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return self.is_key

    def mark_key(self):
        """
        Function description: Mark the vertex as a key.
        :Input: None
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.is_key = True

class Graph:
    """
    Description: Built a graph class using adjacency list concept
    which stores list of Vertex objects
    which stores list of Edge object.
    """

    def __init__(self, num_of_vertices):
        """
        Function description: Initialize the graph with the given number of vertices.
        Approach description (if main function): Create a list of vertices.
        :Input:
            argv1: num_of_vertices, an integer representing the number of vertices.
        :Output, return or postcondition: None

        :Time complexity: O(num_of_vertices), where num_of_vertices is the number of vertices.
        :Aux space complexity: O(num_of_vertices), where num_of_vertices is the number of vertices.
        """

        self.vertices: List[Vertex] = [None] * (num_of_vertices + 1)
        for i in range(num_of_vertices + 1):
            self.vertices[i] = Vertex(i)

    def __str__(self):
        ret = ""
        for vertex in self.vertices:
            ret += "Vertex " + str(vertex) + "\n"
        return ret

    def reset(self, clear_edges_to_dummy_node=False):
        """
        Function description: Reset the graph.
        :Input:
            argv1: clear_edges_to_dummy_node, a boolean representing whether the edges to the dummy node should be cleared.
        :Output, return or postcondition: None

        :Time complexity: O(|V|), where |V| is the number of vertices.
        :Aux space complexity: O(1)
        """
        for vertex in self.vertices:  # o(V)
            vertex.reset()
        if clear_edges_to_dummy_node:
            self.vertices[-1].edges = []

    def get_vertex(self, index) -> Vertex:
        """
        Function description: Get the vertex with the given index.
        :Input:
            argv1: index, an integer representing the index of the vertex.
        :Output, return or postcondition: Vertex object

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return self.vertices[index]

    def dijkstra_aux(self, source: int, total_no_keys: int):
        """
        Function description: format the input from vertex index to vertex object for dijkstra input
        :Input:
            argv1: source, an integer representing the source vertex id.
            argv2: total_no_keys, an integer representing the total number of keys.
        :Output, return or postcondition: Call dijkstra function

        :Time complexity: O(dijsktra), where dijkstra is the time complexity of dijkstra function.
        :Aux space complexity: O(dijsktra), where dijkstra is the aux space complexity of dijkstra function.
        """
        source_vertex = self.vertices[source]
        # get vertex object from the input index and call dijkstra
        return self.dijkstra(source_vertex, total_no_keys)

    def dijkstra(self, source: Vertex, total_key_no: int):
        """
        Function description: Dijkstra's algorithm to find the shortest path from the given source vertex to all keys.

        Approach description (if main function): Djikstra's algorithm, implemented with a MinHeap. Find the shortest path
                                                 from the given source to all vertices. Terminates early when shortest path
                                                 to all keys are found. (total_key_no is used to keep track of the number)

        :Input:
            argv1: source, a Vertex object representing the source vertex.
            argv2: total_key_no, an integer representing the total number of keys.
        :Output, return or postcondition: None

        :Time complexity: O(|E|log|V|) where |V| is the number of vertices and |E| is the number of edges/
                          - number of looping time for while loop + for loop = O(E)
                          - in each loop, min heap is used to rise or sink the vertex, which takes O(logV) time complexity
                          - hence, total time complexity: O(E) * O(logV) = O(ElogV)
        :Aux space complexity: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges. This is for adjacency list.
        """

        # time is the time taken to reach the vertex from source
        # since source is the starting point, its time is 0
        discovered = (0, source)  # (time, vertex) tuple.

        # initialize heap
        # push source vertex to heap
        heap = MinHeap()  # o(1)
        heap.push(discovered)  # o(logV) time complexity o(1) aux space complexity
        # since source is pushed to heap, it is discovered
        source.mark_discovered()  # o(1)

        result = []

        # while heap is not empty
        while not heap.is_empty():

            # serve minimum time vertex
            time, vertex_u = heap.serve()  # o(logV) time complexity o(1) aux space complexity

            # since we are seving the next minimum vertex, its time from source has been confirmed
            # hence, it can be marked as visited
            vertex_u.mark_visited()

            # if the vertex served is key,
            if vertex_u.check_is_key():
                # found min dist to one of the keys
                total_key_no -= 1
            # minus 1 from keys-to-be-found

            # sdijkstra can terminate early when shortest path to all keys are found
            if total_key_no == 0:
                break

            # add all adjacent vertices to queue, mark them as discovered
            for edge in vertex_u.edges:
                # get adjacent vertex
                index_vertex_v = edge.v
                vertex_v = self.vertices[index_vertex_v]

                # if it is not discovered, mark as discovered, set previous vertex, set time
                if not vertex_v.is_discovered():
                    vertex_v.mark_discovered()
                    vertex_v.previous = vertex_u
                    vertex_v.time = vertex_u.time + edge.w
                    # push new time and vertex to heap
                    heap.push((vertex_v.time, vertex_v))  # o(logV) time complexity o(1) aux space complexity

                # else, it is discovered but not visited
                # means still in the heap
                # as long as it has not been visited, it means that the time is not confirmed
                # hence, check if time can be updated
                elif not vertex_v.is_visited():
                    # if time from current vertex is smaller than the time from its previously stores vertex
                    if vertex_v.time > vertex_u.time + edge.w:
                        # update time and previous vertex
                        vertex_v.time = vertex_u.time + edge.w
                        vertex_v.previous = vertex_u
                        # update heap with new time and vertex
                        heap.update((vertex_v.time, vertex_v), vertex_v.index)  # O(logV) time complexity

    def backtracking_shortest_path(self, target: Vertex, flipped_graph=False):
        """
        Function description: Backtrack from the given target vertex to the source vertex to find the shortest path.
        Approach description (if main function): A while loop is used to backtrack.
                                                 if the graph is flipped, ignore the last dummy node while backtracking.
                                                 we want path from (key-exit), not (key-dummy node).
                                                 if the graph is not flipped, reverse the entire path collected.
                                                 hence, you get from (key-start) to (start-key)
        :Input:
            argv1: target, a Vertex object representing the target vertex.
            argv2: flipped_graph, a boolean representing whether the graph is flipped.
        :Output, return or postcondition: A list of integers representing the shortest path from the source vertex to the target vertex.

        :Time complexity:
                    best: O(1), when key is the start or exit.
                    worst: O(|V|), where |V| is the number of vertices.
                        when the path requires traversal on the entire graph and
                        key is the start for flipped graph
                        or exit for non-flipped graph.
        :Aux space complexity: O(|V|), where |V| is the number of vertices to be stored in path.
        """
        # analyze time and space complexity for this function
        current = target
        path = []

        while current.previous is not None:
            path.append(current.id)
            current = current.previous

        if not flipped_graph:
            path.append(current.id)
            path.reverse()
        # else, # ignore last node, it is dummy node

        return path

    def add_edge(self, u: int, v: int, w: int):
        """
        Function description: Add an edge to the graph.
        :Input:
            argv1: u, an integer representing the source vertex id.
            argv2: v, an integer representing the target vertex id.
            argv3: w, an integer representing the weight of the edge.
        :Output, return or postcondition: None

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        edge = Edge(u, v, w)
        self.vertices[u].add_edge(edge)

    def add_keys(self, keys: List[tuple[int, int]]):
        """
        Function description: Add keys to the graph.
        :Input:
            argv1: keys, a list of tuples representing the keys in the floor graph.
        :Output, return or postcondition: None

        :Time complexity: O(len(keys)), where len(keys) is the number of keys.
        :Aux space complexity: O(1)
        """
        for key_vertex, fighting_time in keys:
            key_vertex = self.vertices[key_vertex]
            key_vertex.mark_key()

class FloorGraph:
    """
    Class description: This class represents the floor graph.
                       It has function to create graph data structure and determine shortest route from start to exit which passes through a key.
                       Shortest route is determined by smallest sum of weighted edges and key's fighting time from that route.

    Approach description (if main function): Initialize the floor graph by with the given paths and keys.
                                             Floorgraph object can be reused for different combinations of start and exits to cater to diff floor of the tower.
                                             This function calls many functions from other class in order to maintain OOP design.

    """

    def __init__(self, paths: List[tuple[int, int, int]], keys: List[tuple[int, int]]):
        """
        Function description: Initialize the floor graph with the given paths and keys.
        Approach description (if main function): Create a graph by calling create_graph function which creates a Graph object.
                                                 Graph data structure is implemented using adjacency list.

        Precondition:
            argv1: paths
                • u, v, and x are non-negative integer.
                • The paths are directed.
                • the location IDs are continuous from 0 to |V | - 1 where |V | is total
                number of locations.
                • all locations are connected by at least 1 path.
            arg2: keys
                • k and y are non-negative integer.
                • all of the k values are from the same set as the location ID, that is from the set of {0, 1, 2, ..., |V | − 1}.
                •each of the k values in the list keys list is unique.

        :Input:
            argv1: paths, a list of tuples representing the paths in the floor graph. (u,v,x)
                    • u is the starting location ID for the path.
                    • v is the ending location ID for the path.
                    • x is the amount of time needed to travel down the path from location-u to location-v.
            argv2: keys, a list of tuples representing the keys in the floor graph. (k, y)
                • k is the location ID where a key can be found in the floor.
                • y is the amount of time needed to defeat the monster and retrieve the key if you choose
                to fight the monster.

        :Output: None

        :Time complexity:
            best & worst: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges.
                        • referencing self.paths and self.keys to input list has o(1) time complexity
                        • looping through edges to find max vertex id has worst case of o(E), but best case of o(1)
                        • when a Graph class is initialized, it will create a list of vertices with size |V|+1. (o(V))
                        • then, when edges are added to the graph, each vertex will have its own list of edges. (o(E))

        :Space complexity:
                    input: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges.
                        • worst case for length of keys is |V|, hence o(|V|)
                    :Aux: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges.
                        • only creating graph will take up additional space
                        • when a Graph class is initialized, it will create a list of vertices with size |V|+1. (o(V))
                        • then, when edges are added to the graph, each vertex will have its own list of edges. (o(E))
                    total: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges.
        """

        # store input argument as class attributes
        self.paths = paths  # o(1) time and aux space complexity
        self.keys = keys  # o(1) time and aux space complexity

        # keep track of total no. of keys so dijkstra can end early if keys are lesser than vertices and are closer to source
        self.total_no_keys = len(keys)  # o(1) time and aux space complexity

        # from the paths given, find max vertex id by searching maximum integer among all locations u and v.
        # this is used to create graph and dummy node later.
        self.max_vertex_id = self.find_max_vertex_id(
            paths)  # o(E) worst time complexity, o(1) best time complexity, o(1) aux space complexity

        # create graph by calling create_graph function
        # create graph function will create a Graph object and add edges and keys to the graph
        self.graph = self.create_graph(paths, to_be_inverted=False)  # o(|V|+|E|) time and aux space complexity

        # create another graph which is inverted graph
        # it has same locations and keys, only the edges u,v are switched to v,u
        self.inverted_graph = self.create_graph(paths, to_be_inverted=True)  # o(|V|+|E|) time and aux space complexity

    def find_max_vertex_id(self, paths):
        """
        Function description: Find the maximum vertex id.
        Approach description (if main function): Iterate through the paths and find the maximum vertex id.

        :Input:
            argv1: paths, a list of tuples representing the paths in the floor graph.
        :Output, return or postcondition: An integer representing the maximum vertex id.

        :Time complexity: O(len(paths)), where len(paths) is the number of paths.
        :Aux space complexity: O(1)
        """

        # initialize max vertex id as 0
        max_vertex_id = 0

        # find total no of vertices by finding max location
        for u, v, x in paths:
            # if u > v, then compare u with the max vertex id
            if u > v:
                if u > max_vertex_id:
                    # if u is greater than max vertex id, update u as max vertex id
                    max_vertex_id = u
            else:
                # if v > u, then compare v with the max vertex id
                if v > max_vertex_id:
                    # if v is greater than max vertex id, update v as max vertex id
                    max_vertex_id = v

        # return max vertex id
        return max_vertex_id

    def create_graph(self, paths, to_be_inverted=False):
        """
        Function description: Create a graph with the given paths and keys.
        Approach description (if main function): Create Graph object and add edges and keys to the graph.

        :Input:
            argv1: paths, a list of tuples representing the paths in the floor graph.
            argv2: keys, a list of tuples representing the keys in the floor graph.
        :Output, return or postcondition: Graph object

        :Time complexity: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges.
        :Aux space complexity: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges.
        """

        # set edges of the graph
        if not to_be_inverted:
            # initialize graph
            graph = Graph(self.max_vertex_id)  # o(V)
            for u, v, x in paths:
                graph.add_edge(u, v, x)  # o(E)
        else:
            # initialize inverted graph with 1 dummy node
            graph = Graph(self.max_vertex_id + 1)  # add one dummy node    o(V)
            for u, v, x in paths:  # reverse the edge
                graph.add_edge(v, u, x)  # o(E)

        # set keys for the graph
        graph.add_keys(self.keys)  # o(v)

        return graph

    def climb(self, start: int, exits: List[int]) -> tuple[int, List[int]] or None:
        # generate document for climb function
        """
        Function description: Determine shortest route from start to exit which passes through a key.
        Approach description : Run dijkstra to get shortest path from source to all keys.
                                Now we need to find the sortest path from all keys to any of the exits, but avoid running dijkstra |keys| times,
                                we can connect a dummy node to all exits with 0 weight edges.
                                Now we have only one exit.
                                Invert the graph and run Dijkstra from dummy node to find shortest path from dummy node to each key.
                                Since dummy node is only connected to exits, each path from dummy node to key will pass through exit.
                                This makes the paths valid.
                                Hence, in other words, we are finding min path from any of the exits to every key.
                                Now, in each key vertex, we have its shortest path from source and its shortest path from exit.
                                Looping through each key vertex, we find the shortest path from source to exit by combining its time attribute in both graphs and fighting time.
                                However, we only accept key vertices that are visited in both graphs, if it is not visited, despite having smaller time, it is not valid.
                                This is because it means that the key cannot be reached from start or any of the exits.

        Precondition:
            argv1: start
                • start is a non-negative integer.
                • start is from the same set as the location ID, that is from the set of {0, 1, 2, ..., |V | − 1}.
            argv2: exits
                • exits is a list of non-negative integers.
                • each of the exit values in the list exits is from the same set as the location ID, that is from the set of {0, 1, 2, ..., |V | − 1}.
                • each of the exit values in the list exits is unique.

        :Input:
            argv1: start, an integer representing the start location.
            argv2: exits, a list of integers representing the exit locations.

        :Output: A tuple of (total_time, route) or None if there is no valid route.
            argv1: total_time, an integer representing the total time taken to reach the exit.
            argv2: route, a list of integers representing the route taken from the start location to the exit location.

        :Time complexity: O(|E|log|V|), where |V| is the number of vertices and |E| is the number of edges.
                            • reset: O(|V|)
                            • dijkstra: O(|E|log|V|) more explanation on the documentation of function itself
                            • connect_dummy_node: O(|V|)
                            • dijkstra 2nd time: O(|E|log|V|)
                            • loop to find min key vertex: O(|V|)
                            • backtracking_shortest_path: O(|V|)
                            • extend function: O(|V|)

        :Aux space complexity: O(|V|+|E|), where |V| is the number of vertices and |E| is the number of edges.
                            • dijkstra: O(|V|+|E|)
                            • connect_dummy_node: O(|V|) when all vertices are exits
                            • dijkstra: O(|V|+|E|)
                            • backtracking_shortest_path: O(|V|) when shortest path involves all vertices and
                                                          key is at start in flipped graph or exit in normal graph
                            • extend: O(|V|)
        """

        # reset graph and inverted graph incase the graph is reused for different start and exits
        self.graph.reset()  # o(V) time complexity, o(1) aux space complexity
        self.inverted_graph.reset(clear_edges_to_dummy_node=True)  # o(V) time complexity, o(1) aux space complexity

        # run dijkstra to get shortest path from source to all keys
        self.graph.dijkstra_aux(start,
                                self.total_no_keys)  # o(|V|+|E|) time complexity, o(|V|+|E|) aux space complexity

        dummy_node_index = self.connect_dummy_node(exits)  # o(V) time complexity, o(1) aux space complexity

        # run Dijkstra from dummy node to find shortest path from dummy node to each key
        # since dummy node is linked only directly to the exits, the only way for it to
        # reach any of the keys is to pass through exit
        # hence, in other words, we are finding min path from any of the exits to every key
        self.inverted_graph.dijkstra_aux(dummy_node_index,
                                         self.total_no_keys)  # o(|V|+|E|) time complexity, o(|V|+|E|) aux space complexity

        total_time = float('inf')
        min_path_key_index = 0

        # get minimum time from start to exit and its relevant key vertex
        # loop through each key vertex
        for vertex_index, fighting_time in self.keys:  # o(V) time complexity, o(1) aux space complexity
            # get key vertex object from both graphs
            key_vertex_g1 = self.graph.get_vertex(vertex_index)
            key_vertex_g2 = self.inverted_graph.get_vertex(vertex_index)
            # if that key vertex has been visited in both graphs
            # means it can be reached from start and exit
            if key_vertex_g1.is_visited() and key_vertex_g2.is_visited():
                # calculate total time for shortest path going through this key
                path_total_time = key_vertex_g1.time + key_vertex_g2.time + fighting_time
                # if it has shorter path time, update total time and key index
                if path_total_time < total_time:
                    total_time = path_total_time
                    min_path_key_index = vertex_index

        # if total time is infinity, means there are no valid route
        if total_time == float('inf'):
            # return None
            return None
        # else, we have found the key vertex in the shortest path
        # hence, we can backtrack to find its shortest path
        else:
            # get vertex object
            graph_1_vertex = self.graph.get_vertex(min_path_key_index)
            graph_2_vertex = self.inverted_graph.get_vertex(min_path_key_index)

            # backtrack from key to start
            route = self.graph.backtracking_shortest_path(
                graph_1_vertex)  # o(V) time complexity, o(V) aux space complexity
            # backtrack from (key+1) to exit
            # to prevent overlapping of the key vertex, start tracking key to exit from key+1
            key_to_exit_path = self.inverted_graph.backtracking_shortest_path(graph_2_vertex.previous,
                                                                              flipped_graph=True)
            # combine both paths to get start to exit path
            route.extend(key_to_exit_path)  # o(V) time complexity, o(1) aux space complexity

            # return total time and route
            return (total_time, route)

    def connect_dummy_node(self, nodes_to_be_connected: List[int]):
        """
        Function description: Add a dummy node to the graph.
        Approach description (if main function): Iterate through the nodes to be connected and add the dummy node to the graph.

        :Input:
            argv1: paths, a list of tuples representing the paths in the floor graph.
            argv2: nodes_to_be_connected, a list of integers representing the nodes to be connected.
        :Output, return or postcondition: None

        :Time complexity: O(len(nodes_to_be_connected)), where len(nodes_to_be_connected) is the number of nodes to be connected.
                          In this case len(nodes_to_be_connected) is the number of exits.
                          worst case scenario is when all locations are exits, hence o(V).
        :Aux space complexity: O(1)
        """
        dummy_node_index = self.max_vertex_id + 1
        for node in nodes_to_be_connected:
            self.inverted_graph.add_edge(dummy_node_index, node, 0)
        return dummy_node_index

    def __str__(self):
        ret = "\nAdjacency list:\n"
        ret += str(self.graph)

        # check keys
        # ret += str([str(x) for x in self.graph.vertices if x.check_is_key()])
        return ret

class MinHeap:
    """
    Description: Built a min heap which stores a list of tuples representing the time and vertex.
               Difference from usual python built in heap is it has an update function.
               Index of each vertex in heap will be keep tracked as an attribute of that Vertex object.
               whenever we found a shorter time for that Vertex and we wish to update its time in the heap,
               we can use the index recorded to access our heap and change the time.
               then, that vertex will be risen to its supposed position based on its time.
               This saves space as compared to built in heap since it will not contain all the previous useless time of the vertex.
    """

    def __init__(self):
        """
        Function description: Initialize the heap.
        :Input: None
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.heap: List[(Vertex.time, Vertex)] = [0]

    def push(self, time_vertex: tuple[int, Vertex]):
        """
        Function description: Push the given time vertex to the heap.
        :Input:
            argv1: dist_vertex, a tuple representing the time and vertex.
        :Output, return or postcondition: None
        :Time complexity: O(log v), where v is the number of vertices.
        :Aux space complexity: O(1)
        """
        self.heap.append(time_vertex)  # o(1) time and aux space complexity
        # this vertex is at the end of the heap, update its index in heap
        time_vertex[1].update_index(len(self.heap) - 1)  # o(1) time and aux space complexity
        # rise the vertex to its supposed position based on its time
        self.rise(
            len(self.heap) - 1)  # o(log v) worst time complexity, o(1) best time complexity, o(1) aux space complexity

    def rise(self, index):
        """
        Function description: Rise the given index.
        Approach description (if main function): Compare the given index with its parent index.
                                                 If the given index is smaller than its parent index,
        :Input:
            argv1: index, an integer representing the index of the vertex.
        :Output, return or postcondition: None
        :Time complexity:
                worst: O(log v), where v is the number of vertices.
                best: O(1)
        :Aux space complexity: O(1)
        """
        # if index is at top of the heap, it cant be rise anymore
        # if not, enter the loop
        while index > 1:  # o(log n) worst time complexity, o(1) best time complexity, o(1) aux space complexity
            # get parent index
            parent_index = index // 2
            # if the given index is smaller than its parent index, swap them
            if self.heap[index][0] < self.heap[parent_index][0]:
                self.swap(index, parent_index)  # o(1) time and aux space complexity
                # update index to parent index
                index = parent_index
            else:
                # break if current index is larger or equal to parent index
                break

    def swap(self, i, j):
        """
        Function description: Swap the given indices.
        :Input:
            argv1: i, an integer representing the index of the vertex.
            argv2: j, an integer representing the index of the vertex.
        :Output, return or postcondition: None
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # swap content of both indices in heap
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        # update the swapped index
        self.heap[i][1].update_index(j)
        self.heap[j][1].update_index(i)

    def update(self, time_vertex: tuple[int, Vertex], index: int):
        """
        Function description: Update the given time vertex at the given index.
        :Input:
            argv1: time_vertex, a tuple representing the time and vertex.
            argv2: index, an integer representing the index of the vertex.
        :Output, return or postcondition: None
        :Time complexity: O(log v), where v is the number of vertices.
        :Aux space complexity: O(1)
        """
        if 0 < index < len(self.heap):
            # Vertex is already in the heap, update its time
            self.heap[index] = time_vertex
            self.rise(index)

    def serve(self):
        """
        Function description: Serve the minimum time vertex.
        :Input: None
        :Output, return or postcondition: A tuple representing the time and vertex.
        :Time complexity: O(log v), where v is the number of vertices.
        :Aux space complexity:  O(1)
        """
        if len(self.heap) > 1:
            # swap
            self.swap(1, len(self.heap) - 1)  # o(1)
            # extract last min element
            min_dis_vertex = self.heap.pop()  # o(1) time and aux space complexity
            # sink the first element which was swapped

            self.sink(1)  # o(log v) worst time complexity, o(1) best time complexity, o(1) aux space complexity

            # return min dist vertex
            return min_dis_vertex
        raise IndexError("pop from empty heap")

    def largest_child(self, index: int) -> int:
        """
        Function description: Get the largest child of the given index.
        Approach description (if main function): Compare the left child and right child of the given index.
        :Input:
            argv1: index, an integer representing the index of the vertex.
        :Output, return or postcondition: An integer representing the index of the largest child.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        left_child = 2 * index
        right_child = 2 * index + 1
        if left_child == len(self.heap) - 1 or self.heap[left_child][0] < self.heap[right_child][0]:
            return left_child
        else:
            return right_child

    def sink(self, index: int) -> None:
        """
        Function description: Sink the given index.
        Approach description (if main function): Compare the given index with its largest child index.
        :Input:
            argv1: index, an integer representing the index of the vertex.
        :Output, return or postcondition: None
        :Time complexity: O(log v), where v is the number of vertices.
        :Aux space complexity: o(1)
        """
        while len(self.heap) > (index * 2):
            child = self.largest_child(index)  # o(1)
            # if current time is smaller than child time, break
            if self.heap[index][0] <= self.heap[child][0]:
                break
            else:
                # else, swap with child and loop again
                self.swap(child, index)
                index = child

    def __len__(self):
        return len(self.heap)

    def is_empty(self):
        """
        Function description: Check if the heap is empty.
        :Input: None
        :Output, return or postcondition: Boolean value representing whether the heap is empty.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return len(self.heap) == 1