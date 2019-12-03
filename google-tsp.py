from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(G,start,nodes):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = nx.to_numpy_matrix(G).tolist()
    print(data['distance_matrix'])  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = start
    return data


def print_solution(manager, routing, assignment):
    """Prints assignment on console."""
    #print('Objective: {} miles'.format(assignment.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = []
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output.append(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    #plan_output += ' {}\n'.format(manager.IndexToNode(index))
    #print(plan_output)
    #plan_output += 'Route distance: {}miles\n'.format(route_distance)
    return plan_output


def Google_inefficient_TSP(G,nodes,start,name):
    """Entry point of the program."""
    # Instantiate the data problem.
    G = nx.subgraph(G, nodes)
    data = create_data_model(G,start,nodes)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

