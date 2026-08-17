import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def create_data_model(hub_coords, critical_nodes):
    data = {}
    locations = [hub_coords] + [[n['lat'], n['lon']] for n in critical_nodes]
    data['locations'] = locations
    data['num_vehicles'] = 2 if len(critical_nodes) > 2 else 1
    data['depot'] = 0
    
    # Build Distance Matrix (scaled up to integers for OR-Tools)
    matrix = []
    for i in range(len(locations)):
        row = []
        for j in range(len(locations)):
            dist = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
            row.append(int(dist * 1000)) # meters
        matrix.append(row)
    
    data['distance_matrix'] = matrix
    return data

def calculate_optimal_dispatch(hub_coords, critical_nodes):
    if not critical_nodes:
        return []

    data = create_data_model(hub_coords, critical_nodes)
    
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return []

    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(previous_index)
            to_node = manager.IndexToNode(index)
            
            # Map index back to coordinates
            # Note: hub is [lat, lon]. OR-tools locations are [lat, lon]. Map requires [lon, lat] for PyDeck.
            from_loc = data['locations'][from_node]
            to_loc = data['locations'][to_node]
            
            routes.append({
                "start": [from_loc[1], from_loc[0]], # [lon, lat]
                "end": [to_loc[1], to_loc[0]]
            })

    return routes
