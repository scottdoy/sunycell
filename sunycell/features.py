import numpy as np
from shapely.geometry import Polygon
import pandas as pd
from scipy import stats
from skimage import morphology, segmentation
from matplotlib.path import Path as mplPath
import matplotlib.tri as T


def get_polygon_from_pts(pts):
    polygons = []
    for pt in pts:
        X = pt[0]
        Y = pt[1]
        point_list = [(x,y) for x,y in zip(X,Y)]
        poly = Polygon(point_list)
        polygons.append(poly)
    return polygons

# Obtaining vaid cordinates Satellite/Tumor region
def get_edge_coordinates(elements):
    coord = []
    for i,element in enumerate(elements):
        points = element['points']
        points = [x[:-1] for x in points]

        X = np.array([int(p[0]) for p in points], dtype=np.int64)
        Y = np.array([int(p[1]) for p in points], dtype=np.int64)

        coord.append([X, Y])
        
    return coord 

def get_centroid_coordinates(elements):
    """Given a set of htk elements, calculate the centroid coordinate for each polygon therein."""

    coord = []
    for element in elements:
        points = element['points']
        points = [x[:-1] for x in points]

        X = np.array([int(p[0]) for p in points], dtype=np.int64)
        Y = np.array([int(p[1]) for p in points], dtype=np.int64)

        coord.append([X.mean(), Y.mean()])
        
    return coord 

def cut_triangles(tri, coordinates, verbose=True):
    """Eliminate triangles whose branches cross the object defined by coordinates """
    
    # every triangle in the delaunay object that contains these points  
    eliminate_triangles = tri.find_simplex(coordinates) 
    if verbose:
        print(f'Number of points that were tested: {len(eliminate_triangles)}')

    # eliminates duplicate
    eliminate_triangles = np.unique(eliminate_triangles[eliminate_triangles>-1]) 
    if verbose:
        print(f'List of unique, non-negative simplex coordinates: {eliminate_triangles}')

    #creates a copy of triangles
    tri_simplices = tri.simplices.copy() 
    tri_simplices = np.delete(tri_simplices, eliminate_triangles, axis=0)
    return tri_simplices

def get_poly_centroid(poly):
    """Gets the centroid of the polygon and processes it to be a numpy array."""
    pass

def get_poly_boundaries(poly):
    """Gets the boundaries of the polygon and processes it to be a numpy array."""
    pass

def descriptive_stats(x, feature_prefix=''):
    """Create a pandas dataframe containing the features calculated on x.

    x is an input array for which the features below are calculated.

    feature_prefix is prepended to each of the feature names to create unique columns."""

    feature_names = [
        'minimum', 'maximum', 'mean', 'variance', 'standard_deviation',
        'skewness', 'kurtosis', 'moment_5', 'moment_6', 'moment_7',
        'moment_8', 'moment_9', 'moment_10', 'moment_11', 
        'geometric_mean', 'harmonic_mean'
    ]
    feature_names = [feature_prefix+x for x in feature_names]

    # Ensure x is a numpy array
    x = np.array(x)

    minimum = np.amin(x)
    maximum = np.amax(x)
    mean = np.mean(x)
    variance = np.var(x)
    standard_deviation = np.std(x)
    skewness = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    moment_5 = stats.moment(x, moment = 5)
    moment_6 = stats.moment(x, moment = 6)
    moment_7 = stats.moment(x, moment = 7)
    moment_8 = stats.moment(x, moment = 8)
    moment_9 = stats.moment(x, moment = 9)
    moment_10 = stats.moment(x, moment = 10)
    moment_11 = stats.moment(x, moment = 11)

    # mean values require x > 0, so we will scale these values appropriately
    geometric_mean = stats.gmean(x - x.min() + 1)
    harmonic_mean = stats.hmean(x - x.min() + 1)
    features = [minimum, maximum, mean, variance, standard_deviation,\
                skewness, kurtosis, moment_5, moment_6, moment_7,\
                moment_8, moment_9, moment_10, moment_11, geometric_mean, harmonic_mean]
    features_dict = {}
    for feature_name, feature in zip(feature_names, features):
        features_dict[feature_name] = feature
    return pd.DataFrame([features_dict])

def assign_wave_index(tum_bin, sat_bounds, max_dilations=1500):
    tum_counter = 0
    sat_wave_number = np.zeros((len(sat_bounds),1))

    while True:
        # Dilate the tumor
        tum_bin = morphology.binary_dilation(tum_bin)

        # Increment the counter
        tum_counter += 1

        # Check to see if any satellites are "hit" by the (dilated) tumor
        # Get dilated tumor boundary points
        img_tum_boundary = segmentation.find_boundaries(tum_bin)
        boundary_tum = np.nonzero(img_tum_boundary)

        # Split apart boundary coordinates
        boundary_tum_x = boundary_tum[0]
        boundary_tum_y = boundary_tum[1]
        tum_bin_points = np.array([boundary_tum_x, boundary_tum_y]).T
            
        # Get satellite wave number   
        for sat_idx, sat_bound in enumerate(sat_bounds):
            
            sat_bound = np.array([sat_bound[:,0], sat_bound[:,1]]).T
            tum_poly = mplPath(tum_bin_points)
            sat_hit = tum_poly.contains_points(sat_bound)
            if np.any(sat_hit == True) and sat_wave_number[sat_idx] == 0:
                sat_wave_number[sat_idx] = tum_counter

        # Check to see if every satellite has been hit
        if np.all(sat_wave_number > 0):
            return sat_wave_number

        # Make sure we aren't in an infinite loop
        if tum_counter > max_dilations:
            print(f"Not all sats have been assigned an index after {max_dilations} iterations. Exiting.")
            return sat_wave_number

def assign_wave_index_shapely(tum_bounds, sat_bounds, max_dilations=1500):
    tum_counter = 0
    sat_wave_numbers = np.zeros((len(sat_bounds),1))

    while True:
        # Dilate the tumor
        #tum_bin = morphology.binary_dilation(tum_bin)

        # Increment the counter
        tum_counter += 1

        # Check to see if any satellites are "hit" by the (dilated) tumor
        # Get dilated tumor boundary points
        #img_tum_boundary = segmentation.find_boundaries(tum_bin)
        #boundary_tum = np.nonzero(img_tum_boundary)

        # Split apart boundary coordinates
        #boundary_tum_x = boundary_tum[0]
        #boundary_tum_y = boundary_tum[1]
        #tum_bin_points = np.array([boundary_tum_x, boundary_tum_y]).T
        tum_bin_points = tum_bounds * 2
            
        # Get satellite wave number   
        for sat_idx, sat_bound in enumerate(sat_bounds):
            
            sat_bound = np.array([sat_bound[0], sat_bound[1]]).T    
            tum_poly = mplPath(tum_bin_points)
            sat_hit = tum_poly.contains_points(sat_bound)
            if np.any(sat_hit == True) and sat_wave_numbers[sat_idx] == 0:
                sat_wave_numbers[sat_idx] = tum_counter

        # Check to see if every satellite has been hit
        if np.all(sat_wave_numbers > 0):
            return sat_wave_numbers

        # Make sure we aren't in an infinite loop
        if tum_counter > max_dilations:
            print(f"Not all sats have been assigned an index after {max_dilations} iterations. Exiting.")
            return sat_wave_numbers

def compute_wave_graph():
    new_sat_centroids = sat_centroids
    used_indices = np.zeros(len(sat_wave_indices))
    fig = plt.figure()
    distances_list = []
    while (len(sat_wave_indices) != 0):

        flag = 0
        max_sat_number = np.max(sat_wave_indices)
        max_sat_idx = np.argmax(sat_wave_indices)
        used_indices[max_sat_idx] += 1
        while (flag == 0):
            initialTOtargets_min_distance = []
            target_matrix = []
            if used_indices[max_sat_idx] > 1:
                sat_wave_indices = np.delete(sat_wave_indices, max_sat_idx)
                new_sat_centroids = np.delete(new_sat_centroids, max_sat_idx, 0)
                used_indices = np.delete(used_indices, max_sat_idx)
                flag = 1
            else:
                target_satellites = sat_wave_indices < max_sat_number
                sat_initial_bounds = [(new_sat_centroids[max_sat_idx][0], new_sat_centroids[max_sat_idx][1])]
            
                tum_sat_distance = distance.cdist(sat_initial_bounds, tum_bounds, metric = 'euclidean')
                min_tum_distance = np.min(tum_sat_distance)
                min_tum_idx = np.argmin(tum_sat_distance)
                for ind, outcome in enumerate(target_satellites):
                    if outcome == True:
                        sat_target_bounds = [(new_sat_centroids[ind][0], new_sat_centroids[ind][1])]
                        target_distance = distance.cdist(sat_initial_bounds, sat_target_bounds, metric = 'euclidean')
                        target_matrix.append(target_distance)
                        initialTOtargets_min_distance.append(np.min(target_distance))
        
                        # if (len(initialTOtargets_min_distance) == len(np.array(np.nonzero(target_satellites)).T)):
                        if (len(initialTOtargets_min_distance) == len(target_satellites)):
                            distance_min = np.min(initialTOtargets_min_distance)
                            distance_idx = np.argmin(initialTOtargets_min_distance)
                            sat_target_number = sat_wave_indices[distance_idx]
                            
                            if distance_min < min_tum_distance:
                                distances_list.append([(max_sat_number, sat_target_number, distance_min)])
                                new_target_bounds = [(new_sat_centroids[distance_idx][0], new_sat_centroids[distance_idx][1])]
                                # fig = plt.figure()
                                for sat_bound in sat_bounds:
                                    sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                    plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                plt.scatter(tum_bounds[:,0], tum_bounds[:,1], edgecolors = 'b')
                                x = sat_initial_bounds
                                y = new_target_bounds
                                plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                plt.show()
                                if np.any((sat_wave_indices > max_sat_number))==True:
                                    sat_wave_indices = sat_wave_indices
                                    max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                    max_sat_idx = distance_idx
                                    used_indices[distance_idx] += 1
                                    target_satellites = []
                                else:
                                    sat_wave_indices = np.delete(sat_wave_indices, max_sat_idx)
                                    new_sat_centroids = np.delete(new_sat_centroids, max_sat_idx, 0)
                                    used_indices = np.delete(used_indices, max_sat_idx)
                                    if distance_idx > max_sat_idx:
                                        max_sat_number = np.float64(sat_wave_indices[distance_idx - 1])
                                        max_sat_idx = distance_idx - 1
                                        used_indices[distance_idx - 1] += 1
                                        target_satellites = []
                                        # sat_wave_indices = new_wave_indices
                                    else:
                                        max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                        max_sat_idx = distance_idx
                                        used_indices[distance_idx] += 1
                                        target_satellites = []
                                        # sat_wave_indices = new_wave_indices
                            else:
                                distances_list.append([(max_sat_number, 0, min_tum_distance)])
                                new_tum_bounds = [(tum_bounds[min_tum_idx][0], tum_bounds[min_tum_idx][1])]
                                
                                # fig, ax = plt.subplots()
                                # fig = plt.figure()
                                for sat_bound in sat_bounds:
                                    sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                    plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                plt.scatter(tum_bounds[:,0], tum_bounds[:,1],edgecolors ='b')
                                x = sat_initial_bounds
                                y = tum_bounds[min_tum_idx]
                                y = [(y[0], y[1])]
                                plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                plt.show()
                                flag = 1
                    else:
                        target_distance = np.inf
                        target_matrix.append(target_distance)
                        initialTOtargets_min_distance.append(target_distance)
                        
                        # if (len(initialTOtargets_min_distance) == len(np.array(np.nonzero(target_satellites)).T)):
                        if (len(initialTOtargets_min_distance) == len(target_satellites)):
                            distance_min = np.min(initialTOtargets_min_distance)
                            distance_idx = np.argmin(initialTOtargets_min_distance)
                            sat_target_number = sat_wave_indices[distance_idx]
                            if distance_min < min_tum_distance:
                                distances_list.append([(max_sat_number, sat_target_number, distance_min)])
                                new_target_bounds = [(new_sat_centroids[distance_idx][0], new_sat_centroids[distance_idx][1])]
                                
                                # fig, ax = plt.subplots()
                                # fig = plt.figure()
                                for sat_bound in sat_bounds:
                                    sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                    plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                plt.scatter(tum_bounds[:,0], tum_bounds[:,1],edgecolors = 'b')
                                x = sat_initial_bounds
                                y = new_target_bounds
                                plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                plt.show()
                                if np.any((sat_wave_indices > max_sat_number))==True:
                                    sat_wave_indices = sat_wave_indices
                                    max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                    max_sat_idx = distance_idx
                                    used_indices[distance_idx] += 1
                                    target_satellites = []
                                else:
                                    sat_wave_indices = np.delete(sat_wave_indices, max_sat_idx)
                                    new_sat_centroids = np.delete(new_sat_centroids, max_sat_idx, 0)
                                    used_indices = np.delete(used_indices, max_sat_idx)
                                    if distance_idx > max_sat_idx:
                                        max_sat_number = np.float64(sat_wave_indices[distance_idx - 1])
                                        max_sat_idx = distance_idx - 1
                                        used_indices[distance_idx - 1] += 1
                                        target_satellites = []
                                        # sat_wave_indices = new_wave_indices
                                    else:
                                        max_sat_number = np.float64(sat_wave_indices[distance_idx])
                                        max_sat_idx = distance_idx
                                        used_indices[distance_idx] += 1
                                        target_satellites = []
                                        # sat_wave_indices = new_wave_indices
                            else:
                                distances_list.append([(max_sat_number, 0, min_tum_distance)])
                                new_tum_bounds = [(tum_bounds[min_tum_idx][0], tum_bounds[min_tum_idx][1])]
                                
                                # fig, ax = plt.subplots()
                                # fig = plt.figure()
                                for sat_bound in sat_bounds:
                                    sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
                                    plt.scatter(sat_bound[:,0], sat_bound[:,1])
                                plt.scatter(tum_bounds[:,0], tum_bounds[:,1],edgecolors ='b')
                                x = sat_initial_bounds
                                y = tum_bounds[min_tum_idx]
                                y = [(y[0], y[1])]
                                plt.plot([x[0][0], y[0][0]], [x[0][1], y[0][1]], 'k', linewidth = 3.0)
                                plt.show()
                                flag = 1
    #Calculating cumulative distance from each satellite
    sat_tum_distances = []
    sat_wave_index = np.reshape(sat_wave_number, [len(sat_wave_number),])
    distances = np.array(distances_list)
    dis = np.reshape(distances, [distances.shape[0],3])
    for wave_num in sat_wave_index:
        number = 1
        sum_dist = 0
        idx, = np.where(dis[:,0] == wave_num)[0]
        while(number != 0):
            number = dis[idx,1]
            sum_dist = sum_dist + dis[idx,2]
            if number > 0:
                idx, = np.where(dis[:,0] == number)[0]
            else:
                number = 0    
        sat_tum_distances.append(sum_dist)
def get_triangle_lengths(tri_centroids, tri_simplices):

    t = T.Triangulation(tri_centroids[:,0], tri_centroids[:,1], tri_simplices)
    triangle_lengths = []

    for edge in t.edges:
        x1 = tri_centroids[edge[0], 0]
        x2 = tri_centroids[edge[1], 0]
        y1 = tri_centroids[edge[0], 1]
        y2 = tri_centroids[edge[1], 1]
        triangle_lengths.append( np.sqrt((x2-x1)**2 + (y2-y1)**2 ) )
        
    return triangle_lengths

def get_triangle_areas(tri_centroids, tri_simplices):
    '''Calculate area of triangles given a set of centroids and a set of simplices.'''
    
    triangle_areas = []
    for simplex in tri_simplices:
        # Pull out the points for this triangle
        p1 = tri_centroids[simplex[0], :]
        p2 = tri_centroids[simplex[1], :]
        p3 = tri_centroids[simplex[2], :]

        # Calculate edge lengths for this triangle
        e12 = np.sqrt( (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 )
        e13 = np.sqrt( (p3[0]-p1[0])**2 + (p3[1]-p1[1])**2 )
        e23 = np.sqrt( (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2 )

        # Calculate area for this triangle
        s = (e12 + e13 + e23) / 2
        a = np.sqrt( s * (s-e12) * (s-e13) * (s-e23))
        triangle_areas.append(a)
        
    return triangle_areas

def extract_architecture_features():
    pass

