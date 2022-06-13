from matplotlib.path import Path as mplPath
import matplotlib.tri as T
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import stats
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from skimage import morphology, segmentation


def get_polygon_from_pts(pts):
    polygons = []
    for pt in pts:
        X = pt[0]
        Y = pt[1]
        point_list = [(x, y) for x, y in zip(X, Y)]
        poly = Polygon(point_list)
        polygons.append(poly)
    return polygons


def get_polygon_from_mask(mask, offset_matrix=[1, 0, 0, 1, 0, 0]):
    """Given a mask and an offset matrix, compute a MultiPolygon consisting
    of the objects in the mask.

    Original method from here:
    https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/

    Offset matrix is used to adjust the coordinates of the polygons after
    creation:
    https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.affine_transform
    """
    all_polygons = []

    # rasterio.features.shapes() will generate shapes and values from the
    # array in the first argument.
    # Since we aren't using the values here, the first argument can be
    # anything (we use the mask), but the second needs to be the boolean
    # mask that defines the geometry.
    # The "transform=rasterio.Affine()" defines the transformation that is
    # applied to the geometry;
    # Here, we use [1.0, 0, 0, 0, 1.0, 0] but this is the default valuei
    # (we can leave this blank).
    for shape, value in features.shapes(mask.astype(np.int16),
                                        mask=(mask > 0),
                                        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

        # Using "shapely.geometry.shape()" here allows shapely to decide
        # what kind of object to create; however, rasterio.features.shapes()
        # will create a polygon, so we could just use Polygon here.
        all_polygons.append(shapely.geometry.shape(shape))

    # Construct an initial MultiPolygon from this list
    all_polygons = MultiPolygon(all_polygons)

    # Check to see if this is a "valid" MultiPolygon
    if not all_polygons.is_valid:
        # If not, apply the 0 buffer trick
        all_polygons = all_polygons.buffer(0)

        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])

    # Apply the affine transform to adjust the coordinates of the polygon
    # accordingly
    all_polygons = affine_transform(all_polygons, offset_matrix)

    return all_polygons


def get_edge_coordinates(elements):
    # Obtaining vaid cordinates Satellite/Tumor region
    coord = []
    for i, element in enumerate(elements):
        # If the element is not a polygon, it may not contain "points".
        # If so, skip it.
        if 'points' not in element.keys():
            continue
        points = element['points']
        points = [x[:-1] for x in points]

        X = np.array([int(p[0]) for p in points], dtype=np.int64)
        Y = np.array([int(p[1]) for p in points], dtype=np.int64)

        coord.append([X, Y])

    return coord


def get_centroid_coordinates(elements):
    """Given a set of htk elements, calculate the centroid coordinate for
    each polygon therein."""

    coord = []
    for element in elements:
        points = element['points']
        points = [x[:-1] for x in points]

        X = np.array([int(p[0]) for p in points], dtype=np.int64)
        Y = np.array([int(p[1]) for p in points], dtype=np.int64)

        coord.append([X.mean(), Y.mean()])

    return coord


def cut_triangles(tri, coordinates, verbose=True):
    """Eliminate triangles whose branches cross the object defined by
    coordinates """

    # every triangle in the delaunay object that contains these points
    eliminate_triangles = tri.find_simplex(coordinates)
    if verbose:
        print(f'Number of points that were tested: {len(eliminate_triangles)}')

    # eliminates duplicate
    eliminate_triangles = np.unique(eliminate_triangles[eliminate_triangles > -1])
    if verbose:
        print(f'List of unique, non-negative simplex coordinates: {eliminate_triangles}')

    # Creates a copy of triangles
    tri_simplices = tri.simplices.copy()
    tri_simplices = np.delete(tri_simplices, eliminate_triangles, axis=0)
    return tri_simplices


def get_poly_centroid(poly):
    """Gets the centroid of the polygon and processes it to be a numpy
    array."""
    pass


def get_poly_boundaries(poly):
    """Gets the boundaries of the polygon and processes it to be a numpy
    array."""
    pass


def descriptive_stats(x, feature_prefix=''):
    """Create a pandas dataframe containing the features calculated on x.

    x is an input array for which the features below are calculated.

    feature_prefix is prepended to each of the feature names to create
    unique columns."""

    feature_names = [
        'minimum', 'maximum', 'mean', 'variance', 'standard_deviation',
        'skewness', 'kurtosis', 'moment_5', 'moment_6', 'moment_7',
        'moment_8', 'moment_9', 'moment_10', 'moment_11',
        'geometric_mean', 'harmonic_mean'
    ]
    feature_names = [feature_prefix+x for x in feature_names]

    # Check to see if input array is None
    if x is None:
        # If so, construct a dataframe using NAs to represent the features
        features = [np.nan for _ in range(16)]
    else:
        # Otherwise, actually calculate the features
        # Ensure x is a numpy array
        x = np.array(x)

        minimum = np.amin(x)
        maximum = np.amax(x)
        mean = np.mean(x)
        variance = np.var(x)
        standard_deviation = np.std(x)
        skewness = stats.skew(x)
        kurtosis = stats.kurtosis(x)
        moment_5 = stats.moment(x, moment=5)
        moment_6 = stats.moment(x, moment=6)
        moment_7 = stats.moment(x, moment=7)
        moment_8 = stats.moment(x, moment=8)
        moment_9 = stats.moment(x, moment=9)
        moment_10 = stats.moment(x, moment=10)
        moment_11 = stats.moment(x, moment=11)

        # mean values require x > 0, so we will scale these values appropriately
        geometric_mean = stats.gmean(x - x.min() + 1)
        harmonic_mean = stats.hmean(x - x.min() + 1)
        features = [minimum, maximum, mean, variance, standard_deviation,
                    skewness, kurtosis, moment_5, moment_6, moment_7,
                    moment_8, moment_9, moment_10, moment_11,
                    geometric_mean, harmonic_mean]
    features_dict = {}
    for feature_name, feature in zip(feature_names, features):
        features_dict[feature_name] = feature
    return pd.DataFrame([features_dict])


def assign_wave_index(tum_bin, sat_bounds, max_dilations=1500):
    tum_counter = 0
    sat_wave_number = np.zeros((len(sat_bounds), 1))

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

            sat_bound = np.array([sat_bound[:, 0], sat_bound[:, 1]]).T
            tum_poly = mplPath(tum_bin_points)
            sat_hit = tum_poly.contains_points(sat_bound)
            if np.any(sat_hit is True) and sat_wave_number[sat_idx] == 0:
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
    sat_wave_numbers = np.zeros((len(sat_bounds), 1))

    while True:
        # Dilate the tumor
        # tum_bin = morphology.binary_dilation(tum_bin)

        # Increment the counter
        tum_counter += 1

        # Check to see if any satellites are "hit" by the (dilated) tumor
        # Get dilated tumor boundary points
        # img_tum_boundary = segmentation.find_boundaries(tum_bin)
        # boundary_tum = np.nonzero(img_tum_boundary)

        # Split apart boundary coordinates
        # boundary_tum_x = boundary_tum[0]
        # boundary_tum_y = boundary_tum[1]
        # tum_bin_points = np.array([boundary_tum_x, boundary_tum_y]).T
        tum_bin_points = tum_bounds * 2

        # Get satellite wave number
        for sat_idx, sat_bound in enumerate(sat_bounds):

            sat_bound = np.array([sat_bound[0], sat_bound[1]]).T
            tum_poly = mplPath(tum_bin_points)
            sat_hit = tum_poly.contains_points(sat_bound)
            if np.any(sat_hit is True) and sat_wave_numbers[sat_idx] == 0:
                sat_wave_numbers[sat_idx] = tum_counter

        # Check to see if every satellite has been hit
        if np.all(sat_wave_numbers > 0):
            return sat_wave_numbers

        # Make sure we aren't in an infinite loop
        if tum_counter > max_dilations:
            print(f"Not all sats have been assigned an index after {max_dilations} iterations. Exiting.")
            return sat_wave_numbers


def compute_wave_dict(mt_polygon, sat_polygons, max_iters=10000):
    """Compute a wave dictionary, where the keys are satellite indices and
    the values are the wave numbers.

    Used by the wave graph feature extraction function.
    """

    # Create a list of all the wave numbers for the corresponding satellites
    wave_dict = dict()

    # Initialize growing MT object
    mt_buffer = mt_polygon

    # Set the first index of wave_dict equal to 0
    wave_dict[0] = 0

    for wave_number in trange(10000):
        # If we've assigned a wave number to each satellite, then quit
        if len(wave_dict) >= len(sat_polygons)+1:
            break

        mt_buffer = mt_buffer.buffer(100)

        # Create a query tree of the current polygon queue
        sat_strtree = STRtree(sat_polygons)

        # Create a dictionary to get the indices of the retrieved results
        index_by_id = dict((id(poly), i) for i, poly in enumerate(sat_polygons, start=1))

        # Compute the result of the query
        query_result = [index_by_id[id(poly)] for poly in sat_strtree.query(mt_buffer) if poly.intersects(mt_buffer)]

        # Iterate through the polygons
        # Start at 1, because 0 is the main tumor
        for result_id, sat_polygon in enumerate(sat_polygons, start=1):
            # First, only process this polygon if it is not currently in the
            # wave dictionary
            if result_id not in wave_dict.keys():

                # Check to see if this polygon appears in the query results
                if result_id in query_result:
                    # This polygon should be assigned the current wave number
                    wave_dict[result_id] = wave_number
    return wave_dict


def compute_wave_distances(mt_polygon, sat_polygons, wave_dict):
    """Perform the wave distance computation, given the satellite polygons
    and wave dictionary.

    Compute the wave dictionary with compute_wave_dict().
    """
    # Create a list that will hold all the line segments
    wave_distances = []
    # wave_lines = []

    # Iterate through the satellite list
    for sat_idx, sat_polygon in enumerate(sat_polygons, start=1):
        # For the current polygon, get a list of all polygons that are lower
        # wave number
        lower_polygons = []
        lower_polygons = [(idx, sat_poly) for idx, sat_poly in enumerate(sat_polygons, start=1) if wave_dict[idx] < wave_dict[sat_idx]]

        # Include the tumor in this!
        lower_polygons += [(0, mt_poly) for mt_poly in mt_polygon.geoms]

        # If lower_polygons is empty (len 1, for just the main tumor), that
        # means the current satellite has no others with a lower wave number.
        # Thus, connect this satellite to the main tumor
        if len(lower_polygons) == 1:
            print(f'Satellite {sat_idx} has no lower satellites, so connecting it with main tumor.')
            nearest_point = nearest_points(sat_polygon, mt_polygon)

            # Save distance and point-pair
            wave_distances.append(sat_polygon.distance(nearest_point[1]))
            # wave_lines.append(nearest_point)
            continue

        # Iterate through lower_polygons to find the distances between the
        # current satellite and each lower_polygon
        lower_multipolygon = MultiPolygon([lower_poly[1] for lower_poly in lower_polygons])
        nearest_point = nearest_points(sat_polygon, lower_multipolygon)

        # Store the distance values for feature calculation
        wave_distances.append(sat_polygon.distance(nearest_point[1]))

        # Store the actual point pairs for display purposes
        # wave_lines.append(nearest_point)

    return wave_distances


def get_triangle_lengths(tri_centroids, tri_simplices):

    t = T.Triangulation(tri_centroids[:, 0], tri_centroids[:, 1], tri_simplices)
    triangle_lengths = []

    for edge in t.edges:
        x1 = tri_centroids[edge[0], 0]
        x2 = tri_centroids[edge[1], 0]
        y1 = tri_centroids[edge[0], 1]
        y2 = tri_centroids[edge[1], 1]
        triangle_lengths.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))

    return triangle_lengths


def get_triangle_areas(tri_centroids, tri_simplices):
    '''Calculate area of triangles given a set of centroids and a set of
    simplices.'''

    triangle_areas = []
    for simplex in tri_simplices:
        # Pull out the points for this triangle
        p1 = tri_centroids[simplex[0], :]
        p2 = tri_centroids[simplex[1], :]
        p3 = tri_centroids[simplex[2], :]

        # Calculate edge lengths for this triangle
        e12 = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        e13 = np.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
        e23 = np.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)

        # Calculate area for this triangle
        s = (e12 + e13 + e23) / 2
        a = np.sqrt(s * (s-e12) * (s-e13) * (s-e23))
        triangle_areas.append(a)

    return triangle_areas


def extract_architecture_features():
    pass
