"""Shapely manipulations of the DSA annotations.

This package contains functions to aid in processing DSA annotations as Shapely objects.
"""

from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, Point
from sunycell import dsa
import numpy as np

def get_polygons_from_elements(target_elements: list) -> MultiPolygon:
    """Return Shapely polygons from a list of DSA elements."""

    # Create a polygon for each object in target_elements
    polygons = []
     
    # Cycle through each target to make a list of polygons
    for target_element in target_elements:
        # If the element has no 'points' key, is not a polygon
        if 'points' not in target_element.keys():
            # Maybe it's a rectangle?
            if target_element['type'] == 'rectangle':
                left = target_element['center'][0]-target_element['width']/2.0
                right = left + target_element['width']
                top = target_element['center'][1]-target_element['height']/2.0
                bot = top + target_element['height']
                pts = [[left, top], [right, top], [right, bot], [left, bot], [left, top]]
            else:
                #print(f'Skipping element with type: {target_element["type"]}')
                continue
        else:
            # We have a polygon, so extract the points directly
            pts = target_element['points']
            
            # Filter out the Z-stack coordinate
            pts = [x[:2] for x in pts]

        # Sometimes we do not have a proper polygon (>3 points), so skip if that's the case
        if len(pts) <= 3:
            continue

        # Append a shapely polygon to our list made up of these points
        polygons.append(Polygon(pts))

    # Make polygons valid by buffering them
    polygons = [p.buffer(0) for p in polygons]

    # Merge polygons that might be overlapping
    polygons = unary_union(polygons)

    # We want polygons to be a MultiPolygon, even if it only contains one object
    if type(polygons) is not MultiPolygon:
        polygons = MultiPolygon([polygons])
    
    return polygons


def get_polygons(gc, slide_id, group_list, target_mpp):
    """Interface for getting polygons using DSA info."""

    target_elements, _ = dsa.slide_elements(gc,
                                            slide_id,
                                            target_mpp=target_mpp,
                                            group_list=group_list)
    
    # TODO: Error checking here

    return get_polygons_from_elements(target_elements)


def get_polygon_grid_coords(polygons, tile_size: int):
    """Given a polygon, return a grid of non-overlapping points across the polygon."""

    # Assuming the polygon is a Multi:
    minx, miny, maxx, maxy = polygons.bounds

    # Create a grid across this area
    area_width = int(maxx) - int(minx)
    area_height = int(maxy) - int(miny)
    
    columns = int(np.ceil(area_width / tile_size))
    rows = int(np.ceil(area_height / tile_size))
    
    top = miny
    bottom = top + rows * tile_size
    left = minx
    right = left + columns * tile_size
    
    left_coords = np.arange(minx, right, tile_size)
    top_coords = np.arange(miny, bottom, tile_size)
    
    # Construct a mesh of points in the bounding box of the polygon
    tile_polygons = []
    for left_coord in left_coords:
        for top_coord in top_coords:
            # Construct a rectangle to ensure tile is entirely inside the polygon
            tile_shape = Polygon([
                [left_coord, top_coord],
                [left_coord+tile_size, top_coord],
                [left_coord+tile_size, top_coord+tile_size],
                [left_coord, top_coord+tile_size]
            ])
            
            # If tile shape is entirely inside the annotated region, then add it to the list
            if tile_shape.within(polygons):
                tile_polygons.append(tile_shape)

    return tile_polygons
