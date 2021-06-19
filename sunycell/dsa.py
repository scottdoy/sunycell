"""
This package contains helper functions for interfacing with the Digital Slide Archive.
Most of these functions are wrappers or helpers for the histomicstk library.
"""

#from .htk_utils import get_histomics_connection

#__all__ = (
    #'get_histomics_connection'
#)

import girder_client
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_scale_factor_and_appendStr,
    get_image_from_htk_response,
    get_bboxes_from_slide_annotations,
    scale_slide_annotations
)

def dsa_connection(api_url: str, api_key: str) -> girder_client.GirderClient:
    """Connect to a DSA server.
    
    Parameters
    ----------
    api_url : string
        URL to the API endpoint.
    
    api_key : string
        API key for the DSA server.

    Returns
    -------
    gc : girder_client.GirderClient
        An authenticated Girder client object.
    """
    gc = girder_client.GirderClient(apiUrl=api_url)
    gc.authenticate(apiKey=api_key)
    return gc

def get_collection_id(collection_name: str, conn: girder_client.GirderClient) -> str:
    """Given a connection, grab the id of the target collection."""
    collection_id = None

    # List all collections and find the target one
    collection_list = conn.listCollection()
    #assert len(collection_list) > 0, f"Cannot find collection named {collection_name} on Histomics, please check that the server connection is working, that you have access to the collection, and that you are spelling everything correctly."
    for collection in collection_list:
        if collection['name'] == collection_name:
            collection_id = collection['_id']
    assert collection_id, f"Connected to server, but could not find collection named {collection_name}"

    return collection_id

def get_folder_id(folder_name: str, conn: girder_client.GirderClient, collection_id: str) -> str:
    """Given a folder name, connection, and collection ID number, return the folder ID number."""

    folder_id = None

    # List all collections and find the target one
    folder_list = conn.listFolder(collection_id, parentFolderType='collection')
    #assert len(folder_list) > 0, f"Cannot find folder list for collection id {collection_id} on Histomics. Please check that the server connection is working, that you have access to the collection, and that you are spelling everything correctly."
    for folder in folder_list:
        if folder['name'] == folder_name:
            folder_id = folder['_id']
    assert folder_id, f"Connected to server and found some folders, but could not find folder named {folder_name}"

    return folder_id

def get_sample_id(conn, sample_name, collection_name, folder_name):
    '''Given a connection, collection & folder combo, and sample name, return the sample ID number.

    We assume that there are no nested folders -- it goes collection /
    folder_list / item_list
    '''

    id_list, item_list = ids_names_from_htk(conn, collection_name, folder_name)

    for (id, item) in zip(id_list, item_list):
        if item == sample_name:
            return id
    print(f'Did not find sample {sample_name} in {collection_name} / {folder_name}. Please recheck your access and spelling of the path.')
    return None

def image_metadata(conn, sample_id):
    resp = conn.get(f'/item/{sample_id}/tiles')
    return resp

def ids_names_from_htk(conn, collection_name, folder_name):
    with conn.session() as session:
        collection_id = get_collection_id(collection_name, conn)
        folder_id = get_folder_id(folder_name, conn, collection_id)
        item_list_htk = conn.listItem(folder_id)

        # Parse the retrieved item list
        item_ids = []
        item_names = []
        for item in item_list_htk:
            item_ids.append(item['_id'])
            item_names.append(item['name'])

    return item_ids, item_names

def slide_annotations(conn, slide_id, target_mpp, log=None, group_list=None):
    """Return a single slide's annotations.
    Accepts an optional annotation_list parameter indicating the groups to get.
    If not provided, grab everything. 
    """
    # Case-insensitive group list!
    gl = [x.lower() for x in group_list]

    # Pull down the annotation objects
    try:
        annotations_resp = conn.get('/annotation/item/' + slide_id)
    except girder_client.HttpError:
        # The server couldn't find the object
        if log is not None:
            log.warning(f'Could not find an item on the server with id {slide_id}.')
        else:
            print(f'Could not find an item on the server with id {slide_id}.')
        return None, None, None
    except:
        # Unclear why this failure happened
        if log is not None:
            log.warning(f'Something went wrong getting annotations for {slide_id}.')
        else:
            print(f'Something went wrong getting annotations for {slide_id}.')
        return None, None, None

    # Do they exist? If not, return false
    if len(annotations_resp) == 0:
        log.warning(f'No annotations were found for {slide_id}.')
        return None, None, None
    
    # Get the scale factor and string for this slide
    scale_factor, appendStr = get_scale_factor_and_appendStr(conn, slide_id, MPP=float(target_mpp), MAG=None)

    # Scale the annotations according to the desired MPP
    _ = scale_slide_annotations(annotations_resp, scale_factor)

    # Get the info of the elements based on the now-scaled annotations
    element_infos = get_bboxes_from_slide_annotations(annotations_resp)
    element_infos = element_infos[element_infos['group'].isin(gl)]
                    
    return element_infos, scale_factor, appendStr 

def slide_elements(conn, item_id, group_list=None):
    """Retrieve a list of elements from the HTK annotation response.

    Each element in this list corresponds to a polygon.

    Optionally, ask for elements that belong to a specific group or list of groups."""
    # Pull down the annotation objects
    annotations_resp = conn.get('annotation/item/' + item_id)
    img_metadata = conn.get('item/' + item_id + '/tiles')
    #img_name = conn.get('item/' + item_id)['name']

    if len(annotations_resp) == 0:
        print('The annotation response had a length of Zero')
        return [], []
    
    # Initialize the gt annotation holder
    target_elements = []

    # Cycle through each annotation on this item
    for annotation in annotations_resp:
        elements = annotation['annotation']['elements']
        # Cycle through each of the annotation elements
        for element in elements:
            # Check that this item has a group (i.e. a class)
            if 'group' in element.keys():
                # If this group is what we're looking for, then pull it
                if group_list is not None and element['group'] in group_list:
                    target_elements.append(element)
                elif group_list is None:
                    target_elements.append(element)
           
    return target_elements, img_metadata


def image_data(conn, sample_id, bounds_dict, appendStr=None):
    """Return a numpy image defined by the connection, sample_id, and ROI."""

    # Convert the keys of the bounds_dict to lowercase
    bounds_dict = {k.lower():v for k,v in bounds_dict.items()}

    # Ensure the keys are present for the bounding box
    assert 'xmin' in bounds_dict and 'xmax' in bounds_dict and 'ymin' in bounds_dict and 'ymax' in bounds_dict, f'bounds_dict is not formatted properly, please make sure xmin, xmax, ymin, ymax is included in the keys.'

    getStr = f"item/{sample_id}/tiles/region?"+ \
        f"left={bounds_dict['xmin']}&"+ \
        f"right={bounds_dict['xmax']}&"+ \
        f"top={bounds_dict['ymin']}&"+ \
        f"bottom={bounds_dict['ymax']}"
    
    if appendStr is not None:
        getStr += appendStr

    # Get the image raw response
    try:
        resp = conn.get(getStr, jsonResp=False)
    except:
        print(f'Could not retrieve the image response using {getStr}, returning None')
        return None
    
    # Sometimes this fails, and I'm not sure why
    try:
        img_roi = get_image_from_htk_response(resp)
    except:
        print(f'Could not convert the image response to image, returning None')
        return None
   
    return img_roi