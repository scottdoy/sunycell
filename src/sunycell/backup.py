"""Digital Slide Archive annotation backups.

This package allows for one-button download of all annotations currently on the server.
Requires just a connection to the server, and will either:

- Automatically recurse through all the collections, folders, and sub-folders to create a comprehensive backup;
- Obtain a target collection and download all annotations in that collection.

@scottdoy
"""
import argparse
from datetime import datetime
from dotenv import load_dotenv
from histomicstk.annotations_and_masks.annotation_database_parser import (
    dump_annotations_locally)
import os
from pathlib import Path
from sunycell import dsa


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--collections', default='TCGA',
        help="Comma-separated collection names on the DSA")
parser.add_argument('--folder', default=None,
        help="Folder name on the DSA")
parser.add_argument('--env_path', default='.env',
        help="Path to the secrets file; must contain 'APIKEY' and 'APIURL' at a minimum")
parser.add_argument('--backup_dir', default='data/backups',
        help="Local path to save the backups")


def backup_folder(gc, folder, config):
    """Perform the download of a folder."""

    # Set up the local path to save the annotations to
    folder_dir = config['save_dir'] / folder['name']
    folder_dir.mkdir(exist_ok=False)
    
    # recursively save annotations -- JSONs + sqlite for folders/items
    dump_annotations_locally(
        gc, folderid=folder['_id'], local=folder_dir,
        save_json=True, save_sqlite=True)

if __name__=="__main__":
    args = parser.parse_args()
    config = dict()

    config['collections'] = str(args.collections).split(',')
    config['folder'] = str(args.folder)
    config['env_path'] = Path(args.env_path)
    config['backup_dir'] = Path(args.backup_dir)

    # Set up the time for running the job
    time_prefix = datetime.now().isoformat()

    # Connect to SUNYCell
    load_dotenv(config['env_path'])
    gc = dsa.dsa_connection(api_key=str(os.getenv("APIKEY")), api_url=str(os.getenv("APIURL")))

    # This is where the annotations and sqlite database will be dumped locally
    for collection_name in config['collections']:
        print(f'Processing {collection_name}...')
        collection_id = dsa.get_collection_id(conn=gc, collection_name=collection_name)
        
        # Set up the path to this collection
        config['save_dir'] = Path(f'./{config["backup_dir"]}/{time_prefix}/{collection_name}')
        config['save_dir'].mkdir(parents=True, exist_ok=True)

        # Get the list of folders that we would like to save for (everything)
        folder_list = list(gc.listFolder(collection_id, parentFolderType='collection'))

        for folder in folder_list:
            backup_folder(gc, folder, config)

