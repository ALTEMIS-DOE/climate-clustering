import os
import glob
import logging

from google.cloud import storage # pip install google-cloud-storage

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def batch_transfer(bucket_id:str, sourcedir:str, destpath:str, ext:str):
    """function to batch transfer the entire directory
        :param bucket_id: string bucket name of S3 
        :param sourceir:  directory path to a file in an GCP instance 
        :param destpath:  path under the bucket name
        :return: None
    """
    # Instantiates a client
    client = storage.Client()

    if len(ext) == 0:
        ext = "*"

    # google.cloud.storage.batch.Batch
    bucket = client.get_bucket(bucket_id)
    try:
        with client.batch():
            for ifile in glob.glob(os.path.join(sourcedir, f'*.{ext}')) : 
            #try:
                blob = bucket.blob(destpath + os.path.basename(ifile))
                blob.upload_from_filename(ifile)
    except Exception as e:
        logger.info(e)
        pass
