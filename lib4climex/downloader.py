import logging

from google.cloud import storage # pip install google-cloud-storage

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def download(bucket_id:str, sourcepath:str, savepath:str):
    """ function to single file download from GCP
        :param bucket_id: string bucket name of S3 
        :param sourcpath: directory path to a file in an GCP instance under the bucket name
        :param savepath:  save directory path on a GCP instance including the filename.  
        :return: None
    """
    # Instantiates a client
    client = storage.Client()

    # google.cloud.storage.batch.Batch
    bucket = client.get_bucket(bucket_id)
    blob = bucket.blob(sourcepath)
    try:
        blob.download_to_filename(savepath)
    except Exception as e:
        logger.info(e)