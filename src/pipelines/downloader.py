# atmos download from aws
import os
import glob
import json
import time
import argparse
import logging
import subprocess
import multiprocessing

import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import boto3 # pip install boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config

from google.cloud import storage # pip install google-cloud-storage

#logging.basicConfig(level = logging.DEBUG)
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class climate_downloader(object):

    def __init__(self, BASECMD="wget --timeout 0 --tries=20", 
                savedir=None,
                model=None, experiment_id=None, ):
        logger.info("")
        self.BASECMD = BASECMD 
        self.savedir = savedir
        self.model   = model.lower()
        self.experiment_id = experiment_id.lower()
        pass

    def wget_downloader(filename:str, savedir:str, BASECMD:str):
        """ function to execute download data via wget from aws 
            :param filename: string filename of target downloadable file with full http/https path 
            :param savedir:  directory path to a local/remote storage 
            :param BASECMD:  wget command line with option. this should be one string
            :return: None
        """
        CMD = " ".join([BASECMD, filename,"-P",  savedir])
        FNULL = open(os.devnull, 'w')
        proc = subprocess.Popen(CMD, shell=True,stdout=FNULL )
        proc.wait() # add wait in purpose to prevent dominant thread/process 

    #def aws_downloader(self, resource,filename:str, bucket_name:str, savedir:str, prefix:str):
    def aws_downloader(self, filename:str, bucket_name:str, savedir:str, prefix:str):
        """function to run boto3 command to download data from S3 bucket 
            
            :param resource: boto3.resource object
            :param filename: string filename of target downloadable file with full http/https path 
            :param bucket_name: string bucket name of S3 
            :param savedir:  directory path to a local/remote storage 
            :param prefix:   http://bucket_name.s3.amazonaws.com/
            :return: None

            Example:
            resource = boto3.resource('s3',config=Config(signature_version=UNSIGNED))
            resource.Bucket("nasanex").download_file(
            'NEX-DCP30/BCSD/rcp85/mon/atmos/pr/r1i1p1/v1.0/CONUS/pr_amon_BCSD_rcp85_r1i1p1_CONUS_HadGEM2-CC_201101-201512.nc', 
            './pr_amon_BCSD_rcp85_r1i1p1_CONUS_HadGEM2-CC_201101-201512.nc')
        """
        # boto3
        resource = boto3.resource('s3',config=Config(signature_version=UNSIGNED))
        basefilename=os.path.basename(filename)
        resource.Bucket(bucket_name).download_file(filename[len(prefix):], os.path.join(savedir, basefilename))

    def aws_downloader_wrapper(self, args):
        """wrapper for multiprocessing
        """
        return self.aws_downloader(*args)

    def run(self, ncores, values):
        # multiprocess
        ncpus = ncores if ncores <= multiprocessing.cpu_count() else  multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(ncpus)
        p.map(self.aws_downloader_wrapper, values)

    def json_decoder(self, jsonfilename:str):
        """function to decode json file and apply extractor to get downloadable filenames

            e.g. an example of json file contents. 
                 A key-value of given json is a downloadable filename and their attributes 
            key:
                 http://nasanex.s3.amazonaws.com/NEX-DCP30/BCSD/rcp60/mon/atmos/tasmax/r1i1p1/v1.0/CONUS/tasmax_amon_BCSD_rcp60_r1i1p1_CONUS_NorESM1-M_202101-202512.nc
            value:
            {
                'year_start': '2021', 
                'year_end': '2025', 
                'experiment_id': 'rcp60', 
                'variable': 'tasmax', 
                'model': 'noresm1-m', 
                'md5': '79a6abd88b7d4a03da1d03b5ffce8ec3'
            }

        """
        with open(jsonfilename) as f:
            json_obj = json.load(f)

        filenames = [ikey for ikey in json_obj.keys() ]
        filelist = []
        #filelist  = list(map(lambda x: self.extractor(x, model=self.model,experiment_id=self.experiment_id), filenames ))
        for filename in filenames:
            data = json_obj[filename]
            if data['model'] == self.model:
                if data['experiment_id'] == self.experiment_id:
                    filelist.append(filename) 
        logging.info(f" Filter {len(filelist)} files to be downloaded : Model == {self.model} | Scenario == {self.experiment_id}")
        return filelist


    def batch_transfer(self, bucket_id:str, sourcedir:str, destpath:str):
        """function to batch transfer the entire directory
            :param bucket_id: string bucket name of S3 
            :param sourceir:  directory path to a file in an GCP instance 
            :param destpath:  path under the bucket name
            :return: None
        """
        # Instantiates a client
        client = storage.Client()

        # google.cloud.storage.batch.Batch
        bucket = client.get_bucket(bucket_id)
        try:
            with client.batch():
                for ifile in glob.glob(os.path.join(sourcedir, '*.nc')) : 
                #try:
                    blob = bucket.blob(destpath + os.path.basename(ifile))
                    blob.upload_from_filename(ifile)
        except Exception as e:
            logger.info(e)
            pass
    
    def download(self, bucket_id:str, sourcepath:str, savepath:str):
        """ TODO test should be done
            function to single file download from GCP
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


def parse_args(verbose=False):
    """
    workers_per_node: Number of workers started per node, 
                      which corresponds to the number of tasks 
                      that can execute concurrently on a node.
    nodes_per_block: Number of nodes requested per block
    """
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--savedir",     type=str,  default='/home/tkurihana_uchicago_edu/climate-data', help='base save directory name')
    p.add_argument("--ncores",      type=int,  default=2, help='number of cores per node')
    p.add_argument("--model",       type=str,  default="GFDL-ESM2G", help='model name in NEX-DCP30')
    p.add_argument("--scenario",    type=str,  default="rcp85",      help='cmip5 scenario rcp26, rcp45, rcp60, rcp85, historical are option')
    p.add_argument("--bucket_name", type=str,  default="nasanex",    help='bucket name')
    p.add_argument("--prefix",      type=str,  default="http://nasanex.s3.amazonaws.com/",    help='s3 url')
    p.add_argument("--bucket_id",   type=str,  default="",    help='S3 bucket name')
    p.add_argument("--level",       type=str,  default="",    help='either atmos or land')
    FLAGS = p.parse_args()
    if verbose:
        for f in FLAGS.__dict__:
            print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
        print("\n")
    return FLAGS

if __name__ == "__main__":
    FLAGS = parse_args(True)

    # checker
    if len(FLAGS.level) ==0 :
        logger.info("Specify --level option: either 'atmos' or 'land' ")
        exit(0)

    # make directory
    savedir = os.path.join( *[FLAGS.savedir, FLAGS.model, FLAGS.scenario,FLAGS.level] )
    os.makedirs(savedir, exist_ok=True)

    # call climate downloader class
    CD = climate_downloader(model=FLAGS.model, experiment_id=FLAGS.scenario)
    # get list of data
    filelist = CD.json_decoder("./nex-dcp30-s3-files.json")
    fs = sorted(filelist)

    # start downloading
    s1 = time.time()
    values = [ (ifile, FLAGS.bucket_name, savedir, FLAGS.prefix) for ifile in fs[4:] ] # set args
    CD.run(FLAGS.ncores, values) # multiprocessing within class
    s2 = time.time()
    logging.info(f"### Download took { (s2-s1)/60.0 } minutes ###")

    # batch data transfer for atmos data
    s1 = time.time()
    CD.batch_transfer(FLAGS.bucket_id, savedir, f"nex-dcp30/{FLAGS.scenario}/{FLAGS.level}/{FLAGS.model}/" )
    s2 = time.time()
    logger.info(f"### Transfer took { (s2-s1)/60.0 } minutes ###")

        