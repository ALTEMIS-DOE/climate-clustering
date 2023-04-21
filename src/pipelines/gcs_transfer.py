import os
import glob
import time
import argparse
import logging
from google.cloud import storage 

from downloader import climate_downloader 
from downloader import parse_args 

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    FLAGS = parse_args(True)
    savedir = os.path.join( *[FLAGS.savedir, FLAGS.model, FLAGS.scenario, FLAGS.level] )
    CD = climate_downloader(model=FLAGS.model, experiment_id=FLAGS.scenario)

    # batch data transfer for atmos data
    s1 = time.time()
    CD.batch_transfer(FLAGS.bucket_id, savedir, f"nex-dcp30/{FLAGS.scenario}/{FLAGS.level}/{FLAGS.model}/" )
    s2 = time.time()
    logger.info(f"### Transfer took { (s2-s1)/60.0 } minutes ###")