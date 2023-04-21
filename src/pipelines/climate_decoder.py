# 1. Migrate ilijana's regional data processing script from notebook.
# --------- After this 30mis session --------------
# 2. Takuya will make a structure of this python script & 
# 3. Takuya pushes the script to gitlab
# 4. ilijana pulls my script
# 5. ilijana finalizes the code (check the code if this is ok or not)
# 6. Takuya check and ok!


import os
import glob
import netCDF4
import logging
import numpy as np
import pandas as pd
from google.cloud import storage
from collections import defaultdict
from typing import List 

__author__ = 'digital-twin climate subgroup'


logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)


class regional_data_processing(object):
    """ thi
    """

    def __init__(self, bucket_id:str, model:str, savedir:str, 
                    scenarios=['rcp26', 'rcp45', 'rcp60', 'rcp85'],
                    time=['historical', 'mid_century','late_century'], 
                    operation='mean'):
        # put parameter you want to control from outside of code
        # To ilijana: !here if we specify, we can use self.model under class 
        self.model = model              
        self.bucket_id = bucket_id
        self.savedir   = savedir # name of directory where you want to save CSV file data
        self.scenarios = scenarios
        self.time      = time
        self.WUNIT = 4.743e-06 # kg-water/m^2/sec
        self.CUNIT = 0.15*1e3/12 # m/year --> mm/month
        self.mhist1=12*(50+21)    # 2021
        self.mhist2=12*(50+21+40) # 2061
        self.operation = operation

    def load_dataset(self, fn, varname) -> np.ma.core.MaskedArray:
        """open netcdf file, check variables, and check dimensions (with filename) 
            
            for example, 
                fn=f'/home/jupyter/{model}/regional/Extraction_pr.nc'
                varname = 'pr'

                Then, you will get preciptation data from the netCDF file
        """
        logging.info(f"\n Open {fn} ")
        ds = netCDF4.Dataset(fn)
        var = ds.variables[varname][:]
        # check variable name and shape
        logging.info(f"{varname} {var.shape} \n")
        return var
        
    def split_data(self, var, from_here:str, until_here:str) -> np.ma.core.MaskedArray :
        # we need varibale to use for diving time series
        # at the same time, we will take spatial mean
        # we need to get variable data  as argument
        logging.info(f"split variable from time={from_here} to {until_here}")
        data=var[:, from_here:until_here].mean(axis=(2,3))

        # check the shape of variables
        logging.info(f"{data.shape} \n")
        return data

    def _convert_unit(self, climate_data):
        return self.WUNIT/self.CUNIT*climate_data

    def calculate_median(self, precip, et, ) -> List:
        rchs_list = []
        for i in range(len(self.scenarios)):        
            rchs  = self._convert_unit( precip[i] - et[i] )
            rchs_median = np.median(rchs)
            rchs_list.append(rchs_median)
        return rchs_list

    def calculate_mean(self, precip, et, ) -> List:
        rchs_list = []
        for i in range(len(self.scenarios)):        
            rchs  = self._convert_unit( precip[i] - et[i] )
            rchs_mean = np.mean(rchs)
            rchs_list.append(rchs_mean)
        return rchs_list

    def calculate_recharge(self, 
                            precip_hists, et_hists, 
                            precip_sms1, et_sms1,
                            precip_sms2, et_sms2,
                            operation='mean', csvname='rchs1_hists.csv') -> pd.core.frame.DataFrame:
        # we need list of scenarios= [historical, mid_century, late_century]
        # we need list of RCPs= []
        # precip and ET values 
        rchs_list = []
        if operation == 'mean':
            rchs_list += self.calculate_mean(precip=precip_hists, et=et_hists)
            rchs_list += self.calculate_mean(precip=precip_sms1, et=et_sms1)
            rchs_list += self.calculate_mean(precip=precip_sms2, et=et_sms2)
        elif operation == 'median':
            rchs_list += self.calculate_median(precip=precip_hists, et=et_hists)
            rchs_list += self.calculate_median(precip=precip_sms1, et=et_sms1)
            rchs_list += self.calculate_median(precip=precip_sms2, et=et_sms2)
        else:
            raise ValueError(f"Operation option is not defined {operation}")
        
        # save data as pandas --> return is pandas or save as csv here
        df = pd.DataFrame(rchs_list)
        df['time']= [ self.time[j] for j in range(len(self.time)) for i in range(len(self.scenarios)) ] #For ilihana [hist, mid, late] --> [hist, hist, ..., mid, mid, ..., late,]
        df['RCP'] = self.scenarios * 3
        my_columns = [ "rchs",'time','scenarios']
        df.columns = my_columns
        os.makedirs(self.savedir,exist_ok=True)
        df.to_csv(os.path.join(self.savedir, csvname),index=False)

        return df

    def collect_raw_data(self,  
                        precip_hists, et_hists, 
                        precip_sms1, et_sms1,
                        precip_sms2, et_sms2,
                        csvname='extraction_pr_et.csv') -> pd.core.frame.DataFrame:
        # we need time and rcps 
        # we need here pr and et
        results_dict = defaultdict(list)

        #TODO the following iterated nesting should be function but intentionally not to do for ilijana
        for j in range(len(self.scenarios)):        
            results_dict['pr'].extend(precip_hists[j])
            results_dict['et'].extend(et_hists[j])
            results_dict['time'].extend([ self.time[0] for i in range(len(precip_hists[1]))  ]) 
            results_dict['scenario'].extend([ self.scenarios[j] for i in range(len(et_hists[1]))  ])
    
        for j in range(len(self.scenarios)):        
            results_dict['pr'].extend(precip_sms1[j])
            results_dict['et'].extend(et_sms1[j])
            results_dict['time'].extend([ self.time[1] for i in range(len(precip_sms1[1]))  ]) 
            results_dict['scenario'].extend([ self.scenarios[j] for i in range(len(et_sms1[1]))  ])
    
        for j in range(len(self.scenarios)):        
            results_dict['pr'].extend(precip_sms2[j])
            results_dict['et'].extend(et_sms2[j])
            results_dict['time'].extend([ self.time[2] for i in range(len(precip_sms2[1]))  ]) 
            results_dict['scenario'].extend([ self.scenarios[j] for i in range(len(et_sms2[1]))  ])

        # save data as pandas --> return is pandas or save as csv
        df = pd.DataFrame.from_dict(results_dict)
        os.makedirs(self.savedir,exist_ok=True)
        df.to_csv(os.path.join(self.savedir, csvname),index=False)

        return df

    def transfer_to_GCS(self, sourcedir:str, destpath:str, extension:str) -> None:
        """function to batch transfer the entire directory
            :param bucket_id: string bucket name of S3 
            :param sourceir:  directory path to a file in an GCP instance 
            :param destpath:  path under the bucket name
            :return: None
        """
        # Instantiates a client
        client = storage.Client()

        # google.cloud.storage.batch.Batch
        bucket = client.get_bucket(self.bucket_id)
        try:
            with client.batch():
                for ifile in glob.glob(os.path.join(sourcedir, f'*.{extension}')) : 
                    if destpath[-1] == "/":
                        blob = bucket.blob(destpath + os.path.basename(ifile))
                    else:
                        blob = bucket.blob(destpath + "/" + os.path.basename(ifile))
                    blob.upload_from_filename(ifile)
        except Exception as e:
            logging.info(e)
            pass

    def main(self, datadir:str, datasubdir:str, precip_nc_fn:str, et_nc_fn:str) -> None:
        
        # extract data from netCDF4
        fn1 = os.path.join(*[datadir, self.model, datasubdir, precip_nc_fn ])  # this is equivalent to fn1=f'/home/jupyter/{model}/regional/Extraction_pr.nc'
        fn2 = os.path.join(*[datadir, self.model, datasubdir, et_nc_fn ])  # this is equivalent to fn2=f'/home/jupyter/{model}/regional/Extraction_et.nc'
        precip = self.load_dataset(fn1, 'pr')
        et = self.load_dataset(fn2, 'et')

        # divide data
        precip_hists = self.split_data(precip, from_here=0,           until_here=self.mhist1)
        precip_sms1  = self.split_data(precip, from_here=self.mhist1, until_here=self.mhist2)
        precip_sms2  = self.split_data(precip, from_here=self.mhist2, until_here=None)

        et_hists = self.split_data(et, from_here=0,           until_here=self.mhist1)
        et_sms1  = self.split_data(et, from_here=self.mhist1, until_here=self.mhist2)
        et_sms2  = self.split_data(et, from_here=self.mhist2, until_here=None)

        # calculate rechage 
        df_rch = self.calculate_recharge(precip_hists, et_hists, 
                                        precip_sms1, et_sms1,
                                        precip_sms2, et_sms2,
                                        operation=self.operation)

        # save raw data as csv
        df_raw  = self.collect_raw_data(precip_hists,et_hists,  
                                        precip_sms1, et_sms1,
                                        precip_sms2, et_sms2,)

        
        # trasfer decoded data to GCP bucket
        # we need bucket_id, savedir, destpath, and extension i.e. .csv or .pkl, or .nc
        self.transfer_to_GCS(sourcedir=self.savedir, destpath=f'data-climate-recharge/{self.model}/regional/', extension='csv')

if __name__ == "__main__":
    # TODO: ilijana --> the below variables should be specified by argparse module but for your sake, 
    #                   I intentionally remain these variables by hard-code manner
    #bucket_id = 'us-digitaltwiner-dev-features'
    bucket_id = 'us-digitaltwiner-dev-staging'
    model     = 'MIROC-ESM'
    savedir   =os.path.join(os.getcwd(),f"csvs/{model}/regional")
    datadir   = "/home/jupyter/climate-data/downscaled" #'/home/jupyter'
    datasubdir= "regional/hydro5"  #'regional'      # datadir/model/datasubdir/filename
    precip_nc_fn = "Extraction_pr.nc"
    et_nc_fn =     "Extraction_et.nc"


    RDP = regional_data_processing(bucket_id=bucket_id, model=model, savedir=savedir, )    
    RDP.main(datadir=datadir, datasubdir=datasubdir, precip_nc_fn=precip_nc_fn, et_nc_fn=et_nc_fn) 
    
