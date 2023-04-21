import os
import gc
import copy
import time
import pickle
import logging
import numpy as np

#from mpi4py   import MPI
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# climate explorer libraries
from lib4climex import loader

import tensorflow as tf
print(tf.__version__)

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_feature(writer, rcp, meta, coord, patch):
    feature = {
        "rcp": _bytes_feature(bytes(rcp, encoding="utf-8")),
        "meta": _bytes_feature(bytes(meta, encoding="utf-8")),
        "coordinate": _int64_feature(coord),
        "shape": _int64_feature(patch.shape),
        "patch": _bytes_feature(patch.ravel().tobytes()),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

    
def write_timesep_patches_v2(patches, out_dir, patches_per_record, rank=0):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
    Args:
        patches: Iterable of (rcp, coordinate, patch) which defines tfrecord example
            to write.
        out_dir: Directory to save tfrecords.
        patches_per_record: Number of examples to save in each tfrecord.
    """
    #rank = MPI.COMM_WORLD.Get_rank()
    ntest = 0
    nval  = 0
    ntotal= 0
    for i, _patch in enumerate(patches):

        (rcp, meta, coord, patch, train_flag) = _patch # extract just train flag
        patch = (rcp, meta, coord, patch)

        if i % patches_per_record == 0 and train_flag:
            # TRAIN RECORD
            rec = f"extraction_{rank}-{(i // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec)
            f = tf.io.TFRecordWriter(os.path.join(out_dir, rec))

        if ntest % patches_per_record == 0 and train_flag:
            # TEST RECORD
            rec2 = f"cluster_{rank}-{( ntest // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec2)
            ft = tf.io.TFRecordWriter(os.path.join(out_dir, rec2))

        if nval % patches_per_record == 0 and not train_flag:
            # Validation RECORD
            rec3 = f"evaluation_{rank}-{( nval // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec3)
            fe = tf.io.TFRecordWriter(os.path.join(out_dir, rec3))

        if train_flag: 
            #write_feature(f, *patch)
            #print("Rank", rank, "wrote", i + 1, "patches", flush=True)
            flag_choice=np.random.choice([True, False], p=[0.7, 0.3]) # False is test patch
            if flag_choice:
                write_feature(f, *patch)
                print("Rank", rank, "wrote", i + 1, "patches", flush=True)
            else:
                write_feature(ft, *patch)
                print("Rank", rank, "wrote", ntest + 1, "cluster patches", flush=True)
                ntest+=1
            ntotal+=1
        else:
            write_feature(fe, *patch)
            print("Rank", rank, "wrote", nval + 1, "evaluation patches", flush=True)
            nval+=1

def write_timesep_patches(patches, out_dir, patches_per_record, rank=0):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
    Args:
        patches: Iterable of (rcp, coordinate, patch) which defines tfrecord example
            to write.
        out_dir: Directory to save tfrecords.
        patches_per_record: Number of examples to save in each tfrecord.
    """
    #rank = MPI.COMM_WORLD.Get_rank()
    ntest = 0
    for i, _patch in enumerate(patches):

        (rcp, meta, coord, patch, train_flag) = _patch # extract just train flag
        patch = (rcp, meta, coord, patch)

        if i % patches_per_record == 0 and train_flag:
            # TRAIN RECORD
            rec = f"extraction_{rank}-{(i // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec)
            f = tf.io.TFRecordWriter(os.path.join(out_dir, rec))

            # TEST RECORD
        if ntest % patches_per_record == 0 and not train_flag:
            rec2 = f"test_{rank}-{( ntest // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec2)
            ft = tf.io.TFRecordWriter(os.path.join(out_dir, rec2))

        if train_flag: 
            write_feature(f, *patch)
            print("Rank", rank, "wrote", i + 1, "patches", flush=True)
        else:
            write_feature(ft, *patch)
            print("Rank", rank, "wrote", ntest + 1, "test patches", flush=True)
            ntest+=1

def write_patches(patches, out_dir, patches_per_record, rank=0):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
       Write patches into training and testing tfrecord file based on timeindices
    Args:
        patches: Iterable of (rcp, coordinate, patch) which defines tfrecord example
            to write.
        out_dir: Directory to save tfrecords.
        patches_per_record: Number of examples to save in each tfrecord.
    """
    #rank = MPI.COMM_WORLD.Get_rank()
    ntotal=0
    ntest = 0
    for i, patch in enumerate(patches):

        if i % patches_per_record == 0:
            rec = f"extraction_{rank}-{(i // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec)
            f = tf.io.TFRecordWriter(os.path.join(out_dir, rec))

        if ntest % patches_per_record == 0:
            # TEST RECORD
            rec2 = f"test_{rank}-{( ntest // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec2)
            ft = tf.io.TFRecordWriter(os.path.join(out_dir, rec2))

        # initial index is always training data to avoid writing error to tfrecord
        if i == 0:
            flag_choice = True
        else:
            flag_choice=np.random.choice([True, False], p=[0.7, 0.3]) # False is test patch

        if  i > 0 and not flag_choice: 
            write_feature(ft, *patch)
            print("Rank", rank, "wrote", ntest + 1, "test patches", flush=True)
            ntest+=1
        else:
            write_feature(f, *patch)
            print("Rank", rank, "wrote", i + 1, "patches", flush=True)
            ntotal+=1
    print(f"N Total = {ntotal}")

def alltogether_write_patches(patches, out_dir, patches_per_record, rank=0):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
       Write patches into training and testing tfrecord file based on timeindices
    Args:
        patches: Iterable of (rcp, coordinate, patch) which defines tfrecord example
            to write.
        out_dir: Directory to save tfrecords.
        patches_per_record: Number of examples to save in each tfrecord.
    """
    #rank = MPI.COMM_WORLD.Get_rank()
    for i, patch in enumerate(patches):

        if i % patches_per_record == 0:
            rec = f"extraction_{rank}-{(i // patches_per_record)}.tfrecord"
            logger.info("Writing to", rec)
            f = tf.io.TFRecordWriter(os.path.join(out_dir, rec))

        write_feature(f, *patch)
        print("Rank", rank, "wrote", i + 1, "patches", flush=True)

# long-term 3 month mean
def longTermMean(X):
    nmonth_per_year=12 # 12 months per year
    nmonth = X.shape[0]
    lms_index = {
        1 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(3)])),
        2 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(1,4) ])),
        3 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(2,5) ])),
        4 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(3,6) ])),
        5 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(4,7) ])),
        6 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(5,8) ])),
        7 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(6,9) ])),
        8 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(7,10) ])),
        9 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(8,11) ])),
        10 : np.sort(np.hstack([np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(9,12) ])),
        11:  np.sort(np.hstack([0]+[np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(10,13) ])),
        12 : np.sort(np.hstack([0,1]+[np.arange(i,nmonth,nmonth_per_year).tolist() for i in range(11,13) ])),
    }

    # compute long term mean
    long_seasonal_means = np.zeros((nmonth_per_year))
    for month, lms in lms_index.items():
        long_seasonal_means[month-1] = np.mean(X[lms]) # index is -1
    
    for i in range(nmonth):
        imonth =  i % nmonth_per_year # 0 - 11
        #print(imonth)
        X[i] -= long_seasonal_means[imonth]
    return X


def StrideAnnualMean(X, stride,nt):
    long_term_mean = np.ma.mean(X,dtype=np.float32)
    Xc = None
    for it in range(0, len(X), stride):
        if it+nt < len(X):
            tmp = np.ma.mean(X[it:it+nt], axis=0, keepdims=True,dtype=np.float32) - long_term_mean
            #print(np.ma.mean(tmp), flush=True)
            if it == 0:
                Xc = tmp
            else:
                Xc = np.ma.concatenate([Xc, tmp], axis=0)
    #print(type(Xc), flush=True)
    #exit(0)
    # Xc should be numpy.ma.core.MaskedArray
    return Xc.astype(np.float32)
    

# here 3 month running mean
def movingMean(X):
    
    def  average_calculator(x):    
        """ x[3month, nlat, nlon] --> x[nlat, nlon]
        """
        x = np.mean(x, axis=0)
        return x

    # Get monthly mean
    detrend_precip = []
    Xc = copy.deepcopy(X)
    for it in range(0, len(X)):
        if it == len(X)-1:
            average = average_calculator(np.sum(X[-3:], axis=0))
            #print( X[it], average)
            seasonality_removed = X[it]-average
            #detrend_precip.append(seasonality_removed)
               
        elif it == 0:
            average = average_calculator(np.sum(X[:3], axis=0))
            seasonality_removed = X[it]-average
            #detrend_precip.append(seasonality_removed)
              
        else:
            average = average_calculator(np.sum(X[it:it+3],axis=0))
            seasonality_removed = X[it+1]-average
            #detrend_precip.append(seasonality_removed)
        Xc[it] = seasonality_removed
    return Xc 
    #np.array(detrend_precip)


# latitude average
def latitudal_weight(lats):
    """
    input: latitude (np.nd.array) : [#nlat,]
    """
    AreaWeight = np.cos(np.deg2rad(lats))
    return AreaWeight

# standardize data
def scaling(data, lats):
    """function to standardize data

        Input
            data = [#month(after deseasonalized), #nlat, #nlon]
    
        Output
            rescaled data = [#month(after deseasonalized), #nlat, #nlon]
    
        Standardization
            scaleX = (X - mean)/std

        Weight
            avg = sum(a * weights) / sum(weights)

    """
    AreaWeight = latitudal_weight(lats) # (#lats, )
    # zonal mean
    print('data before', type(data), flush=True)
    zonal_mean = np.average(data,axis=2) # get zonal mean data
    print('zonal mean ', type(zonal_mean), flush=True)
    global_mean = np.average(zonal_mean,axis=1,weights = AreaWeight)
    print('global mean', type(global_mean), flush=True)
    gc.collect() # clean up gc
    print("global mean here is fine", flush=True)


    global_stdv = None
    nt, nx, ny = data.shape 
    for (idx, edata), eglobal_mean in zip(enumerate(data), global_mean):
        # edata [nlat, nlon]: each time
        x = AreaWeight.reshape(-1,1) * edata
        x = np.abs(x - eglobal_mean)**2
        x = np.sqrt(np.sum(x)/(nx*ny))
        if idx == 0:
            global_stdv = x
        else:
            global_stdv = np.hstack([global_stdv, x])

    gc.collect() # clean up gc
    print("global std here is fine", flush=True)
    print('global stdv', type(global_stdv), flush=True)

    # logging
    logger.info(f"Global mean = {global_mean.max()} | Global std {global_stdv.max()}")
    print(global_mean.shape, global_stdv.shape)

    # standardized data
    if global_mean.shape[0] == 1:
        data[0] = (data[0] - global_mean) / global_stdv
    else:
        for (idx, gmean), gstdv in zip(enumerate(global_mean), global_stdv):
            data[idx] = (data[idx] - gmean) / gstdv

    return data

def save_pickle(savedata, savedir, savefname):
        with open(os.path.join(savedir, savefname) , 'wb' ) as f:
            pickle.dump(savedata, f)



def input_historical_data_generator_v2(precip, et, lats, rcp='rcpxx', cmip_meta="", stride=8, patch_size=16, timewindow=3, mhist=852):
    """ function to yield historical patches for training and future run for test
        version 2.0 of input_data_generator

        :param mhist: integer of index which separates historical time (1950-2020) and future projection time (2021-2099/2100)

    """

    def _exec_preprocess(var):
        var = movingMean(var)
        print('mv1',type(var))
        var = scaling(var,lats)
        return var

    # apply our preprocessing process 
    s1 = time.time()
    precip = _exec_preprocess(precip)
    #save_pickle(precip, ".", f'scaled_precip_{rcp}.pkl')
    et = _exec_preprocess(et)
    #save_pickle(et, ".", f'scaled_et_{rcp}.pkl')
    s2 = time.time()
    logger.info(f" Global preprocessing time {s2-s1} sec")
    

    # create coordination
    max_t, max_lat, max_lon = precip.shape
    coords = []
    for x in range(0, max_lat, stride):
        for y in range(0, max_lon, stride):
            if x + patch_size < max_lat and y + patch_size < max_lon:
                coords.append((x, y))

    for it in range(0, max_t - timewindow + 1): 
        
        train_flag =True if it <= mhist else False # boolean : if it <= mhist train is True else False to be test data

        for i, j in coords:
            try:
                _pr = precip[it:it+timewindow]
                _et = et[it:it+timewindow]
            except Exception as e:
                logger.debug(f" IndexError : timewindow reaches the list of out indices {e}")
            _pr = _pr[:, i:i + patch_size, j:j + patch_size] # [t,  x , y]
            _et = _et[:, i:i + patch_size, j:j + patch_size]
            #if it == 0:
            #    print("SHAPE CHECK : ", _pr.shape, _et.shape)

            # move axis and merge data
            if not _pr.mask.any() and not _et.mask.any():
                if _pr.max() < 1.0e5 and _et.max() < 1.0e5:
                    #print(_pr.max(), flush=True)
                    #print(_et.max(), flush=True)
                    #exit()
                    # moveaxis to be channel
                    mdata = np.concatenate([_pr, _et], axis=0) 
                    patch = np.moveaxis(mdata, 0,-1) # [t,x,y] --> [x,y,t]
                    yield rcp, cmip_meta, (i,j,it), patch, train_flag 

def input_recharge_data_generator_v2(precip, et, lats, rcp='rcpxx', cmip_meta="", stride=8, patch_size=16, timewindow=3, mhist=852):
    """ function to yield recharge patches for training and testing
        version 2.0 of input_recharge_data_generator to incorporate historical / future separation

        :param mhist: integer of index which separates historical time (1950-2020) and future projection time (2021-2099/2100)

    """

    def _exec_preprocess(var):
        var = movingMean(var)
        print('mv1',type(var))
        var = scaling(var,lats)
        return var

    # apply our preprocessing process 
    s1 = time.time()
    rch = precip - et
    rch =  _exec_preprocess(rch)
    #precip = _exec_preprocess(precip)
    #et = _exec_preprocess(et)
    save_pickle(et, ".", f'scaled_rch_hist_overlap_{rcp}.pkl')
    s2 = time.time()
    logger.info(f" Global preprocessing time {s2-s1} sec")
    

    # create coordination
    max_t, max_lat, max_lon = precip.shape
    coords = []
    for x in range(0, max_lat, stride):
        for y in range(0, max_lon, stride):
            if x + patch_size < max_lat and y + patch_size < max_lon:
                coords.append((x, y))

    #for it in range(0, max_t - timewindow + 1): 
    for it in range(0, max_t - timewindow + 1): 
        
        train_flag =True if it <= mhist else False # boolean : if it <= mhist train is True else False to be test data

        for i, j in coords:
            try:
                _rch = rch[it:it+timewindow]
            except Exception as e:
                logger.debug(f" IndexError : timewindow reaches the list of out indices {e}")
            _rch = _rch[:, i:i + patch_size, j:j + patch_size] # [t,  x , y]
            #if it == 0:
            #    print("SHAPE CHECK : ", _pr.shape, _et.shape)

            # move axis and merge data
            if not _rch.mask.any():
                if _rch.max() < 1.0e5:
                    #print(_pr.max(), flush=True)
                    #print(_et.max(), flush=True)
                    #exit()
                    # moveaxis to be channel
                    patch = np.moveaxis(_rch, 0,-1) # [t,x,y] --> [x,y,t]
                    yield rcp, cmip_meta, (i,j,it), patch, train_flag
                    
def input_elevation_data_generator(elev, lats, rcp='rcpxx', cmip_meta="", 
                                    stride=8, patch_size=16, timewindow=3, time_stride=1):
    """ function to yield precip + ET + elevation data
        version 4.0 of input_data_generator
        Also this is the base code for input_data_generator_v2 to add stride for timeseries axis

        :param mhist: integer of index which separates historical time (1950-2020) and future projection time (2021-2099/2100)

    """

    # apply our preprocessing process 
    s1 = time.time()
    elev = scaling(np.expand_dims(elev, axis=0), lats) # add timeaxis (t, x,y)
    s2 = time.time()
    logger.info(f" Global preprocessing time {s2-s1} sec")
    

    # create coordination
    _, max_lat, max_lon = elev.shape
    coords = []
    for x in range(0, max_lat, stride):
        for y in range(0, max_lon, stride):
            if x + patch_size < max_lat and y + patch_size < max_lon:
                coords.append((x, y))

    for i, j in coords:
        _elev = elev[0, i:i + patch_size, j:j + patch_size] # [0, x , y]

        # move axis and merge data
        if not _elev.mask.any():
            if np.abs(_elev).max() < 1.0e5:
                # add to channel axis
                patch = np.expand_dims(_elev, axis=-1)
                # change np.float64 to np.float32
                patch = patch.astype(np.float32)
                yield rcp, cmip_meta, (i,j,0), patch 


def get_args(verbose=False):
    """function to load argument
    """
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__
    )

    # SHOW config on outfile
    p.add_argument("--precip_filename",     type=str, default='Extraction_pr.nc',           help='filename of precipitation netcdf file. In future should be parsing list')
    p.add_argument("--et_filename",         type=str, default='Extraction_et.nc',           help='filename of et netcdf file. In future should be parsing list')
    p.add_argument("--output_datadir",      type=str, default='/home/jupyter/climate-data', help='output of tfrecord data directory')
    p.add_argument("--stride",              type=int, default=8,    help='number of pixels where the next patch extraction starts. stride < patch_size results in overlapping two adjacent patches')
    p.add_argument("--patch_size",          type=int, default=16,   help='width and height of patch i.e. a subset of large image')
    p.add_argument("--timewindow",          type=int, default=3,    help='number of month incorporated into each patch')
    p.add_argument("--patches_per_record",  type=int, default=3,    help='number of month incorporated into each patch')
    p.add_argument("--cmip_meta",           type=str, default="",   required=True,help='one docstring to clarify data source. format=(model name, area, downscaled or not, reso) e.g. gfdl-esm2g conus downscaled 1/8 deg. ')
    p.add_argument("--rcps",                nargs='+', default=['rcp26', 'rcp45', 'rcp60', 'rcp85'] , help="List of rcp scnerarios in dataset e.g. ['rcp26', 'rcp45', ]",)
    # New Added V2
    p.add_argument("--mhist",               type=int, default=852,  help='number of month at the boundary of historical and future projection. Year 2020 in this config')
    # New Added ELEV
    p.add_argument("--elev_filename",       type=str, default='elev_thin_PRISM_us_dem_12km.npy', help='filename of precipitation netcdf file. In future should be parsing list')
    
    args = p.parse_args()
    if verbose:
        for f in args.__dict__:
            print("\t", f, (25 - len(f)) * " ", args.__dict__[f], flush=True)
        print("\n")
    return args

def main(verbose):

    FLAGS = get_args(verbose=verbose)
    os.makedirs(FLAGS.output_datadir, exist_ok=True)

    # load data
    s1 = time.time()
    precip = loader.netcdf_reader(FLAGS.precip_filename, varname='pr')
    et=loader.netcdf_reader(FLAGS.et_filename,varname='et')
    lat=loader.netcdf_reader(FLAGS.precip_filename,varname='latitude')
    s2 = time.time()
    logger.info(f" Loading time {s2-s1} sec")

    # main process start
    # TODO add rcps as a list of argument
    for rdx, rcp in enumerate(FLAGS.rcps):
        # V2: Train : Test = historical : mid & late century
        # deseazonalize, tandardize, and subdivde data
        patches = input_historical_data_generator_v2(
                    precip[rdx], et[rdx], lat, rcp=rcp, cmip_meta=FLAGS.cmip_meta, 
                    stride=FLAGS.stride, patch_size=FLAGS.patch_size, timewindow=FLAGS.timewindow, mhist=FLAGS.mhist)

        # IO data
        # historical timewindow training and future projection testing and evaluation
        write_timesep_patches_v2(patches, FLAGS.output_datadir, FLAGS.patches_per_record, rank=rdx)


def npy_loader(filename, nan_mask=True, thres=-9999.000):
    """function to load numpy data with missing/invalid value defined as either np.nan or `thres` 
        INPUT
        :param nan_mask: boolean flag. Make True if the loading npy file defined invalid/missing piexls by np.nan
        :param thres: (Specify esp. nan_mask is False) int or float value which defines invalid/missin value
    """
    data = np.load(filename)
    # convert numpy.ndarray to np.masked.Array object
    if nan_mask:
        return np.ma.masked_invalid(data)
    else:
        return np.ma.masked_values(data, thres)

def main_elev(verbose):

    FLAGS = get_args(verbose=verbose)
    os.makedirs(FLAGS.output_datadir, exist_ok=True)

    # load data
    s1 = time.time()
    lat=loader.netcdf_reader(FLAGS.precip_filename,varname='latitude')
    elev=npy_loader(FLAGS.elev_filename, nan_mask=True)
    s2 = time.time()
    logger.info(f" Loading time {s2-s1} sec")

    patches = input_elevation_data_generator(elev, lat, rcp="PRISM-DS-12km", cmip_meta=FLAGS.cmip_meta, 
                    stride=FLAGS.stride, patch_size=FLAGS.patch_size,)

    alltogether_write_patches(patches, FLAGS.output_datadir, FLAGS.patches_per_record, rank=0)
if __name__ == "__main__":
    main(True)
    #main_elev(True)






        

