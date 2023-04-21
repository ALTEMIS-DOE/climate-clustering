__author__ = "team digital twin: climate subgroup"

import os
import sys
import glob
import json
import time
import math
import logging
import argparse
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.image import image_gradients
from tensorflow.data.experimental import parallel_interleave
# horovod
from horovod import tensorflow as hvd

#wandb
import wandb
from wandb.keras import WandbCallback

# Import model
from models import model_synmetric_resize_fn

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)


# version check
logging.info(f"tensorflow == {tf.__version__}")

def get_args(verbose=False):
    """function to load argument
    """
    p = argparse.ArgumentParser()
    p.add_argument(
    '--logdir',
    type=str,
    default='./'
    )
    p.add_argument(
        '--input_datadir',
        nargs="+",
        default='list of clouds tfrecord data directories'
    )
    p.add_argument(
        '--output_modeldir',
        type=str,
        default='./'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=0.001
    )
    p.add_argument(
        '--expname',
        type=str,
        default='new'
    )
    p.add_argument(
        '--num_epoch',
        type=int,
        default=5
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='number of pictures in minibatch'
    )
    p.add_argument(
        '--npatches',
        type=int,
        default=2000,
        help='number of patches/tfrecord'
    )
    p.add_argument(
        '--height',
        type=int,
        default=32
    )
    p.add_argument(
        '--width',
        type=int,
        default=32
    )
    p.add_argument(
        '--channel',
        type=int,
        default=1
    )
    p.add_argument(
        '--nblocks',
        type=int,
        default=5
    )
    p.add_argument(
        '--base_dim',
        type=int,
        help="coef for size of filter at first convolutional layer in first block in encoder. 2**(base_dim)",
        default=4
    )
    p.add_argument(
        '--nstack_layer',
        type=int,
        help="Number of convolution layers/block",
        default=3
    )
    p.add_argument(
        '--conv_kernel_size',
        type=int,
        help="Number of conv2D kernel size. The kernel size will decrease if conv2d height is smaller than the kernel size.",
        default=3
    )
    p.add_argument(
        '--retrain',
        action="store_true",
        help='attach this FLAGS if you need retraining from trained code',
        default=False
    )
    p.add_argument(
        '--retrain_datadir',
        type=str,
        help='you should express directory if you retrain from a trained model',
        default='./'
    )
    p.add_argument(
        '--save_every',
        type=int,
        default=10
    )
    p.add_argument(
        '--band',
        type=str,
        default='pr'
    )
    p.add_argument(
        '--nshuffle',
        type=int,
        default=2048
    )

    # SHOW config on outfile
    args = p.parse_args()
    if verbose:
        for f in args.__dict__:
            print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
        print("\n")
    return args


def input_clouds_fn(filelist, batch_size=32, band='pr',
                    prefetch=1, read_threads=4, distribute=(1, 0), nshuffle=2048):
    """
      INPUT:
        prefetch: tf.int64. How many "minibatch" we asynchronously prepare on CPU ahead of GPU
    """

    def parser(ser):
        """
        Decode & Pass datast in tf.record
        *Cuation*
        floating point: tfrecord data ==> tf.float64
        """
        features = {
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "patch": tf.io.FixedLenFeature([], tf.string),
            "rcp":  tf.io.FixedLenFeature([], tf.string),
            "meta": tf.io.FixedLenFeature([], tf.string),
            "coordinate": tf.io.FixedLenFeature([3], tf.int64),
        }
        #"rcp": _bytes_feature(bytes(rcp, encoding="utf-8")),
        #"meta": _bytes_feature(bytes(meta, encoding="utf-8")),
        #"coordinate": _int64_feature(coord),
        #"shape": _int64_feature(patch.shape),
        #"patch": _bytes_feature(patch.ravel().tobytes()),
        
        decoded = tf.io.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.io.decode_raw(decoded["patch"], tf.float32), decoded["shape"]
        )
        # conversion of tensor
        if band == 'pr':
            patch = patch[:,:,:3] # (#lon, #lat, #channel)
        elif band == 'et':
            patch = patch[:,:,3:] # (#lon, #lat, #channel)
        elif band == 'rch' or 'recharge':
            pass # assume you use recharge tfrecord file
        return patch


    dataset = (
       tf.data.Dataset.list_files(filelist, shuffle=True)
           .shard(*distribute)
           .apply(
           parallel_interleave(
               lambda f: tf.data.TFRecordDataset(f).map(parser),
               cycle_length=read_threads,
               sloppy=True,
           )
       )
    )
    dataset = dataset.shuffle(nshuffle).cache().repeat().batch(batch_size).prefetch(prefetch)
    return dataset

def load_latest_model_weights(model, model_dir, name):
    """
      INPUT:
        model: encoder or decoder
        model_dir: model directory 
        name: model name.
      OUTPUT:
        step: global step 
    """
    latest = 0, None
    # get trained wegiht 
    for m in os.listdir(model_dir):
        if ".h5" in m and name in m:
            step = int(m.split("-")[1].replace(".h5", ""))
            latest = max(latest, (step, m))

    step, model_file = latest

    if not os.listdir(model_dir):
        raise NameError("no directory. check model path again")

    if model_file:
        model_file = os.path.join(model_dir, model_file)
        model.load_weights(model_file)
        print(" ... loaded weights for %s from %s", name, model_file)

    else:
        print("no weights for %s in %s", name, model_dir)

    return step


def loss_l2(imgs, encoder, decoder):
  encoded_imgs = encoder(imgs, training=True)
  decoded_imgs = decoder(encoded_imgs, training=True)
  return tf.math.reduce_mean(tf.square(decoded_imgs - imgs))


if __name__ == '__main__':
    # iniVt hvd
    hvd.init()

    ### GPU 
    # V.2: Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    if hvd.rank() == 0:
        logger.info("Number of GPUs {}".format(hvd.size()))


    # time for data preparation
    prep_stime = time.time()

    # get arg-parse as FLAGS
    FLAGS = get_args(hvd.rank() == 0)


    ## get filenames of training data as list
    train_images_list = []
    for input_datadir in FLAGS.input_datadir:
        train_images_list.extend(glob.glob(os.path.abspath(input_datadir)+'/extraction_*.tfrecord'))
        #train_images_list.extend(glob.glob(os.path.abspath(input_datadir)+'/extraction_1-*.tfrecord'))
    print("Number of tfrecords = ", len(train_images_list), flush=True)

    # make dirs
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs(FLAGS.output_modeldir, exist_ok=True)

    # loss log filename
    # outputnames
    ctime = datetime.now()
    bname1 = '_nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)+'_nbatch-'+str(FLAGS.batch_size)
    ofilename = 'loss_'+FLAGS.expname+bname1+'_'+str(ctime.strftime("%s"))+'.txt'


#-----------------------------------------------------
# Pipeline
#-----------------------------------------------------
    #### TRAIN
    with tf.device('/GPU'):
        # get dataset and one-shot-iterator
        dataset = input_clouds_fn(train_images_list,
                                    band=FLAGS.band,
                                    batch_size=FLAGS.batch_size,
                                    distribute=(hvd.size(), hvd.rank()),
                                    nshuffle=FLAGS.nshuffle,
                                    )
        iterator = iter(dataset)


    ## add resize layer 128x128 rsesize layer
    encoder, decoder = model_synmetric_resize_fn(
                    shape=(FLAGS.height, FLAGS.width, FLAGS.channel),
                    nblocks=FLAGS.nblocks,
                    base_dim=FLAGS.base_dim,
                    nstack_layer=FLAGS.nstack_layer,
                    conv_kernel_size=FLAGS.conv_kernel_size,
                    rheight=FLAGS.height, rwidth=FLAGS.width)

    if hvd.rank() == 0:
        print("\n {} \n".format(encoder.summary()))
        print("\n {} \n".format(decoder.summary()))

    # wandb init
    wandb.init(project="my-test-project",  sync_tensorboard=True)

    # Number of iteration (for learning rate scheduler)
    num_batches=FLAGS.npatches //FLAGS.batch_size // hvd.size() # npatches is the number of all patches from all files

    # Learning rate decay
    initial_learning_rate = FLAGS.lr   
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=40*num_batches,
        decay_rate=0.96,
        staircase=True,
        name='lr_decay'
    )

    wandb.config = {
        "learning_rate": initial_learning_rate,
        "epochs": FLAGS.num_epoch,
        "batch_size":  FLAGS.batch_size
        }

    # Apply optimization 
    train_opt = tf.keras.optimizers.SGD(lr_schedule)

    # set-up save models
    save_models = {"encoder": encoder, "decoder": decoder}

    # save model definition
    for m in save_models:
        with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
            f.write(save_models[m].to_json())


    # End for prep-processing during main code
    if hvd.rank() == 0:
        logger.info("\n### Entering Training Loop ###\n")

    #--------------------------------------------------------------------
    # Restart
    #--------------------------------------------------------------------
    if FLAGS.retrain:
        # e.g. restart_modeldir = os.path.abspath('./output_model/66153901')
        restart_modeldir = os.path.abspath(FLAGS.retrain_datadir)
        for m in save_models:
            gs = load_latest_model_weights(save_models[m],restart_modeldir,m)
  
    # TRAINING
    # train function
    @tf.function
    def train_step(imgs, first_batch=False):
        with tf.GradientTape() as tape:
            # get loss
            loss = loss_l2(imgs, encoder, decoder)       

        # --  autoencoder 
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, encoder.trainable_weights+decoder.trainable_weights)
        train_opt.apply_gradients(zip(grads, encoder.trainable_weights+decoder.trainable_weights))
        # boradcaset should be done after the first gradient step
        if first_batch:
            hvd.broadcast_variables(encoder.variables, root_rank=0)
            hvd.broadcast_variables(decoder.variables, root_rank=0)
            hvd.broadcast_variables(train_opt.variables(), root_rank=0)
        return loss


    # tf summary
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'tflogs/{FLAGS.expname}/train'
    os.makedirs(train_log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(train_log_dir)

    # custom training loop
    loss_list = []
    stime = time.time()    
    for epoch in range(1,FLAGS.num_epoch+1,1):
        if hvd.rank() == 0: 
            print("\nStart of epoch %d" % (epoch,), flush=True)
        
        start_time = time.time()
        for step in  range(num_batches):
            try:
                imgs = iterator.get_next() # next(dataset)

                if step ==0 and epoch == 0:
                    loss  = train_step(imgs, first_batch=True)
                else:
                    loss  = train_step(imgs, first_batch=False)

                if  hvd.rank() == 0 and step % 100 == 0:
                    # tf summary
                    #tf.summary.scalar('L2_loss', loss, step=(epoch-1)*num_batches+step)
                    #writer.flush()
                    #wandb.tensorflow.log(writer.flush())

                    # log metrics using wandb.log
                    wandb.log({'step': (epoch-1)*num_batches+step,
                            'loss':float(loss)})

                    print(
                         "iteration {:7} | Loss {:10} ".format(
                          step, loss), 
                        flush=True
                    )  
                    loss_list.append(loss)

            except tf.errors.OutOfRangeError as e:
                if hvd.rank() == 0:
                    print(f"End EPOCH {epoch} \n", flush=True) 
                pass        

        # show compute time every epochs
        print(f"Rank  %d   Time taken: %.3fs" % ( hvd.rank(),time.time() - start_time), flush=True)
          
        # save model at every N steps
        if hvd.rank() == 0 and  epoch % FLAGS.save_every == 0 and epoch > 0:
            for m in save_models:
                save_models[m].save_weights(
                    os.path.join(
                        FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
                    )
                )

    # Finalize code
    wandb.finish()
    print("### TRAINING NORMAL END ###")
    # FINISH
    etime = (time.time() -stime)/60.0 # minutes
    print("   Execution time [minutes]  : %f" % etime, flush=True)