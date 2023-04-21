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
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
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
    p.add_argument('--logdir',type=str,default='./', help="name of directory to store log record at local pc/instance")
    p.add_argument('--input_datadir',nargs="+",default='list of cmip5 tfrecord data directories')
    p.add_argument('--output_modeldir',type=str,default='./')
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--expname',type=str,default='new')
    p.add_argument('--num_epoch',type=int,default=5)
    p.add_argument('--batch_size',type=int,default=32,help='number of pictures in minibatch')
    p.add_argument('--npatches',type=int,default=2000,help='number of patches across all tfrecord files')
    p.add_argument('--height',type=int,default=32)
    p.add_argument('--width',type=int,default=3)
    p.add_argument('--channel',type=int,default=1)
    p.add_argument('--nblocks',type=int,default=5)
    p.add_argument('--base_dim',type=int,help="coef for size of filter at first convolutional layer in first block in encoder. 2**(base_dim)",default=4)
    p.add_argument('--nstack_layer',type=int,help="Number of convolution layers/block",default=3)
    p.add_argument('--conv_kernel_size',type=int,help="Number of conv2D kernel size. The kernel size will decrease if conv2d height is smaller than the kernel size.",default=3)
    #--------------------------------------
    # New to clustering loss
    p.add_argument('--nclusters',type=int,help='Number of clusters that we predefined for network',default=5)
    p.add_argument('--sinkhorn_iterations',type=int,help="Number of iteration step for sinkhorn calculation. ",default=3)
    # lambda_r*(pr_l2 + et_l2) + FLAGS.lambda_c
    p.add_argument('--lambda_r',type=float,help="Coefficient of L2 loss ",default=0.6)
    p.add_argument('--lambda_c',type=float,help="Coefficient of Clustering loss ",default=0.4)
    p.add_argument('--epsilon',type=float,help="Coefficient of C^T * Z",default=0.05)
    p.add_argument('--temperature',type=float, help="Coefficient of Z^T C in log softmax. Caron 2021 was 0.1", default=1.0)
    #--------------------------------------
    p.add_argument('--retrain', action="store_true", help='attach this FLAGS if you need retraining from trained code',default=False)
    p.add_argument('--retrain_datadir',type=str, help='you should specify directory if you retrain from a trained model',default='./')
    p.add_argument('--save_every',type=int,default=10)

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
        band: str either pr or et
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
        
        decoded = tf.io.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.io.decode_raw(decoded["patch"], tf.float32), decoded["shape"]
        )
        # conversion of tensor
        pr_patch = patch[:,:,:3] # (#lon, #lat, #channel)
        et_patch = patch[:,:,3:] # (#lon, #lat, #channel)
        return pr_patch, et_patch


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

def cluster_space(shape):
    """ Prototype model in Caron 2021 et al. 
    """
    x = encoder_input = Input(shape=shape)
    x = Flatten()(x)
    x = Dense(FLAGS.nclusters, activation=None)(x)
    cluster_layer = Model(encoder_input, x)
    return cluster_layer


def sinkhorn(cluster_layer):
    Q = tf.transpose(tf.math.exp( cluster_layer / FLAGS.epsilon))
    B = Q.shape[1]
    K = Q.shape[0] 

    # make the matrix sums to 1 for each image
    sum_Q = tf.reduce_sum(Q)
    Q /= sum_Q

    for it in range(FLAGS.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = tf.reduce_sum(Q, axis=1, keepdims=True)
        Q /= sum_of_rows
        Q /= K
        
        # normalize each column: total weight per sample must be 1/B
        sum_of_cols = tf.reduce_sum(Q, axis=0, keepdims=True)
        Q /= sum_of_cols
        Q /= B
    
    # finalize result
    Q *= B
    return tf.transpose(Q) 

def softmax_fn(logits, axis):
    """ logits: pre-latent space activated by activation function"""
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)

def clustering_loss(pr_imgs, et_imgs, pr_encoder, et_encoder, pr_decoder, et_decoder, cluster_layer):
    # precip L2
    pr_encoded_imgs = pr_encoder(pr_imgs, training=True)
    pr_decoded_imgs = pr_decoder(pr_encoded_imgs, training=True)
    pr_l2 = tf.math.reduce_mean(tf.square(pr_decoded_imgs - pr_imgs))

    # et L2
    et_encoded_imgs = et_encoder(et_imgs, training=True)
    et_decoded_imgs = et_decoder(et_encoded_imgs, training=True)
    et_l2 = tf.math.reduce_mean(tf.square(et_decoded_imgs - et_imgs))

    # online clustering loss
    cluster_output = cluster_layer(tf.concat([pr_encoded_imgs, et_encoded_imgs], axis=-1)) 
    # normalize the prototype
    cluster_output = tf.math.l2_normalize(cluster_output, axis=1)


    # singhnop 
    q = sinkhorn(cluster_output)

    # cluster assignment prediction
    closs = -1 * tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(cluster_output/FLAGS.temperature, axis=1), axis=1))    

    # sum of loss
    loss = FLAGS.lambda_r*(pr_l2 + et_l2) + FLAGS.lambda_c*closs

    return loss,  pr_l2, et_l2, closs 

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
    global FLAGS
    FLAGS = get_args(hvd.rank() == 0)

    ## get filenames of training data as list
    train_images_list = []
    for input_datadir in FLAGS.input_datadir:
        train_images_list.extend(glob.glob(os.path.abspath(input_datadir)+'/extraction_0-*.tfrecord'))
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
                                    batch_size=FLAGS.batch_size,
                                    distribute=(hvd.size(), hvd.rank()),
                                    )
        iterator = iter(dataset)


    ## add resize layer 128x128 rsesize layer
    pr_encoder, pr_decoder = model_synmetric_resize_fn(
                    shape=(FLAGS.height, FLAGS.width, FLAGS.channel),
                    nblocks=FLAGS.nblocks,
                    base_dim=FLAGS.base_dim,
                    nstack_layer=FLAGS.nstack_layer,
                    conv_kernel_size=FLAGS.conv_kernel_size,
                    rheight=FLAGS.height, rwidth=FLAGS.width)

    et_encoder, et_decoder = model_synmetric_resize_fn(
                    shape=(FLAGS.height, FLAGS.width, FLAGS.channel),
                    nblocks=FLAGS.nblocks,
                    base_dim=FLAGS.base_dim,
                    nstack_layer=FLAGS.nstack_layer,
                    conv_kernel_size=FLAGS.conv_kernel_size,
                    rheight=FLAGS.height, rwidth=FLAGS.width)

    
    for ldx, layer in enumerate(pr_decoder.layers):
        if ldx == 0:
            _, rh, rw, rc = layer.output_shape[0]
    cluster_layer = cluster_space((rh, rw, 2*rc)) 

    if hvd.rank() == 0:
        print("\n {} \n".format(pr_encoder.summary()))
        print("\n {} \n".format(pr_decoder.summary()))
        print("\n {} \n".format(cluster_layer.summary()))

    # wandb init
    wandb.init(project="siamese-project",  sync_tensorboard=True)

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
    save_models = { "pr_encoder": pr_encoder, "pr_decoder": pr_decoder,
                    "et_encoder": et_encoder, "et_decoder": et_decoder,
                    "prototypes": cluster_layer,
                    }

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
    def train_step(pr_imgs, et_imgs, first_batch=False):
        with tf.GradientTape() as tape:
            # get L2 and clustering loss
            loss,  pr_l2, et_l2, closs  = clustering_loss(pr_imgs, et_imgs, pr_encoder, et_encoder, pr_decoder, et_decoder, cluster_layer)

        # --  autoencoder 
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, pr_encoder.trainable_weights+pr_decoder.trainable_weights+et_encoder.trainable_weights+et_decoder.trainable_weights+cluster_layer.trainable_weights)
        train_opt.apply_gradients(zip(grads, pr_encoder.trainable_weights+pr_decoder.trainable_weights+et_encoder.trainable_weights+et_decoder.trainable_weights+cluster_layer.trainable_weights))
        # boradcaset should be done after the first gradient step
        if first_batch:
            hvd.broadcast_variables(pr_encoder.variables, root_rank=0)
            hvd.broadcast_variables(pr_decoder.variables, root_rank=0)
            hvd.broadcast_variables(et_encoder.variables, root_rank=0)
            hvd.broadcast_variables(et_decoder.variables, root_rank=0)
            hvd.broadcast_variables(cluster_layer.variables, root_rank=0)
            hvd.broadcast_variables(train_opt.variables(), root_rank=0)
        return loss,  pr_l2, et_l2, closs


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
                pr_imgs, et_imgs = iterator.get_next() # next(dataset)

                if step ==0 and epoch == 0:
                    loss,  pr_l2, et_l2, closs  = train_step(pr_imgs, et_imgs, first_batch=True)
                else:
                    loss,  pr_l2, et_l2, closs  = train_step(pr_imgs, et_imgs, first_batch=False)

                if  hvd.rank() == 0 and step % 100 == 0:

                    # log metrics using wandb.log
                    wandb.log({'step': (epoch-1)*num_batches+step,
                            'loss':float(loss),
                            'pr_l2':float(pr_l2),
                            'et_l2':float(et_l2),
                            'closs':float(closs),
                            })

                    print(
                         "iteration {:7} | Loss {:10} | Clustering loss {:10} ".format(
                          step, loss, closs), 
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