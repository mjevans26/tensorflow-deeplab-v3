"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='dataset/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./dataset/inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./dataset/sample_images_list.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

parser.add_argument('--bands', nargs = 3, default = ['R','G','B'],
                    help='Which set of 3 bands to use?')

parser.add_argument('--patch_dims', type = int, default = 256, help = 'size of output predicted patch')

parser.add_argument('--buffer_size', type = int, default = 128, help = 'size of patch buffer for predictions')

_NUM_CLASSES = 2

def make_example(pred_dict):
  buffer_shape = [FLAGS.buffer_size, FLAGS.buffer_size]
  x_buffer = int(buffer_shape[0] / 2)
  y_buffer = int(buffer_shape[1] / 2)
  class_id = np.squeeze(pred_dict['classes'][:, x_buffer:x_buffer+FLAGS.patch_dims, y_buffer:y_buffer+FLAGS.patch.dims, :]).flatten()
  probability = np.squeeze(pred_dict['probabilities'][:, x_buffer:x_buffer+FLAGS.patch_dims, y_buffer:y_buffer+FLAGS.patch_dims, 1]).flatten()
  return tf.train.Example(
    features=tf.train.Features(
      feature={
        'class_id': tf.train.Feature(
            float_list=tf.train.FloatList(
                value=class_id)),
        'probability': tf.train.Feature(
            float_list=tf.train.FloatList(
                value=probability))
      }
    )
  )

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })

  #examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  #image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]
  image_files = tf.gfile.Glob('{}/*tfrecord.gz'.format(FLAGS.data_dir))
  print(image_files)
  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files, bands = FLAGS.bands, batch_size = 1, side = FLAGS.patch_dims+FLAGS.buffer_size),
        hooks=pred_hooks,
        yield_single_examples = False)
  
  print(next(predictions))
  output_dir = FLAGS.output_dir
  MAX_RECORDS_PER_FILE = 50
  output_path = output_dir + '-{:05}.tfrecord'

  # Create the records we'll ingest into EE
  file_number = 0
  still_writing = True
  total_patches = 0
  while still_writing:
    file_path = output_path.format(file_number)
    writer = tf.python_io.TFRecordWriter(file_path)
    print("Writing file: {}".format(file_path))
    try:
      written_records = 0
      while True:
        pred_dict = next(predictions)
        
        writer.write(make_example(pred_dict).SerializeToString())
      
        written_records += 1 
        total_patches += 1
      
        if written_records % 5 == 0:
          print("  Writing patch: {}".format(written_records))
      
        if written_records == MAX_RECORDS_PER_FILE:
          break
    except: 
      still_writing=False
    finally:
      file_number += 1
      writer.close()
  
  print('Wrote: {} patches.'.format(total_patches))
  
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for pred_dict, image_path in zip(predictions, image_files):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = image_basename + '_pred.npy'
    output_filename = image_basename + '_mask.png'
    path_to_output = os.path.join(output_dir, output_filename)

    print("generating:", path_to_output)
    #mask = pred_dict['decoded_labels']
    classes = pred_dict['classes']
    probs = pred_dict['probabilities']
    print(probs.shape, classes.shape)
    out = np.concatenate([probs, classes], axis = -1)
    np.save(path_to_output, out)
    #mask = Image.fromarray(mask)
    plt.axis('off')
    plt.imshow(probs[:, :, 0], cmap='hot', interpolation='nearest', vmin = 0.9, vmax = 1)
    plt.show()
    #plt.imshow(mask)
    #plt.savefig(path_to_output, bbox_inches='tight')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
