# MIT License, see LICENSE
# Copyright (c) 2018 ClusterOne Inc.
# Author: Adrian Yi, adrian@clusterone.com

"""
Runs distributed training of a self-steering car model.
"""

import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from models.model import *
from utils.data_reader import *
from utils.view_steering_model import *

try:
    # These environment variables will be available on all distributed TensorFlow jobs on Clusterone
    job_name = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except KeyError as e:
    # This will be used for single instance jobs.
    # If running distributed job locally manually, you need to pass in these values as arguments.
    job_name = None
    task_index = 0
    ps_hosts = ''
    worker_hosts = ''


def make_tf_config(opts):
    # Distributed TF Estimator codes require TF_CONFIG environment variable
    if opts.job_name is None:
        return {}
    tf_config = {'task': {'type': opts.job_name, 'index': opts.task_index},
                 'cluster': {'master': [opts.worker_hosts[0]],
                             'worker': opts.worker_hosts,
                             'ps': opts.ps_hosts},
                 'environment': 'cloud'}
    # Nodes may need to refer to itself as localhost
    local_ip = 'localhost:' + tf_config['cluster'][opts.job_name][opts.task_index].split(':')[1]
    tf_config['cluster'][opts.job_name][opts.task_index] = local_ip
    if job_name == 'worker' and task_index == 0:
        tf_config['task']['type'] = 'master'
        tf_config['cluster']['master'][0] = local_ip
    return tf_config


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # Configuration for distributed task
    parser.add_argument('--job_name', type=str, default=job_name,
                        help='worker/ps or None')
    parser.add_argument('--task_index', type=int, default=task_index,
                        help='Worker task index, should be >= 0. task_index=0 is the chief worker.')
    parser.add_argument('--ps_hosts', type=str, default=ps_hosts,
                        help='Comma-separated list of hostname:port pairs')
    parser.add_argument('--worker_hosts', type=str, default=worker_hosts,
                        help='Comma-separated list of hostname:port pairs')
    # Experiment related parameters
    parser.add_argument('--local_data_root', type=str, default=os.path.abspath('./data/'),
                        help='Path to dataset. This path will be /data on Clusterone.')
    parser.add_argument('--local_log_root', type=str, default=os.path.abspath('./logs/'),
                        help='Path to store logs and checkpoints. This path will be /logs on Clusterone.')
    parser.add_argument('--data_subpath', type=str, default='',
                        help='Which sub-directory the data will sit inside local_data_root (locally) ' +
                             'or /data/ (on Clusterone)')
    # Model params
    parser.add_argument('--dropout_rate1', type=float, default=0.2)
    parser.add_argument('--dropout_rate2', type=float, default=0.5)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--nogood', action='store_true',
                        help='Ignore "goods" filters')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--learning_decay', type=float, default=0.0001)
    # Training params
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--frames_per_sample', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'],
                        help='TF logging level. To see intermediate results printed, set this to INFO or DEBUG.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads to use to prepare data')
    parser.add_argument('--max_ckpts', type=int, default=2,
                        help='Maximum number of checkpoints to keep')
    parser.add_argument('--ckpt_steps', type=int, default=100,
                        help='How frequently to save a model checkpoint')
    parser.add_argument('--save_summary_steps', type=int, default=10,
                        help='How frequently to save TensorBoard summaries')
    parser.add_argument('--log_step_count_steps', type=int, default=10,
                        help='How frequently to log loss & global steps/s')
    parser.add_argument('--eval_secs', type=int, default=60,
                        help='How frequently to run evaluation step. ' +
                             'By default, there is no evaluation dataset, thus effectively no evaluation.')
    # Parse args
    opts = parser.parse_args()
    opts.train_data = get_data_path(dataset_name='*/*',
                                    local_root=opts.local_data_root,
                                    local_repo=opts.data_subpath,
                                    path='camera/training/*.h5')
    opts.log_dir = get_logs_path(root=opts.local_log_root)
    opts.ps_hosts = opts.ps_hosts.split(',')
    opts.worker_hosts = opts.worker_hosts.split(',')
    return opts


def read_row(filenames):
    reader = DataReader(filenames)
    x, y, s = reader.read_row_tf()
    x.set_shape((3, 160, 320))
    y.set_shape(1)
    s.set_shape(1)
    return x, y, s


def get_input_fn(files, opts, is_train=True):
    """Returns input_fn.  is_train=True shuffles and repeats data indefinitely"""
    def input_fn():
        with tf.device('/cpu:0'):
            x, y, s = read_row(files)
            if is_train:
                X, Y, S = tf.train.shuffle_batch([x, y, s],
                                                 batch_size=opts.batch_size,
                                                 capacity=5 * opts.batch_size,
                                                 min_after_dequeue=2 * opts.batch_size,
                                                 num_threads=opts.num_threads)
            else:
                X, Y, S = tf.train.batch([x, y, s],
                                         batch_size=opts.batch_size,
                                         capacity=5 * opts.batch_size,
                                         num_threads=opts.num_threads)
            return {'features': X, 's': S}, Y
    return input_fn


def get_model_fn(opts):
    """Returns input_fn.  is_train=True shuffles and repeats data indefinitely"""
    def model_fn(features, labels, mode):
        features, s = features['features'], features['s']
        y_pred = get_model(features, opts)

        tf.summary.image("green-is-predicted", render_steering_tf(features, labels, s, y_pred))

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'prediction': y_pred}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = get_loss(y_pred, labels)
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            lr = tf.train.exponential_decay(learning_rate=opts.learning_rate,
                                            global_step=global_step,
                                            decay_steps=1,
                                            decay_rate=opts.learning_decay)
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    return model_fn


def main(opts):
    # Create an estimator
    config = tf.estimator.RunConfig(
        model_dir=opts.log_dir,
        save_summary_steps=opts.save_summary_steps,
        save_checkpoints_steps=opts.ckpt_steps,
        keep_checkpoint_max=opts.max_ckpts,
        log_step_count_steps=opts.log_step_count_steps)
    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(opts),
        config=config)

    # Create input fn
    # We do not provide evaluation data, so we'll just use training data for both train & evaluation.
    train_input_fn = get_input_fn(opts.train_data, opts, is_train=True)
    eval_input_fn = get_input_fn(opts.train_data, opts, is_train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1e6)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=1,
                                      start_delay_secs=0,
                                      throttle_secs=opts.eval_secs)

    # Train and evaluate!
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    args = parse_args()
    tf.logging.set_verbosity(args.verbosity)

    print('=' * 30, 'Environment Variables', '=' * 30)
    for k, v in os.environ.items():
        print('{}: {}'.format(k, v))

    print('='*30, 'Arguments', '='*30)
    for k, v in sorted(args.__dict__.items()):
        if v is not None:
            print('{}: {}'.format(k, v))

    TF_CONFIG = make_tf_config(args)
    print('='*30, 'TF_CONFIG', '='*30)
    print(TF_CONFIG)
    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)

    print('='*30, 'Train starting', '='*30)
    main(args)
