""" Main training loop. """
import os

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import ml_collections
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

import input_pipeline
import model as model
import utils

torch.manual_seed(1118)

class Experiment:
  def __init__(self, config: ml_collections.ConfigDict, device):
    self.config = config
    self.device = device

    # initialize models
    logging.info("==== initialize model ====")
    self.model_config = model.GridCellConfig(**config.model)
    self.model = model.GridCell(self.model_config).to(device)

    # initialize dataset
    logging.info("==== initialize dataset ====")
    train_dataset = input_pipeline.TrainDataset(config.data, self.model_config)
    self.train_iter = iter(train_dataset)

    # initialize optimizer
    logging.info("==== initialize optimizer ====")
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr)

  def train_and_evaluate(self, workdir):
    logging.info('==== Experiment.train_and_evaluate() ===')

    if not tf.io.gfile.exists(workdir):
      tf.io.gfile.mkdir(workdir)
    config = self.config.train
    logging.info('num_steps_train=%d', config.num_steps_train)

    writer = metric_writers.create_default_writer(workdir)

    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_steps_train, writer=writer)
    hooks += [report_progress]

    train_metrics = []
    module_size = self.model_config.module_size
    num_grid = self.model_config.num_grid
    num_module = self.model_config.num_neurons // module_size

    logging.info('==== Start of training ====')
    with metric_writers.ensure_flushes(writer):
      for step in range(1, config.num_steps_train+1):
        batch_data = utils.dict_to_device(next(self.train_iter), self.device)

        # lr decay 
        decay_step = 6000 
        if step > decay_step: 
          lr = config.lr - (step - decay_step) * (config.lr / (config.num_steps_train - decay_step))
        elif step < 3000: 
          lr = config.lr * step / 3000
        else: 
          lr = config.lr
          
        for param_group in self.optimizer.param_groups: 
          param_group['lr'] = lr

        self.optimizer.zero_grad()
        loss, metrics_step = self.model(batch_data, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm(parameters=self.model.parameters(), max_norm=10)
        self.optimizer.step()

        # postive v
        if config.positive_v:
          self.model.encoder.v.data = self.model.encoder.v.data.clamp(min=0.)

        # normalize v
        if config.norm_v:
          with torch.no_grad():
            v = self.model.encoder.v.data.reshape(
                (-1, module_size, num_grid, num_grid))

            v_normed = nn.functional.normalize(v, dim=1) / np.sqrt(num_module)
            self.model.encoder.v.data = v_normed.reshape(
                (-1, num_grid, num_grid))
  
        metrics_step = utils.dict_to_numpy(metrics_step)
        train_metrics.append(metrics_step)

        # Quick indication that training is happening.
        logging.log_first_n(
            logging.WARNING, 'Finished training step %d.', 3, step)
        for h in hooks:
          h(step)

        if step % config.steps_per_logging == 0 or step == 1:
          train_metrics = utils.average_appended_metrics(train_metrics)
          writer.write_scalars(step, train_metrics)
          train_metrics = []

        if step % config.steps_per_large_logging == 0:
          # visualize v and heatmaps.
          with torch.no_grad():
            def visualize(weights, name):
              weights = weights.data.cpu().detach().numpy()
              weights = weights.reshape(
                  (-1, module_size, num_grid, num_grid))[:10, :10]
              writer.write_images(step, {name: utils.draw_heatmap(weights)})

            visualize(self.model.encoder.v, 'v')

        if step == config.num_steps_train:
          ckpt_dir = os.path.join(workdir, 'ckpt')
          if not tf.io.gfile.exists(ckpt_dir):
            tf.io.gfile.makedirs(ckpt_dir)
          self._save_checkpoint(step, ckpt_dir)
          
  def _save_checkpoint(self, step, ckpt_dir):
    """
    Saving checkpoints
    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(self.model).__name__
    state = {
        'arch': arch,
        'step': step,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'config': self.config
    }
    filename = os.path.join(ckpt_dir, 'checkpoint-step{}.pth'.format(step))
    torch.save(state, filename)
    logging.info("Saving checkpoint: {} ...".format(filename))

  def _resume_checkpoint(self, resume_path):
    """
    Resume from saved checkpoints
    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    self.logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    self.start_epoch = checkpoint['epoch'] + 1

    # load architecture params from checkpoint.
    if checkpoint['config']['arch'] != self.config['arch']:
      self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                          "checkpoint. This may yield an exception while state_dict is being loaded.")
    self.model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
      self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                          "Optimizer parameters not being resumed.")
    else:
      self.optimizer.load_state_dict(checkpoint['optimizer'])

    self.logger.info(
        "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
