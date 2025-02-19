import ml_collections


def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the hyperparameters for the model"""
  config = ml_collections.ConfigDict()
  config.gpu = 0

  # training config
  config.train = d(
      num_steps_train=20000,
      lr=0.003,
      steps_per_logging=20,
      steps_per_large_logging=500,
      steps_per_integration=2000,
      norm_v=True,
      positive_v=True,
  )

  # simulated data
  config.data = d(
      max_dr_trans=3.,
      max_dr_isometry=10.,
      batch_size=4000,
      sigma_data=0.48,
  )

  # model parameter
  config.model = d(
      trans_type='linear_polar',
      num_grid=40,
      num_neurons=24,
      module_size=24,
      sigma=0.07,
      w_trans=1.0,
      w_isometry=10.0,
      s_fixed=5.0,
      num_theta=18,
      activation='',
  )

  return config
