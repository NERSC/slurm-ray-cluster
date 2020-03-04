from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import (train, test, get_data_loaders, ConvNet)

class TrainMyModel(tune.Trainable):
  def _setup(self, config):
    pass

  def _train(self):
    # this is currently needed to handle Cori GPU multiple interfaces
    self.current_ip()
    ...
    return {"mean_accuracy": acc}

  def _save(self, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    torch.save(self.model.state_dict(), checkpoint_path)
    return checkpoint_path

  def _restore(self, checkpoint_path):
    self.model.load_state_dict(torch.load(checkpoint_path))

  # this is currently needed to handle Cori GPU multiple interfaces
  def current_ip(self):
    import socket
    hostname = socket.getfqdn(socket.gethostname())
    self._local_ip = socket.gethostbyname(hostname)
    return self._local_ip 

if __name__ == "__main__":
  # ip_head and redis_passwords are set by ray cluster shell scripts
  print(os.environ["ip_head"], os.environ["redis_password"])
  ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0], redis_password=os.environ["redis_password"])
  sched = ASHAScheduler(metric="mean_accuracy")
  analysis = tune.run(TrainMyModel,
                      scheduler=sched,
                      stop={"mean_accuracy": 0.99,
                            "training_iteration": 100},
                      resources_per_trial={"cpu":10, "gpu": 1},
                      num_samples=128,
                      checkpoint_at_end=True,
                      config={"lr": tune.uniform(0.001, 1.0),
                              "momentum": tune.uniform(0.1, 0.9),
                             "use_gpu": True})
  print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
