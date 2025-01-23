import os
import pathlib

from mpc_traj_tracker.mpc.mpc_generator import MpcModule
from pkg_motion_model import motion_model
from util.mpc_config import Configurator


### Build mpc_test solver
yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'mpc_test.yaml')
config = Configurator(yaml_fp)

MpcModule(config).build(motion_model.unicycle_model)


### Build mpc_default solver
yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'mpc_default.yaml')
config = Configurator(yaml_fp)

MpcModule(config).build(motion_model.unicycle_model)

### Build mpc_longiter solver
yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'mpc_longiter.yaml')
config = Configurator(yaml_fp)

MpcModule(config).build(motion_model.unicycle_model)