from multiprocessing.spawn import import_main_path
# from models.sync_batchnorm import SynchronizedBatchNorm2d
from models.psp import pSp
from options.train_options import TrainOptions
from training.criteria import adv_loss
import training

# opts = TrainOptions().parse()
# net = pSp(opts).to('cuda:1')