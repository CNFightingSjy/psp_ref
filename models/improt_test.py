from multiprocessing.spawn import import_main_path
from sync_batchnorm import SynchronizedBatchNorm2d
from psp import pSp

net = pSp().to('cuda:1')