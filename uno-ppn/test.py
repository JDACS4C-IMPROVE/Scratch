import os
from mpi4py import MPI
import tensorflow as tf


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rankl = int(os.environ.get('PALS_LOCAL_RANKID', '0'))
print(f"Hello from rank {rank}/{size} and local rank {rankl}",flush=True)

#print("rank ",rank, tf.__version__,flush=True)
#print("rank ",rank, tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('XPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[rankl], 'XPU')

N = 32768
a = tf.random.uniform((N, N), minval=0.0, maxval=1.0, dtype=tf.float64)
b = tf.random.uniform((N, N), minval=0.0, maxval=1.0, dtype=tf.float64)
result = tf.matmul(a, b)
comm.Barrier()
if rank == 0: print('Completed matmul',flush=True)
