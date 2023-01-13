"""

Taken from: https://github.com/rapidsai/dask-cuda/blob/branch-0.18/dask_cuda/benchmarks/local_cudf_merge.py

This is the cuDF merge benchmarking application. It has been 
modified to work for the MVAPICH2-based (http://mvapich.cse.ohio-state.edu) 
communication backend for the Dask Distributed library using the 
dask-mpi package.

"""
import os
#adjust as per compute system
GPUS_PER_NODE      =   1    # number of GPUs in the system
RUNS               =  20    # repititions for the benchmark
DASK_INTERFACE     = 'ib0'  # interface to use for communication
DASK_PROTOCOL      = 'mpi'  # protocol for Dask Distributed. Options include ['mpi', 'tcp']
THREADS_PER_NODE   =  8    # number of threads per node.

rank = os.environ['MV2_COMM_WORLD_LOCAL_RANK']
device_id = int(rank) % GPUS_PER_NODE
os.environ["CUDA_VISIBLE_DEVICES"]=str(device_id)

import mpi4py
from mpi4py import MPI
import mpi4py.rc
mpi4py.rc.threads = True
from dask_mpi import initialize

import math
from collections import defaultdict
from time import perf_counter as clock
import time
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes

def generate_chunk(i_chunk, local_size, num_chunks, chunk_type, frac_match, gpu):
    #print("generate_chunk")
    # Setting a seed that triggers max amount of comm in the two-GPU case.
    if gpu:
        import cupy as xp

        import cudf as xdf
    else:
        import numpy as xp
        import pandas as xdf

    xp.random.seed(2 ** 32 - 1)

    chunk_type = chunk_type or "build"
    frac_match = frac_match or 1.0
    if chunk_type == "build":
        # Build dataframe
        #
        # "key" column is a unique sample within [0, local_size * num_chunks)
        #
        # "shuffle" column is a random selection of partitions (used for shuffle)
        #
        # "payload" column is a random permutation of the chunk_size

        start = local_size * i_chunk
        stop = start + local_size

        parts_array = xp.arange(num_chunks, dtype="int64")
        suffle_array = xp.repeat(parts_array, math.ceil(local_size / num_chunks))
        
        df = xdf.DataFrame(
            {
                "key": xp.arange(start, stop=stop, dtype="int64"),
                "shuffle": xp.random.permutation(suffle_array)[:local_size],
                "payload": xp.random.permutation(xp.arange(local_size, dtype="int64")),
            }
        )
    else:
        #print("chunk type is other")
        # Other dataframe
        #
        # "key" column matches values from the build dataframe
        # for a fraction (`frac_match`) of the entries. The matching
        # entries are perfectly balanced across each partition of the
        # "base" dataframe.
        #
        # "payload" column is a random permutation of the chunk_size

        # Step 1. Choose values that DO match
        sub_local_size = local_size // num_chunks
        sub_local_size_use = max(int(sub_local_size * frac_match), 1)
        arrays = []
        for i in range(num_chunks):
            bgn = (local_size * i) + (sub_local_size * i_chunk)
            end = bgn + sub_local_size
            ar = xp.arange(bgn, stop=end, dtype="int64")
            arrays.append(xp.random.permutation(ar)[:sub_local_size_use])
        key_array_match = xp.concatenate(tuple(arrays), axis=0)

        # Step 2. Add values that DON'T match
        missing_size = local_size - key_array_match.shape[0]
        start = local_size * num_chunks + local_size * i_chunk
        stop = start + missing_size
        key_array_no_match = xp.arange(start, stop=stop, dtype="int64")

        # Step 3. Combine and create the final dataframe chunk (dask_cudf partition)
        key_array_combine = xp.concatenate(
            (key_array_match, key_array_no_match), axis=0
        )

        df = xdf.DataFrame(
            {
                "key": xp.random.permutation(key_array_combine),
                "payload": xp.random.permutation(xp.arange(local_size, dtype="int64")),
            }
        )
    return df

def get_random_ddf(chunk_size, num_chunks, frac_match, chunk_type, device):

    parts = [chunk_size for i in range(num_chunks)]
    device_type = True if device == "gpu" else False
    meta = generate_chunk(0, 4, 1, chunk_type, None, device_type)
    divisions = [None] * (len(parts) + 1)

    name = "generate-data-" + tokenize(chunk_size, num_chunks, frac_match, chunk_type)

    graph = {
        (name, i): (
            generate_chunk,
            i,
            part,
            len(parts),
            chunk_type,
            frac_match,
            device_type,
        )
        for i, part in enumerate(parts)
    }

    ddf = new_dd_object(graph, name, meta, divisions)

    if chunk_type == "build":
        divisions = [i for i in range(num_chunks)] + [num_chunks]
        return ddf.set_index("shuffle", divisions=tuple(divisions))
    else:
        return ddf

def setup_memory_pool(pool_size=1e10, disable_pool=False):
    import cupy

    os.environ['RMM_NO_INITIALIZE'] = 'True'
    import rmm

    rmm.reinitialize(
        pool_allocator=not disable_pool, devices=0, initial_pool_size=pool_size,
    )
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

def merge(ddf1, ddf2):
    broadcast = False

    # The merge/join operation
    ddf_join = ddf1.merge(ddf2, on=["key"], how="inner", broadcast=broadcast)
    wait(ddf_join.persist())

def run(client, n_workers, device="gpu"):
    chunk_size = 1e6
    frac_match = 0.3
    # Generate random Dask dataframes
    ddf_base = get_random_ddf(
        chunk_size, n_workers, frac_match, "build", device
    ).persist()

    ddf_other = get_random_ddf(
        chunk_size, n_workers, frac_match, "other", device
    ).persist()

    wait(ddf_base)
    wait(ddf_other)

    took_list = []
    for i in range(RUNS):
        start = time.time()
        merge(ddf_base, ddf_other)
        if device=='gpu':
            client.run(lambda xp: xp.cuda.Device().synchronize(), cupy)
        duration = time.time() - start
        took_list.append( (duration, 0) )
        print("Time for iteration", i, ":", duration)
        sys.stdout.flush()


if __name__ == '__main__':
    initialize(
        interface=DASK_INTERFACE,
        protocol=DASK_PROTOCOL,
        nthreads=8,
        comm=MPI.COMM_WORLD,
        local_directory=os.environ.get("TMPDIR", None)
    )

    client = Client()
    run(client, n_workers=2)