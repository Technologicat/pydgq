# -*- coding: utf-8 -*-
"""MPI support wrapper using mpi4py as backend.

License: 2-clause BSD; copyright 2012-2017 Juha Jeronen and University of Jyväskylä.

To load MPI, import this module. The module is designed so that it can
always be imported, regardless of whether mpi4py is installed.

The availability of the mpi4py library can then be queried at runtime
using has_mpi().

Some rudimentary wrappers (get_size(), get_rank(), get_comm_world())
are provided; all the rest should be done manually using mpi4py,
and only if has_mpi() returns True. This way, the same code can
run both with and without MPI.

See also gather_varlength_array(), a buffer-based fast gather
for rank-1 NumPy arrays of varying lengths (like gatherv,
but determines the sizes automatically).

Example:

import mpi_shim

MPI    = mpi_shim.get_mpi()         # either a reference to mpi4py.MPI, or None if n/a
nprocs = mpi_shim.get_size()        # actual value from mpi4py, or 1    if n/a
my_id  = mpi_shim.get_rank()        # actual value from mpi4py, or 0    if n/a
comm   = mpi_shim.get_comm_world()  # actual value from mpi4py, or None if n/a

# The rest works as usual ( adapted from http://mpi4py.scipy.org/docs/usrman/tutorial.html ):

data = (my_id+1)**2  # <insert meaningful calculation here>
if MPI is not None:
   data = comm.gather(data, root=0)

if my_id == 0:
   for i in range(nprocs):
       assert data[i] == (i+1)**2
else:
   assert data is None
"""

from __future__ import division, print_function, absolute_import

import numpy as np  # used in gather_varlength_array()

try:
    import mpi4py.MPI as MPI

    __comm_world = MPI.COMM_WORLD
    __nproc = MPI.Comm.Get_size(__comm_world)
    __my_id = MPI.Comm.Get_rank(__comm_world)

    __library_ok = True

    # Considering the code that uses this module,
    # this is maybe more sensible than always True.
    #
    __mpi_available = (__nproc > 1)
except ImportError:
    __comm_world = None
    __nproc = 1
    __my_id = 0

    __library_ok = False
    __mpi_available = False

# version of mpi_shim
#
# Especially important, since this module is to be copied to each project that uses it.
#
__version__ = "1.0.0"


##########################################
# Wrapper functions
##########################################

def has_mpi4py_library():
    """Return whether the mpi4py library is available (bool).

    This is just a simple library loadedness check.

    If you want to also see whether there are at least 2 MPI processes
    in the current group (and hence parallel processing makes sense),
    use has_mpi() instead.

    """
    return __library_ok

def has_mpi():
    """Return True if the mpi4py library is available and there are at least
    2 MPI processes in the current group. Otherwise return False.

    This is a "does parallel processing make sense?" check.

    See has_mpi4py_library() for just checking library availability.

    """
    return __mpi_available

def get_mpi():
    """Return a reference to the loaded mpi4py.MPI instance.

    If mpi4py is not available, return None.
    """
    if __library_ok:
        return MPI
    else:
        return None

def get_size():
    """Return the size (int) of the MPI group.

    See also:
        get_rank()    (get the ID of the currently running instance)
    """
    return __nproc

def get_rank():
    """Return the MPI rank (int) of the running instance.

    The rank is the identifier of the process, numbered 0, 1, ..., get_size()-1.

    See also:
        get_size()    (get number of running instances)
    """
    return __my_id

def get_comm_world():
    """Return mpi4py.MPI.COMM_WORLD (reference) if mpi4py is available, or None if n/a."""
    return __comm_world


###################################
# MPI-aware printing
###################################

def mpi_print(*args):
    """Print args, but only if running in the root process.

    See also:
        mpi_allprint()    (gather args from all MPI ranks, print in root process)
    """
    if __my_id == 0:  # also true if no MPI
        print(*args)

def mpi_allprint(*args):
    """Gather args from all MPI ranks, print them all in the root process.

    The root process makes a separate call to built-in print for args from each rank.

    MPI rank of the messages is not reported; add this data to the message yourself if desired.

    See also:
        mpi_print()    (print only in the root process)
    """
    comm = get_comm_world()

    if comm is None:  # no MPI
        print(*args)
        return

    allargs = comm.gather(args, root=0)
    if get_rank() == 0:
        for args in allargs:
            print(*args)
    # else no-op


###################################
# Helpers
###################################

def gather_varlength_array(data, datatype, do_allgather=False):
    """Fast (buffer-based) MPI gather for variable-sized data using NumPy arrays.

    Data type of all items must be the same. It must be a primitive datatype
    that mpi4py.MPI.Gather() supports.

    Parameters:
        data : Python list or rank-1 array
            Local data items in each process, to be gathered at the root process.
        datatype : NumPy dtype
            Datatype specification (e.g. np.int, np.float64, np.complex128, ...).
        do_allgather : bool, optional
            If True, do an Allgather() instead of a Gather().

    In different processes, `data` is allowed to have a different number of elements,
    but `datatype` must be the same.

    `datatype` is mandatory so that zero-length input can be handled properly.

    Return value:
        In the root process:
            tuple (data, I), where
                data : rank-1 np.array
                    flattened data array, with data from all processes
                    concatenated in order of increasing MPI rank.
                I : rank-1 np.array of length get_size()+1
                    data from process j is located in data[ I[j] : I[j+1] ].
                    The end fencepost for the last process is provided to avoid
                    the need for special-casing in calling code.

        In all other processes:
            If do_allgather is True, same as in root process, otherwise None.
    """
    # Use an efficient buffer transfer for gathering the data.
    #
    # Because the amount of data per process varies, we must first allgather the data lengths
    # to determine a suitable buffer size. We will use the maximum size in all processes.
    #
    nprocs = get_size()
    data_lengths = np.empty( [nprocs], dtype=int )
    n_local_items = np.size(data)
    n_local_items_array = np.array( n_local_items, dtype=int )  # rank-1 array of length 1

    def lengths_to_offsets(lengths):
        return np.concatenate( ( [0], np.cumsum(lengths) ) )

    comm = get_comm_world()
    comm.Allgather( n_local_items_array, data_lengths )  # order of parms: sendbuf, recvbuf

    # Any leftover array elements are simply unused; we use data_lengths to determine
    # which elements to read.
    #
    max_entries_per_process = np.max(data_lengths)
    nentries = nprocs*max_entries_per_process

    # Check special case: nothing to do if no data
    #
    if max_entries_per_process == 0:
        if get_rank() == 0  or  do_allgather:
            return (np.empty( [0], dtype=datatype ), lengths_to_offsets(data_lengths))
        else:
            return None

    # Prepare receive buffer at root process
    # (or at all processes if allgathering)
    #
    if get_rank() == 0  or  do_allgather:
        data_recvbuf = np.empty( [nentries], dtype=datatype )
    else:
        data_recvbuf = None

    # Prepare and fill send buffers in all processes
    #
    data_sendbuf = np.empty( [max_entries_per_process], dtype=datatype )
    data_sendbuf[0:n_local_items] = np.array( data, dtype=datatype )

    if do_allgather:
        comm.Allgather( data_sendbuf, data_recvbuf )
    else:
        comm.Gather( data_sendbuf, data_recvbuf, root=0 )

    # In the root process, extract the data.
    # (or in all processes if we are allgathering)
    #
    if get_rank() == 0  or  do_allgather:
        result = np.empty( [np.sum(data_lengths)], dtype=datatype )

        # This is similar to util.flatten_list_of_arrays(), but from a single np.array buffer.
        offs_out = 0
        for nproc in range(nprocs):
            offs_in = nproc*max_entries_per_process
            datalen = data_lengths[nproc]
            if datalen > 0:
                result[offs_out:(offs_out+datalen)] = data_recvbuf[offs_in:(offs_in+datalen)]
                offs_out += datalen
        assert( offs_out == np.sum(data_lengths) )
        return (result, lengths_to_offsets(data_lengths))
    else:
        return None

def allgather_varlength_array(data, datatype):
    """Same as gather_varlength_array(data, datatype, do_allgather=True)."""
    return gather_varlength_array(data, datatype, do_allgather=True)

