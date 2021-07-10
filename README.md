# PyFR-with-overset-ale
 overset and ale been integrated into PyFR

New module added in solvers/moving_grid solvers/overset
Overset is accomplished by tioga. Only support HEX.
Currently only tested for GPU.
CPU possibly won't work fully.

The current code only works for at least two CORES for overset meshes.
To debug a PYTHON code running with MPI which calls C++ library, one can use
mpiexec -n xterm -e "python -m pdb command to run code".

n xterm terminal will pop up with pdb debugging mode to allow you to check the information.

To debug the tioga library with parallel computing, one can put some lines in the C++ code as

int pid = getpid();
prntf("current pid is %d\n",pid)
int idebugger = 1;
while(idebugger) {

};

When TIOGA is called, the code will halt on the while loop with pid being printed on the console.

Here, one can open gdb
> attach pid
Then one can use gdb debug each process normally.
