# PyFR-with-overset-ale
 overset and ale been integrated into PyFR

New module added in solvers/moving_grid solvers/overset
Overset is accomplished by tioga. Only support HEX.
Currently only tested for GPU.
CPU possibly won't work fully.

The current code only works for at least two CORES for overset meshes. Here
are some tips to debug a parallel code (MPI) to avoid excessive printf sentences.
To debug a PYTHON code running with MPI which calls C++ library, one can use
> mpiexec -n xterm -e "python -m pdb command to run code".

n xterm terminals will pop up with pdb debugging mode to allow you to check the information process by process.

To debug the tioga library with parallel computing, you can put some lines in the C++ code as
```
int pid = getpid();
prntf("current pid is %d\n",pid);
int idebugger = 1;
while(idebugger) {

};
```

When TIOGA is called, the code will halt on the while loop with pid being printed on the console.

Here, you can open gdb and do the following steps for every process
* attach pid
* set idebugger =0
* b /* at some point you want to break in the c++ code*/
* c /* continue to the break point*/

Then you can use gdb debug each process normally.

For Overset boundary, one have to use *overset_* to name it
