OPENQASM 3.0;
include "stdgates.inc";
bit[2] creg_0;
qubit[2] qreg_1;
h qreg_1[0];
cx qreg_1[0], qreg_1[1];
creg_0[0] = measure qreg_1[0];
creg_0[1] = measure qreg_1[1];
