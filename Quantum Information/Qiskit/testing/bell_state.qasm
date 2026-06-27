OPENQASM 3.0;
include "stdgates.inc";
bit[2] creg_0;
qubit[2] qreg_1;
h qreg_1[0];
cx qreg_1[0], qreg_1[1];
