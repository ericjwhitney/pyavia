#!/usr/bin/env python3

# Examples of stress concentration factor along the bore of straight or
# countersunk holes.  Reproduces results of NASA-TP-3192 Figure 4, 7(a) and
# 7(b)

# Written by: Eric J. Whitney  Last updated: 9 April 2020

import numpy as np
import matplotlib.pyplot as plt

from pyavia.struct import kt_hole3d

# ----------------------------------------------------------------------------

bt, rw = 1.0, 1 / 5
rt, zt = [2.5, 1.5, 1.0, 0.5, 0.25], np.linspace(-0.5, +0.5, 100)
scf_ss = np.zeros((len(rt), len(zt)))
labels = []
for i, rt_i in enumerate(rt):
    for j, zt_j in enumerate(zt):
        scf_ss[i, j] = kt_hole3d(rt_i, bt, zt_j, rw, 'tension')
    labels.append(f"$r/t = {rt_i:.2f}$")

plt.figure(1)  # NASA-TP-3192 Figure 4.
for y in scf_ss:
    plt.plot(zt, y)
plt.xlim((0.0, 0.5))
plt.ylim((2.6, 3.4))
plt.xlabel('$z/t$')
plt.ylabel('$K_t$')
plt.title('Tension SCF Along Bore - Countersunk')
plt.legend(labels)
plt.grid()

# ----------------------------------------------------------------------------

rt, rw = 2.0, 1 / 7.5
bt = np.linspace(0.0, 0.75, 4)
scf_cs_t = np.zeros((len(bt), len(zt)))
scf_cs_b = np.zeros((len(bt), len(zt)))
labels = []
for i, bt_i in enumerate(bt):
    for j, zt_j in enumerate(zt):
        scf_cs_t[i, j] = kt_hole3d(rt, bt_i, zt_j, rw, 'tension')
        scf_cs_b[i, j] = kt_hole3d(rt, bt_i, zt_j, rw, 'bending')
    labels.append(f"$b/t = {bt_i:.2f}$")

plt.figure(2)  # NASA-TP-3192 Figure 7(a).
for y in scf_cs_t:
    plt.plot(zt, y)
plt.xlim((-0.5, 0.5))
plt.ylim((1.5, 4.5))
plt.xlabel('$z/t$')
plt.ylabel('$K_t$')
plt.title('Tension SCF Along Bore - Plain')
plt.legend(labels)
plt.grid()

plt.figure(3)  # NASA-TP-3192 Figure 7(b).
for y in scf_cs_b:
    plt.plot(zt, y)
plt.xlim((-0.5, 0.5))
plt.ylim((-3, 3))
plt.xlabel('$z/t$')
plt.ylabel('$K_b$')
plt.title('Bending SCF Along Bore - Countersunk')
plt.legend(labels)
plt.grid()
plt.show()


