# MassiveClicks: A Massively-parallel Framework for Efficient Click Models Training

MassiveClicks is a multi-node multi-GPU framework for training click models.
The framework supports heterogeneous GPU architectures, variable numbers of
GPUs per node, and allows for multi-node multi-core CPU-based training when no
GPUs are available. The following click models are currently supported:

1. *Position-based Model (PBM)*.
2. *User Browsing Model (UBM)*.
3. *Click Chain Model (CCM)*.
4. *Dynamic Bayesian Network Model (DBN)*.

MassiveClicks builds upon the generic EM-based algorithm for CPU-based click
model training [ParClick](https://github.com/uva-sne/ParClick) as a base.
