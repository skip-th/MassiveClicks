# Multi-GPU EM-based Click Model training

This project is an extension of **ParClick**, an EM-based generic algorithm for
training click models using a CPU, which incorporates one or multiple GPUs in
the training process. The following click models are supported:

1. *Position-based Model (PBM)*.
2. *User Browsing Model (UBM)*.
3. *Click Chain Model (CCM)*.
4. *Dynamic Bayesian Network Model (DBN)*.

## TODO

- Update function descriptions with new arguments.
- Change add_to_values() to += operator.
- Compile using CMake instead of Makefile.
