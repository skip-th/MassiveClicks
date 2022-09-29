# GPU-based Click Model training

This project is an extension of **ParClick**, an EM-based generic algorithm for
training click models using a CPU, which incorporates one or multiple GPUs in
the training process. The following click models are supported:

1. *Position-based Model (PBM)*.
2. *User Browsing Model (UBM)*.
3. *Click Chain Model (CCM)*.
4. *Dynamic Bayesian Network Model (DBN)*.

## TODO

- Retrieve distributed parameters and combine them on the root node into a complete click model.
- Add check to see if data will fit in gpu memory.
- Perform parallel sum reduction using CUDA Thrust.
- Combine ClickModel_Host and _Dev.
- Remove parameter reference functions.