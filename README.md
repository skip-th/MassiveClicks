# Par? v3

**Par? v3** is an expanded version of **ParPBM v2**. Version 3 incorporates the
following click models:

1. *Position-based Model (PBM)*. (reuses the implementation from version 2)
2. *User Browsing Model (UBM)*. (unfinished)
3. *Click Chain Model (CCN)*. (unfinished)
4. *Dynamic Bayesian Network Model (DBN)*. (unfinished)

## TODO

- Implement UBM.
- Implement CCM.
- Implement DBN.
- Retrieve distributed parameters and combine them on the root node into a complete click model.
- Add check to see if data will fit in gpu memory.
- Use CMake.
- Potentially combine ClickModel_Host and _Dev.
- Change ClickModel_Host to _Hst.
