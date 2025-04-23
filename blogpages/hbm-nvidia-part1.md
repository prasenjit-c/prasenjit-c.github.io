**Part 1 - Fundamentals of NVIDIA Memory Hierarchy and NSight Compute**

## A Sample CUDA FMA Program

The CUDA kernel performs an element-wise Fused Multiply-Accumulate (FMA) operation.

### Kernel Functionality and Memory Access:

For each element `i`, it performs `OP` FMA operations, utilizing distinct scalar values from the `user_arg[o]` array. Essentially, it accumulates `OP` products of `x[i] * user_arg[o]` into `y[i]`, effectively computing:
$$y[i] = y[i] + x[i] \cdot user\_arg[0] + x[i] \cdot user\_arg[1] + ... + x[i] \cdot user\_arg[OP-1]$$
Each thread processes multiple elements, spaced apart by a `stride`, ensuring comprehensive coverage of the input data.

### Memory Access Patterns:

The kernel's memory access pattern involves:
* Reading from `x[i]`, `y[i]`, and `user_arg[o]`
* Writing the accumulated result back to `y[i]` after each FMA operation

Compiler optimizations can significantly impact the efficiency of these memory accesses. While a deeper exploration of these optimizations is beyond the scope of this discussion, it is worth noting that, using NVCC 12.xx, the generated SASS code (as shown in <<PICTURE>>) consists of three loads followed by an FMA operation and the corresponding store.
