** **
## Part 1 - Fundamentals of NVIDIA Memory Hierarchy and NSight Compute**

The first part of this series is primarily targeted at beginners, covering the basics of NVIDIA GPU architecture and introducing the NVIDIA Nsight Profiler. If you’re already familiar with NVIDIA’s memory hierarchy and the usage of the Nsight Profiler, feel free to skim the performance statistics presented at the end of this part or skip directly to the sections that pique your interest.

However, if you’re new to these concepts or want to refresh your understanding of CUDA, Nsight, and NVIDIA GPU fundamentals, this is an excellent starting point. We'll take a practical approach, using a simple yet representative program. By profiling it under various configurations, we'll unravel the intricacies of the GPU memory hierarchy step by step.

### A Sample CUDA FMA Program

You can find the source code for this program in my [Github Repository](https://github.com/prasenjit-c/cuda-tests/blob/2ea6a92573dea9a803b92e6111a20e630116ea13/ami_measure.cu)

The CUDA kernel performs an element-wise Fused Multiply-Accumulate (FMA) operation.

#### Kernel Functionality and Memory Access:

For each element `i`, it performs `OP` FMA operations, utilizing distinct scalar values from the `user_arg[o]` array. Essentially, it accumulates `OP` products of `x[i] * user_arg[o]` into `y[i]`, effectively computing:
$$y[i] = y[i] + x[i] \cdot user\_arg[0] + x[i] \cdot user\_arg[1] + ... + x[i] \cdot user\_arg[OP-1]$$
Each thread processes multiple elements, spaced apart by a `stride`, ensuring comprehensive coverage of the input data.

#### Memory Access Patterns:

The kernel's memory access pattern involves:
* Reading from `x[i]`, `y[i]`, and `user_arg[o]`
* Writing the accumulated result back to `y[i]` after each FMA operation

Compiler optimizations can significantly impact the efficiency of these memory accesses. While a deeper exploration of these optimizations is beyond the scope of this discussion, it is worth noting that, using NVCC 12.xx, the generated SASS code consists of three loads followed by an FMA operation and the corresponding store.
![](/images/Simple-FMA-Cuda.png "FMA SASS")
