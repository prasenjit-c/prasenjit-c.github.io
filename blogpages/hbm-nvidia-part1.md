** **
## Part 1 - Fundamentals of NVIDIA Memory Hierarchy and NSight Compute

The first part of this series is primarily targeted at beginners, covering the basics of NVIDIA GPU architecture and introducing the NVIDIA Nsight Profiler. If you’re already familiar with NVIDIA’s memory hierarchy and the usage of the Nsight Profiler, feel free to skim the performance statistics presented at the end of this part or skip directly to the sections that pique your interest.

However, if you’re new to these concepts or want to refresh your understanding of CUDA, Nsight, and NVIDIA GPU fundamentals, this is an excellent starting point. We'll take a practical approach, using a simple yet representative program. By profiling it under various configurations, we'll unravel the intricacies of the GPU memory hierarchy step by step.

### A Sample CUDA FMA Program

You can find the source code for this program in my [Github Repository](https://github.com/prasenjit-c/cuda-tests/blob/2ea6a92573dea9a803b92e6111a20e630116ea13/ami_measure.cu)

The CUDA kernel performs an element-wise Fused Multiply-Accumulate (FMA) operation.

#### Kernel Functionality and Memory Access:

For each element `i`, it performs `OP` FMA operations, utilizing distinct scalar values from the `user_arg[o]` array. Essentially, it accumulates `OP` products of `x[i] * user_arg[o]` into `y[i]`, effectively computing:
$$y[i] = y[i] + x[i] * user\_arg[0] + x[i] * user\_arg[1] + ... + x[i] * user\_arg[OP-1]$$
Each thread processes multiple elements, spaced apart by a `stride`, ensuring comprehensive coverage of the input data.

#### Memory Access Patterns:

The kernel's memory access pattern involves:
* Reading from `x[i]`, `y[i]`, and `user_arg[o]`
* Writing the accumulated result back to `y[i]` after each FMA operation

Compiler optimizations can significantly impact the efficiency of these memory accesses. While a deeper exploration of these optimizations is beyond the scope of this discussion, it is worth noting that, using NVCC 12.xx, the generated SASS code consists of three loads followed by an FMA operation and the corresponding store.
![](/images/Simple-FMA-Cuda.png "FMA SASS")

#### Kernel Launch and Thread Management:

The kernel is launched using the following syntax: `run_fma_kernel[numFMAs-1]<<<numBlocks, blockSize>>>(N, x, y, user_arg);`

* **`numBlocks`**: Specifies the total number of thread blocks in the grid.
* **`blockSize`**: Defines the number of threads per block.

The total number of threads launched is $\text{numBlocks} \times \text{blockSize}$. Even if this total is less than $N$, the grid-stride loop within the kernel ensures that all $N$ elements are eventually processed.

### Profiling a CUDA kernel using NSight Compute:

The goal of this blog is not to delve into the specifics of installing or using Nvidia Insight Compute, as I feel that comprehensive guides covering those fundamentals are already well-documented on Nvidia's official website and other resources, making repetition unnecessary. Instead of repeating that material, I have provided links for those who wish to explore it further. However, the goal here is to provide enough working knowledge of Nsight Compute for newcomers to start using the tool confidently, without requiring prior in-depth training. Necessary details will be integrated throughout the relevant sections.
Nvidia Nsight Compute offers flexibility in how you start a profiling session, allowing you to choose based on convenience and system availability. My usual approach is to use the CLI version of Nvidia Insight Compute. I initiate the profiling on the GPU cloud provider's terminal and, once the profiling is complete, I download the profile to the Nvidia Insight Compute application installed on my laptop or desktop. From there, I perform the analysis. Feel free to adopt whichever launch and analysis method integrates best with your own workflow.

```bash
sudo ncu --set full -o <profile-file-name> <executable>

The above command launches a profile with a complete set of events and counters. When starting with profiling, collecting the complete set of performance counters can be a good strategy, as it ensures comprehensive data capture and prevents accidentally overlooking important metrics. However, if you are more experienced or have a specific focus, you can use a targeted set of counters by leveraging the various options provided.Once the profile is collected and loaded into the GUI, you'll typically focus on key sections like 'Computation Analysis' and 'Memory Analysis'. This particular blog post concentrates on the memory hierarchy; therefore, our discussion will center on the findings within the **Memory Analysis** section, setting aside detailed compute optimization strategies for now."

I have listed every command in detail via the link provided below, enabling you to recreate this analysis in your own environment and see the results for yourselves. To illustrate, we start by profiling a highly simplified version of the program on A100 - **configured using a single block, a single thread, 1 MB of memory, and a single FMA instruction (OP = 1)** — produces a profile that, when loaded into the GUI, looks something like this:
