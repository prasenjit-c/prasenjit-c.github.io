---
layout: post
---

** **
## Part 3 - From Theory to Practice: HBM in GEMM Workloads

In the previous part of this series, we explored the architectural characteristics and performance behavior of HBM in NVIDIA GPUs. While that discussion provided a solid theoretical and empirical understanding of HBM performance, this part will focus on a practical application. Specifically, we will examine how HBM performs in one of the most prevalent and critical operations in modern machine learning: GEMM (General Matrix Multiplication).

### Fundamentals of GEMM

Given that the AI/ML community is heavily focused on optimizing GEMM, there’s little value in covering its fundamentals in detail here. If you are unfamiliar with GEMM or wish to better understand its workings, there are countless tutorials and explanations available online. As with other posts in this series, I will include a list of relevant GEMM resources that you may find useful.

For the context of our discussion, a GEMM operation performs the calculation C = αA⋅B + βC. As the figure illustrates, this operation computes each individual element of the output matrix C by taking the dot product of a corresponding row from matrix A and a column from matrix B.

![](/images/hbm-part3-gemm.png "GEMM")

### Roofline Model for GEMM
For simplicity in our roofline analysis, we will assume α=1 and β=0. Let's establish the fundamental roofline calculations (as illustrated in the figure).

For each element of C, there are K FMA operations along the reduction dimension of both matrices A and B. Therefore, the total number of FMAs is:M×N×K

The total elements read from HBM are (M×K)+(K×N), and the total elements written to HBM are M×N.

* Total FLOPs = 2×M×N×K
* Total Memory Transfers = (M×K + K×N + M×N) * sizeof(element)
* $ \text{AMI} = \frac{\text{Total FLOPs}}{\text{Total Memory Transfers}} $
* Asymptotically, $ \text{AMI} \approx \frac{2N^3}{3N^2} \approx O(N) $

This asymptotic analysis reveals that as the size of the matrices increases, GEMM becomes more compute-bound. Consequently, memory bandwidth is typically not a major performance bottleneck for large matrices. However, for smaller matrices, memory access latency and bandwidth play a significant role in overall performance, making it challenging to fully utilize the GPU's compute capabilities.

With the roofline model established, we can now use it as a reference to evaluate how GEMM actually performs on NVIDIA GPUs.

### NVIDIA BLAS

Like other major hardware vendors, NVIDIA provides highly optimized implementations of BLAS (Basic Linear Algebra Subprograms) routines. In the NVIDIA ecosystem, CUTLASS and cuBLAS are the two primary libraries widely adopted as backends for performing GEMM operations across a variety of software platforms.

In this post, our focus will be on analyzing GEMM performance using cuBLAS.

For further details, you can refer to the official [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html) and the [GEMM function API reference](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm).

To conduct this analysis, I created a GEMM program based on NVIDIA's [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples.git). My modified version, which includes switches for profiling and performance measurement, is available in my [GitHub repository](https://github.com/prasenjit-c/cuda-tests/tree/main/gemm).

### Performance Breakdown: GEMM on A100 and H200
Now it’s time to move from setup to real measurements. There are many ways to evaluate GEMM performance, but to keep this analysis focused, I considered two key dimensions for the GEMM matrices:
1. Value patterns of M, N, and K – Whether they are “nice” powers of two or prime numbers.
2. Matrix shapes – Whether they are square or “skinny” (highly rectangular).

These two factors are critical because they directly influence how the GPU's tiling and memory access strategies are formed, ultimately affecting overall utilization.

The figure below illustrates the classification of GEMM test cases used in this study.

![](/images/hbm-part3-gemm-categories.png "GEMM Categories")

With the A100 delivering 19.5 TFLOPS/sec in FP32 performance and 1,555 GB/sec of HBM2e bandwidth, and the H200 offering 67 TFLOPS/sec and 4,000 GB/sec of HBM3e bandwidth, we can use the roofline model to establish the performance bounds for GEMM.

|Gemm Type|M|N|K|FMA|Memory Read (MB)|Memory Write (MB)|AMI|
|----------------------|---- |----|----|----|----|----|----|
|Square-Pow2|8,192|8,192|8,192|549,755,813,888|758|256|1,024|
|Square-Prime|10,007|10,007|10,007|1,002,101,470,343|1,146|382|1,251|
|Skinny-Pow2|32,768|256|16,384|137,438,953,472|2,096|32|123|
|Skinny-Prime|30,011|307|17,497|161,206,457,369|2,059|35|147|

|Gemm|A100 ||H200 ||
||T-Compute (msec)|T-Memory (msec)|T-Compute (msec)|T-Memory (msec)|
|---|----------------------------|----------------|----------------|---------------|
|Square-Pow2|56.4|0.66|16.4|0.26|
|Square-Prime|102.8|0.98|29.9|0.38|
|Skinny-Pow2|14.1|1.37|4.1|0.53|
|Skinny-Prime|16.5|1.35|4.8|0.52|

As the table indicates, GEMM is predominantly compute-bound, with memory transfer time contributing only a negligible portion to the overall execution time. The next step is to verify how well this theoretical expectation holds when running on actual GPUs.

The figure below presents the measured execution times for the four GEMM variants on the A100 and H200 GPUs. For both architectures, the measured results follow the general trend predicted by the roofline model. The two lines in the figure represent the achieved efficiency on each GPU, plotted against the secondary axis.
![](/images/hbm-part3-gemm-perf.png "GEMM Performance")

A key observation is that square matrices consistently outperform their skinny counterparts. More importantly, GEMM variants with dimensions that are a power of two significantly outperform those with prime-sized dimensions. This is clearly depicted by the efficiency lines, where the performance of the "skinny prime" variant shows a notable drop from its theoretical roofline.
Another interesting finding is the performance deviation between the two GPUs. While the H200 performs approximately 2.4-2.7x better than the A100, its efficiency is considerably lower than the theoretical 3.4x performance improvement suggested by the specifications.
We will investigate these deviations in detail to understand where the performance bottlenecks arise and how HBM contributes to or limits GEMM throughput.

As is consistent with this blog post, we will now invoke NSight Compute to profile our GEMM workloads and gain a deeper understanding of their performance, with a special focus on memory behavior.

The table below summarizes the kernel invocations, showing how cuBLAS configures the GPU and tiles the workload into thread blocks and grids. As seen here, the choice of grid size is influenced by both the GPU architecture and the shape and size of the matrices. The only consistent element across all configurations is that each thread block contains 256 threads, regardless of the specific grid dimensions. This distribution is determined by cuBLAS's internal algorithm, which evaluates multiple parameters—including matrix dimensions, tiling strategies, and GPU hardware limits to decide the optimal grid configuration and launches specific kernels to achieve the best possible performance.
![](/images/hbm-part3-kernal-invocation.png "Kernel Invocation")

Next, let’s compare the actual amount of compute and memory transfer performed. The table below shows the number of FMAs and the load/store operations to HBM for both GPU architectures, normalized to their respective roofline calculations.
![](/images/hbm-part3-perf-compare-roofline.png "vs Roofline")

The number of FMAs aligns almost exactly with the roofline model, indicating that there are no superfluous computations. This suggests that the SASS code produced and the overall strategy employed are highly efficient in terms of computational volume. However, the same consistency is not observed for memory transfers. For both GPUs, the total amount of memory moved is significantly higher than the theoretical size of the input matrices. The only exception is the Square-Pow2 GEMM case on the H200. The HBM store volume for the output matrix closely matches the roofline expectation. This suggests that cuBLAS tiles the output matrix and, for each tile, completes the full computation before writing it back to HBM. The large observed HBM transfers cause the Arithmetic Intensity (AI) to drop well below the roofline value. Even with this drop the oevrall achieved AMI is sufficiently high.

We also examine the warp instruction latency profile for GEMM and compare it with the memory-bound workload from the sample kernel analyzed in Part 1 of this blog. The chart below shows the per-instruction latency distribution for both kernels. For the GEMM kernel, the overall instruction latency is remarkably low at 3.4 cycles, with stalls primarily attributed to the Math pipeline (0.74 cycles). The L1 cache and Shared Memory contribute a negligible 0.08 cycles to stalls, thanks to effective latency hiding and data reuse. In stark contrast, the sample memory-bound kernel on the right shows a significantly higher overall instruction latency of 39.8 cycles. In this case, stalls are overwhelmingly dominated by memory, accounting for 34.8 cycles. This contrast highlights the fundamentally compute-bound nature of GEMM, where math pipelines dominate execution, versus the memory-bound kernel where performance is dictated almost entirely by memory latency.

![](/images/hbm-part3-warp-latency.png "Warp Instruction Latency Distribution")

This behavior demonstrates a crucial optimization: for compute-bound dense GEMM kernels, the underlying architecture leverages memory bandwidth to perform multiple data transfers in the shadow of computation, ensuring the compute units are continuously busy and memory never becomes the critical performance bottleneck.

### The Role of HBM in GEMM
Let’s try to understand how much performance advantage HBM actually provides in GEMM. This is not straightforward to generalize, as observed large dense GEMM operations are primarily compute-bound rather than memory-bound. A deeper look into the memory hierarchy reveals how crucial HBM is in enabling this performance.

The figure below shows the compute and memory hierarchy throughput utilization. While compute throughput achieves a high utilization of 85–95%, the memory hierarchy utilization progressively declines: L1 delivers 45–55%, L2 about 15–20%, and finally HBM only 5–10%. This indicates that as the matrices are tiled and fetched from HBM, the underlying cuBLAS algorithm maximizes FLOPs per tile, ensuring that compute remains saturated.
![](/images/hbm-part3-throughput.png "Throughput Utilization")

The table below, showing the memory hierarchy hit rates, provides further evidence of this strategy. Relating this data back to the roofline analysis:
* The result matrix tiles are pinned in L2 and flushed back to HBM only once their final values are computed. This is supported by the 100% L2 store hit rate, and the absence of superfluous store traffic to HBM compared to roofline expectations.
* The high L2 load hit rate indicates that although input tiles are fetched multiple times, each tile is extensively reused before eviction, amortizing HBM traffic.
![](/images/hbm-part3-hit-rates.png "Hit Rate")

This analysis highlights two crucial architectural points:
1. Matrices are effectively blocked with the L2 cache capacity in mind.
2. This tiling strategy produces a highly skewed HBM read-to-write ratio of ~95:5%. This observation could provide valuable insights for memory architects seeking to optimize future designs.

The L1 hit rates revealed some non-intuitive patterns that warrant deeper investigation. These results suggest a complex interplay between warp scheduling, register pressure, and shared memory usage during GEMM execution, a topic that lies beyond the scope of this particular analysis.

#### A100 vs. H200: cuBLAS Kernel Behavior
Differences between the approaches employed by cuBLAS for the A100 and H200 is revealed in the memory chart as shown below for the Square-Prime case.

* Asynchronous Memory Copy: A crucial distinction is the use of memcpy_async. The H200 utilizes this asynchronous memory copy operation, while the A100 does not.
* Tiling Strategy in Shared Memory: A100 uses conventional load instructions through L1 whereas H200 uses a hybrid approach — cuda::memcpy_async + conventional L1 loads.

![](/images/hbm-part3-mem-chart.png "Memory Chart")

The last factor impacting performance is the "tail effect" we touched on previously. For instance, in the Skinny-Prime case on H200, this effect manifests clearly—resulting in reduced compute utilization, since the last partial wave of thread blocks underutilizes available SMs.

## Bottom Line
For large dense GEMM kernels, the limiting factor is not memory bandwidth but compute throughput. HBM contributes only marginally (5–10% utilization), with most of the heavy lifting handled by L2 cache reuse and smart tiling strategies. This explains why GEMM can sustain such high arithmetic intensity — data transfers from HBM are efficiently overlapped with computation so the math pipelines stay busy.

The H200 improves raw performance significantly over the A100, but it still falls short of the full theoretical gain. The gap highlights that algorithm–architecture alignment (e.g., how cuBLAS uses cuda::memcpy_async and how tiles map to cache hierarchies) is just as important as raw FLOPs or HBM bandwidth. This efficiency drop illustrates why top AI labs (e.g., OpenAI, DeepSeek) often develop their own optimized GEMM kernels instead of relying solely on cuBLAS.

For readers new to GPU profiling, the art of performance analysis lies in understanding where the “real bottleneck” is, that means the importance of going beyond peak specs and digging into real execution patterns, not just at the headline FLOPS and HBM bandwidth numbers.

## Complete Profile Data
Following commands used to generate the Gemm performance charts:
./sample_cublasLt_LtSgemm [M] [N] [K] 0

Following command is used to generate the NSight profile for a GEMM:
sudo ncu --set full -o [profile-name] ./sample_cublasLt_LtSgemm [M] [N] [K] 1

## Resources to explore
[Fast CUDA SGEMM from Scratch](https://siboehm.com/articles/22/CUDA-MMM)
[cuBLAS](https://developer.nvidia.com/cublas)
[CUTLASS](https://docs.nvidia.com/cutlass/index.html)
[openAI-GEMM](https://github.com/openai/openai-gemm)
[DeepSeek-GEMM](https://github.com/deepseek-ai/DeepGEMM)
