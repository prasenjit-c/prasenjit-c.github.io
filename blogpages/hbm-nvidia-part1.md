** **
## Part 1 - Fundamentals of NVIDIA Memory Hierarchy and NSight Compute

The first part of this series is primarily targeted at beginners, covering the basics of NVIDIA GPU architecture and introducing the NVIDIA Nsight Profiler. If you’re already familiar with NVIDIA’s memory hierarchy and the usage of the Nsight Profiler, feel free to skim the performance statistics presented at the end of this part or skip directly to the sections that pique your interest.

However, if you’re new to these concepts or want to refresh your understanding of CUDA, Nsight, and NVIDIA GPU fundamentals, this is an excellent starting point. We'll take a practical approach, using a simple yet representative program. By profiling it under various configurations, we'll unravel the intricacies of the GPU memory hierarchy step by step.

### A Sample CUDA FMA Program

You can find the source code for this program in my [Github Repository](https://github.com/prasenjit-c/cuda-tests/blob/2ea6a92573dea9a803b92e6111a20e630116ea13/ami_measure.cu)

The CUDA kernel performs an element-wise Fused Multiply-Accumulate (FMA) operation.

#### Kernel Functionality and Memory Access:

For each element `i`, it performs `OP` FMA operations, utilizing distinct scalar values from the `user_arg[o]` array. Essentially, it accumulates `OP` products of `x[i] * user_arg[o]` into `y[i]`, effectively computing:
y[i] = y[i] + x[i] * user_arg[0] + x[i] * user_arg[1] + ... + x[i] * user_arg[OP-1]
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
```

The above command launches a profile with a complete set of events and counters. When starting with profiling, collecting the complete set of performance counters can be a good strategy, as it ensures comprehensive data capture and prevents accidentally overlooking important metrics. However, if you are more experienced or have a specific focus, you can use a targeted set of counters by leveraging the various options provided.Once the profile is collected and loaded into the GUI, you'll typically focus on key sections like __Computation Analysis__ and __Memory Analysis__. This particular blog post concentrates on the memory hierarchy; therefore, our discussion will center on the findings within the **Memory Analysis** section, setting aside detailed compute optimization strategies for now."

I have listed every command in detail via the link provided below, enabling you to recreate this analysis in your own environment and see the results for yourselves. To illustrate, we start by profiling a highly simplified version of the program on A100 - **configured using a single block, a single thread, 1 MB of memory, and a single FMA instruction (OP = 1)** — produces a profile that, when loaded into the GUI, looks something like this:
![](/images/hbm-part1-image1.png "Profile Summary")

While fields like __Estimated Speedup__ and __Runtime Improvement__ might not be the immediate focus, the other details provide a useful sanity check on the program's execution. Moving to the __Details__ tab, the **GPU Speed of Light Throughput** section quantifies the utilization of the GPU's compute and memory subsystems. As expected, given our rudimentary test program, you'll notice that the compute and memory utilization figures reported here are negligible.
![](/images/hbm-part1-image2.png "SOL")

The __Compute Workload Analysis__ and __Instruction Statistics__ sections provide additional details about the behavior of threads and warps during execution. To gain deeper insights into memory behavior, we can explore the **Memory Workload Analysis** section, which helps us understand the kernel’s memory access patterns. This section begins with numerical data and includes a clear pictorial representation of the GPU's logical memory hierarchy. There is sufficient information to grasp the concepts, so let's proceed step-by-step to analyze it.
![](/images/hbm-part1-image3.png "Memory Hierarchy")

Continuing with the profile collected from our trivial kernel launch, we observe that the kernel was launched with grid and block dimensions set to one for all three axes (X, Y, and Z). This confirms our configuration, indicating that only a single thread is running on the entire GPU.
Next, let's analyze the instruction statistics. The profiler reports approximately 2.3 million total executed instructions. Diving deeper into the instruction mix, the executed fused multiply-add (FMA) instructions, represented as FFMA, are of particular interest. The report indicates that 131,072 FFMA instructions were executed.

#### Verifying FMA Instruction Count

We can verify this FMA count against our kernel's intended operation and input data size. Our kernel processes two arrays, X and Y, each allocated with 0.5 MB of memory. Assuming each element is a 4-byte single-precision floating-point number (float), we can calculate the number of elements in one array:

* Array Size = 0.5 MB = 512 KB = 512 * 1024 bytes = 524,288 bytes
* Size of Element (float) = 4 bytes
* Number of Elements = Array Size / Size of Element = 524,288 bytes / 4 bytes = 131,072 elements
* Expected FMA Operations = Number of Elements = 131,072

This calculated value precisely matches the 131,072 FMA instructions reported by Nsight Compute's Instruction Statistics section, validating our understanding of the kernel's execution behavior and the profiler's output.

#### Verifying the Memory Statistics
Now, let's turn our focus to the __Memory Analysis__ section of the profile. Specifically, examine the table of cache accesses. This table includes columns such as *Instructions, Requests, Referenced Sectors, Hit Rate*, and more.
![](/images/hbm-part1-image4.png "L1 Stats")
In this particular case, since there is only a single thread executing the kernel, the values for Instructions, Requests, and Referenced Sectors are all identical. In later kernel analyses, we will explore scenarios where these values can differ
For now, we observe that the L1 cache accesses resulting from global memory operations as reported by the profiler:
* Global Load Requests: The profiler indicates 393,216 requests targeting the L1 cache originated from global memory load instructions.
* Global Store Requests: Similarly, 131,072 requests targeting the L1 cache originated from global memory store instructions.
Recall from our instruction analysis that the kernel processed 131,072 elements. The observed memory request counts suggest a specific access pattern:
* The 131,072 store requests correspond directly to one store operation per element processed.
* The 393,216 load requests (which is exactly 3 * 131,072) imply that the kernel performs three distinct load operations from global memory for each element it processes.
This quantitative match between instruction counts, element counts, and memory requests reinforces our understanding of the kernel's behavior: for each of the 131,072 iterations (one per element), the code likely reads three values from global memory and writes one value back, with these operations primarily interacting with the L1 cache initially.

#### Calculating the L1 Cache Hit Rate
To accurately evaluate the effectiveness of the L1 cache, particularly its hit rate, we need to consider a fundamental characteristic of many NVIDIA GPU architectures: the L1 data cache typically employs a write-through policy.

A write-through cache policy means that whenever a store operation writes data to the L1 cache, that data is simultaneously (or very shortly thereafter) written to the next level of the memory hierarchy, which is usually the L2 cache.
This policy can sometimes lead to confusing interpretations of profiler metrics related to store operations:
* L1 Store Hits: A profiler might report a very high (even 100%) L1 hit rate for global store operations. This often signifies that the L1 cache successfully accepted the write request.
* L2 Traffic from Stores: Concurrently, because the write must propagate to L2 due to the write-through policy, these same store operations will generate traffic (transactions) between L1 and L2. The profiler might report these L1-to-L2 transactions in a way that appears as "L2 Misses" or increased L2 traffic originating from L1 stores.

It might seem contradictory for a store to be an L1 "hit" while also causing an L1-to-L2 transaction (sometimes logged confusingly). This stems directly from the write-through mechanism: the L1 accepts the write (hit) but does not retain sole ownership; the data must proceed to L2.
To accurately compute the L1 cache hit rate, we use the following calculation:
* Total L1 requests: 393,216 + 131,072 = 524,288
* Sector Misses to L2: 32,769

Thus, the L1 cache hit rate is: 1 - 32,769/524,288 = 93.75%<br>
This calculation aligns with the data and reflects the efficiency of L1 cache utilization for this kernel.

Moving further down the memory hierarchy, we examine the L2 cache statistics to understand its role in servicing memory requests that missed the L1 cache. Key columns to examine in this analysis are Requests, Sectors and Sectors/Request. In NVIDIA GPUs, each cache line is 128 bytes, and these are divided into 4 sectors, each consisting of 32 bytes. Requests between L1 cache, L2 cache and device memory operate at the granularity of a sector. From the profiler, we observe that the total sector misses to device memory amount to 32,770. This indicates that every request reaching L2 and missing there results in an access to device memory.

![](/images/hbm-part1-image5.png "L2 Stats")

The total data transferred between L2 and device memory is calculated as:<br>
32,769 sectors * 32 B/sector=1.05 MB

![](/images/hbm-part1-image6.png "HBM Stats")

This value closely aligns with the working memory size used in this kernel. You can verify this value further by examining the logical memory diagram displayed in the profiler, which provides a detailed visualization of memory accesses and transfers.

### Digging Deeper in the Memory Hierarchy
Having established a baseline understanding with a single thread, let's explore how the execution characteristics change as we increase parallelism within a single thread block. Using the same program, we will allocate 8 MB of memory and increase the number of threads from 1 to 8, 32, and 64 threads, while keeping the number of thread blocks fixed at one. Although these thread counts are still relatively small compared to the GPU's full capacity, this controlled experiment allows us to observe fundamental effects of intra-block parallelism, particularly on metrics related to instruction execution and potentially cache hierarchy interactions.
The table below presents relevant statistics collected from these four runs, similar to those discussed previously. Two metrics related to Fused Multiply-Add (FMA) operations warrant special attention:
* FFMA Executed represents the same statistic we analyzed earlier, measured on a per warp basis by the profiler
* Total FFMA: This represents the aggregate number of FMA operations performed across all threads in the thread block for the entire problem and is derived from FFMA Executed using the formula

`Total FFMA = FMA_Executed (per warp) * MIN(Number_of_Threads, 32)`

For a single thread, Total FFMA and FFMA Executed are identical since a warp consists of only one thread. However, as the number of threads in a warp increases, the FFMA Executed is multiplied by the number of threads (up to a maximum of 32, which is the warp size on recent NVIDIA GPUs).
Observing the results in the table, you'll notice that the calculated Total FMA remains constant across all four runs (1, 8, 32, and 64 threads). This is expected behavior. The total number of FMA operations required is determined by the size of our input arrays (8 MB) and the logic of the kernel (e.g., one FMA per element). This workload doesn't change; we are merely altering how many threads collaborate to perform it.
Increasing the number of threads primarily impacts the overall execution time, which should generally decrease as more threads share the work.

|Num Threads|FFMA Executed|Total FFMA|L1 Loads|L1 Stores|L2 Load|L2 Store|HBM Load|HBM Store|Time (ns)|Speedup|
|-----------|-------------|----------|--------|---------|-------|--------|--------|------------|---------|-------|
|1|1,048,576|1,048,576|3,145,728|1,048,576|262,145|1,048,576|262,240|0|125,427,328|1|
|8|131,072|1,048,576|393,216|131,072|262,145|131,072|262,240|0|50,073,984|2.5|
|32|32,768|1,048,576|98,304|32,768|65,537|32,768|264,536|0|17,901,664|7.0|
|64|32,768|1,048,576|98,304|32,768|65,537|32,768|262,452|0|8,897,376|14.1|

#### Analyzing L1 Cache Behavior with Increased Parallelism
Examining the L1 metrics, we observe an interesting trend: as the number of threads increases, the number of L1 loads and stores decreases. Moreover, this reduction is proportional to the thread factor. For instance:
* In the 8-thread run, there are 8 times fewer loads and stores compared to the single-thread run.
* Similarly, in the 32- and 64-thread runs, there are 32 times fewer loads and stores.

This reduction highlights the cooperative behavior of threads, where memory is accessed collectively by groups of threads.This phenomenon is technically referred to as global memory coalescing, an essential concept for optimizing performance on NVIDIA GPUs.
Another noteworthy observation is that if we divide the number of L1 loads by the FFMA Executed, the result is a factor of 3. This indicates that, at the warp level, each FMA operation requires three loads, aligning with the expected behavior of our program.
Before diving deeper, let’s expand our understanding of L1 cache lines and sectors:
* An L1 cache line in NVIDIA GPUs is typically 128 bytes in size.
* Each cache line is divided into four sectors, with each sector being 32 bytes.
* Memory transactions between L1 and L2, or between L2 and device memory, occur at the sector granularity.

Armed with this knowledge, let’s attempt to calculate the L1 loads and stores manually and verify if they match the numbers reported by the NVIDIA profiler.
To better visualize how threads access data and how coalescing occurs, consider the layout of elements from arrays X and Y relative to the cache structure.
![](/images/hbm-part1-image7.png "Strided Access")

Recall that our data elements are 4-byte single-precision floats, 8 elements fit in a sector, and hence 32 elements fit in a cache line. Let's analyze the memory operations required to process 32 consecutive elements, which corresponds to 32 FMA operations assuming one FMA per element. This workload requires loading data equivalent to one cache line of X, loading one cache line of Y, and storing one cache line of Y. When we increase the number of threads in a warp, the memory access efficiency improves as a single load at the warp level can serve all threads in the warp.<br>

**8 threads per warp serving 32 FFMA:**
* We require 8 loads, one for each sector, multiplied by 2 for X and Y.
* Additionally, we need 4 stores for Y, corresponding to each sector <br>

**32 threads per warp serving 32 FFMA:**
* A single load serves all 32 threads, covering the entire cache line.
* Only 2 loads (one each for X and Y) and 1 store (for Y) are needed.

We can now scale this analysis to our entire 8 MB dataset comprising 1,048,576 FFMAs.<br>

1 thread per warp:
* L1 Loads = 2 x 1,048,576 = 2,097,152
* L1 Stores = 1 x 1,048,576 = 1,048576<br>

8 threads per warp:
* L1 Loads = 8 x (1,048,576/32) = 262,144
* L1 Stores = 4 x (1,048,576/32) = 131,072<br>

32 threads per warp:
* L1 Loads = 2 x (1,048,576/32) = 65,536
* L1 Stores = 1 x (1,048,576/32) = 32,768

As you can see, while the stores match perfectly, the loads are slightly off. This discrepancy arises because we overlooked the contribution of the user_arg array. The user_arg array (or variable) exhibits a different access pattern: the same value(s) from it are required by all threads executing within a given warp. When threads within a warp access the exact same memory address, the load operation is performed only once per warp and then efficiently broadcast to all participating threads within that warp. Taking this into account, here is the updated calculation that includes the access to user_arg.<br>

1 thread per warp:
* L1 Loads = 2,097,152 + 1,048,576<br>

8 threads per warp:
* L1 Loads = 262,144 + 4 x (1,048,576/32) = 393,216<br>

32 threads per warp:
* L1 Loads = 65,536 + 1 x (1,048,576/32) = 98,304<br>

Beyond this, deriving the L2 loads, stores, and HBM (High Bandwidth Memory) loads becomes straightforward. As mentioned earlier:<br>
**L1-L2 Transactions:** Occur at a minimum granularity of a 32-byte sector. Fully coalesced accesses (like those from a full warp hitting or missing L1 for a 128-byte cache line) can potentially utilize wider 128-byte cache line transactions.<br>
**L2-HBM Transactions:** Communication between the L2 cache and HBM on this architecture (e.g., NVIDIA A100) consistently uses a 32-byte sector granularity, considered optimal for HBM device efficiency.<br>
Using this knowledge, you can calculate the L2 loads, stores, and HBM loads similarly. Comparing these values with the table above should confirm that the numbers align with expectations.

One key observation in the profiler results for this 8 MB dataset is the complete absence of HBM store traffic. This is expected behavior for this scenario due to two main factors:
1. Dataset Size vs. L2 Capacity: Our total dataset size (8 MB) is significantly smaller than the large L2 cache capacity of the NVIDIA A100 GPU (40 MB).
2. L2 Cache Policy: The A100's L2 cache employs a write-back policy. 

Finally, having analyzed the transaction counts and data movement through the cache hierarchy, it's logical to evaluate the efficiency of each cache level by examining their respective hit rates. We can now calculate the L1 and L2 hit rates based on the transaction data discussed. The data reported by Nsight can best be explained by using Sector accesses as:<br>
`L1 Hit Rate = Sectors(L1 Loads + L1 Stores - L2 Loads)/Sectors(L1 Loads + L1 Stores)`<br>
`L2 Hit Rate = (L2 Loads + L2 Stores - HBM Loads/4)/(L2 Loads + L2 Stores)`<br>

This results in the following calculated and profiled values.<br>
|Num Threads|L1 Hit Rate (NSight:Calculated)|L2 Hit Rate (NSight:Calculated)|
|-----------|-------------------------------|-------------------------------|
|1|93.8 : 93.8|86.9 : 95|
|8|50 : 50|58.2 : 83.3|
|32|38.5 : 50|35.7 : 32.7|

Evaluating the calculated cache hit rates against the profiler's reported values reveals:
* The L1 cache hit rate derived from our model aligns very closely with the profiled result, confirming our understanding of L1 behavior, including coalescing and the impact of uniform access patterns (user_arg).
* The calculated L2 cache hit rate shows a minor deviation from the profiled value.<br>

This slight discrepancy typically stems from complexities not included in our simplified model, primarily the internal traffic within the L2 cache subsystem. Modern GPUs feature large, partitioned L2 caches, and data movement between these partitions (or slices) via an internal interconnect or fabric contributes to overall L2 activity and latency. Modeling this internal L2 traffic is intricate and requires analyzing specialized profiler metrics (e.g., related to the L2 Transaction System or crossbar), which is beyond the scope of this introductory analysis.

For now, the numbers derived here are highly relevant to the primary focus of this blog: analyzing HBM memory accesses.There's one more aspect we haven’t explored yet: what happens when we increase the grid size or the number of thread blocks running across different SMs of the GPU. In this scenario, there’s no fundamentally new behavior. As the dataset size increases, more threads and resources are employed to process it. The table below outlines the data collected for this case. Additionally, this includes the profile with increasing FFMA per thread as well.
I encourage you to examine this data. Apply the concepts discussed throughout this analysis – coalescing efficiency based on thread counts, transaction granularities (sectors, cache lines), L1/L2 cache policies, and expected load/store ratios per FMA – to verify whether the observed metrics remain consistent with these principles on a per-block or per-warp basis, even amidst the system-level effects of multi-block execution.<br>

### Complete Profile Data
#### NVIDIA A100 SXM4 40GB
__HOST OS: Linux Ubuntu 22.04__<br>
Following commands used to generate the complete profile:<br>
`sudo ncu --set full -o ami_profile_1_1_1 ./ami_measure 1024 1 1 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_1_1_8 ./ami_measure 1024 1 1 8 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_1_8_1 ./ami_measure.exe 1024 1 8 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_1_1L_1 ./ami_measure 8192 1 1 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_1_8L_1 ./ami_measure 8192 1 8 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_1_32_1 ./ami_measure 8192 1 32 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_1_64_1 ./ami_measure 8192 1 64 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_1_32_8 ./ami_measure 8192 1 32 8 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_64_8_1 ./ami_measure.exe 1048576 64 8 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_64_32_1 ./ami_measure 1048576 64 32 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_64_32_2 ./ami_measure 1048576 64 32 2 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_64_32_4 ./ami_measure 1048576 64 32 4 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_64_32_8 ./ami_measure 1048576 64 32 8 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_4096_32_1 ./ami_measure 4194304 4096 32 1 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_4096_32_2 ./ami_measure 4194304 4096 32 2 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_4096_32_4 ./ami_measure 4194304 4096 32 4 1 2 3 4 5 6 7 8<br>
sudo ncu --set full -o ami_profile_4096_32_8 ./ami_measure 4194304 4096 32 8 1 2 3 4 5 6 7 8`

![](/images/hbm-part1-image8.png "Full-Profile-Data-A100")

Complete Data is avilable as a CSV file for easier analysis.

[Full-Profile-Data-A100](/blogpages/hbm-part1-full-data.csv)
