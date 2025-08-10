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
* \[ \text{AMI} = \frac{\text{Total FLOPs}}{\text{Total Memory Transfers}} \]
* Asymptotically, \[ \text{AMI} \approx \frac{2N^3}{3N^2} \approx O(N) \]

This asymptotic analysis reveals that as the size of the matrices increases, GEMM becomes more compute-bound. Consequently, memory bandwidth is typically not a major performance bottleneck for large matrices. However, for smaller matrices, memory access latency and bandwidth play a significant role in overall performance, making it challenging to fully utilize the GPU's compute capabilities.

With the roofline model established, we can now use it as a reference to evaluate how GEMM actually performs on NVIDIA GPUs.
