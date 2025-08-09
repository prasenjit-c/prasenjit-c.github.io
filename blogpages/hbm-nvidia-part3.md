** **
## Part 3 - From Theory to Practice: HBM in GEMM Workloads

In the previous part of this series, we explored the architectural characteristics and performance behavior of HBM in NVIDIA GPUs. While that discussion provided a solid theoretical and empirical understanding of HBM performance, this part will focus on a practical application. Specifically, we will examine how HBM performs in one of the most prevalent and critical operations in modern machine learning: GEMM (General Matrix Multiplication).

### Fundamentals of GEMM

Given that the AI/ML community is heavily focused on optimizing GEMM, there’s little value in covering its fundamentals in detail here. If you are unfamiliar with GEMM or wish to better understand its workings, there are countless tutorials and explanations available online. As with other posts in this series, I will include a list of relevant GEMM resources that you may find useful.

For the context of our discussion, a GEMM operation performs the calculation C = αA⋅B + βC. As the figure illustrates, this operation computes each individual element of the output matrix C by taking the dot product of a corresponding row from matrix A and a column from matrix B.

![](/images/hbm-part3-gemm.png "GEMM")
