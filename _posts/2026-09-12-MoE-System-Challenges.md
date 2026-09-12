---
layout: post
title: "Mixture of Experts (MoE): A Blessing for Models, a Challenge for Systems"
date: 2026-09-12
categories: jekyll update
---

Mixture of Experts (MoE) has emerged as one of the most effective ways to scale modern LLMs without increasing computation in proportion to model size. Instead of activating the entire parameter set for every token, an MoE model routes each token to only a small subset of specialized experts. This allows the model to grow to trillions of parameters while keeping the amount of computation performed per token relatively modest. From a model-design perspective, this is an extremely attractive proposition: significantly more model capacity without paying the full computational cost of a similarly sized dense model.

What makes MoE a boon for model designers, however, is precisely what turns it into a headache for system architects. This article focuses on that paradox from the perspective of a hardware and systems architect. Rather than discussing how MoE models are designed or proposing particular optimizations, the goal is to examine where the systems challenges come from. We will look at MoE inference serving through the three fundamental system dimensions - compute, memory, and communication and examine why sparse scaling breaks many of the assumptions that make dense Transformer execution efficient.

# Key MoE shapes and notations
* T – Tokens per batch
* E – Number of experts
* k – top-k
* f – static capacity factor (padding or dropping tokens per expert)

* D - Transformer hidden dimension
* n - Expert FFN dimension

The standard MoE computation for an expert ‘e’ with SwiGLU activation can be broken down into the following components:
* ***Input:*** Xe[Te x D]
* ***UpProjection He:*** Xe[Te x D] x W1e[D x 2n] -> [Te x 2n] -> [Ge, Ue]
* ***Activation Ae:*** SwiGLU(He) -> SiLU(Ge) x Ue -> [Te x n]
* ***DownProjection Ye:*** Ae[Te x n] x W2e[n x D] -> [Te x D]

* **Expert Capacity (C)** — Maximum number of token assignments an expert is provisioned to process in a batch
    * C = k * f * (T/E)
* **Expert Granularity (G)** — Measures how small an individual expert is relative to the Transformer hidden dimension
    * G = D / n
    * Higher means smaller, finer-grained experts
* **MoE Sparsity (s)** — Fraction of the available experts activated for each token
    * s = E / K
    * Higher s means a sparser MoE

# System-Level Challenges of MoE

## Challenge 1 – Sparse and Irregular Computation

MoE reduces computation by activating only a subset of experts for each token. However, this also turns the regular, predictable computation of a dense Transformer into a dynamic and uneven workload. In other words, MoE performs fewer FLOPs, but makes those FLOPs harder for the system to execute efficiently.

* **Dynamic workload:** The number of tokens routed to each expert can vary across layers and training iterations. As a result, the amount of work assigned to an expert is not fixed; studies have observed workload variations of up to 4.38x within a single training run.
* **Small and irregular GEMMs:** GPUs achieve their highest efficiency on large, regular GEMMs. In MoE, different experts may receive very different numbers of tokens, resulting in small and uneven GEMMs that are harder to map efficiently onto the hardware.
* **Fewer FLOPs do not necessarily mean higher hardware utilization:** Expert computation requires tokens to be gathered from different positions before the GEMM and the results to be scattered back afterward. These dynamic memory accesses introduce additional data movement that does not exist to the same extent in regular dense computation.
* **Increasing granularity and sparsity make the problem harder:** Modern MoEs are moving toward more fine-grained experts, where each expert has a smaller intermediate dimension, and greater sparsity, where the total number of experts increases while the number activated per token remains relatively small. Both trends reduce the amount of useful computation performed by an individual expert and make efficient GPU execution increasingly difficult.
* **Changing execution requirements:** Because the workload varies across experts, layers, and iterations, the system must continually deal with two fundamental questions:
    * What is the most efficient parallelization strategy for the current expert workload?
    * How can high GEMM efficiency be maintained despite small and uneven expert batch sizes?
 
| Model | Release Date | Parameters | Expert Sparsity (E/K) | Expert Granularity (D/n) |
|---|---:|---:|---:|---:|
| Mixtral 8x22B | 11/23 | 131B | 8/2 = 4.0 | 6144/16384 = 0.38 |
| DBRX | 03/24 | 132B | 16/4 = 4.0 | 6144/10752 = 0.57 |
| Phi-3.5-MoE | 09/24 | 42B | 16/2 = 8.0 | 4096/6400 = 0.64 |
| OLMoE | 09/24 | 7B | 64/8 = 8.0 | 2048/1024 = 2.00 |
| Granite 3.1-MoE | 12/24 | 3B | 40/8 = 5.0 | 1536/512 = 3.00 |
| DeepSeek-V3 | 12/24 | 671B | 256/8 = 32.0 | 7168/2048 = 3.50 |
| Qwen3 MoE | 04/25 | 235B | 128/8 = 16.0 | 4096/1536 = 2.67 |
| Qwen3-30B-A3B | 05/25 | 30.5B | 128/8 = 16.0 | 2048/768 = 2.67 |
| Kimi K2 | 07/25 | 1.04T | 384/8 = 48.0 | 7168/2048 = 3.50 |
| gpt-oss-120b | 08/25 | 120B | 128/4 = 32.0 | 2880/2880 = 1.00 |
| GLM-4.5-Air | 08/25 | 106B | 128/8 = 16.0 | 4096/1408 = 2.91 |
| Qwen3-Next-80B-A3B-Instruct | 09/25 | 81B | 512/10 = 51.2 | 2048/512 = 4.00 |
| DeepSeek-V3.2-Exp | 10/25 | 685B | 256/8 = 32.0 | 7168/2048 = 3.50 |

## Challenge 2 – Memory Pressure and Bandwidth-Bound Execution

As MoEs become more fine-grained and sparse, the amount of useful computation performed by each expert decreases while memory and data-movement overheads do not shrink proportionally. This increasingly pushes MoE execution toward memory-capacity and memory-bandwidth limits.

* **Larger activation footprint:** In fine-grained MoEs, more experts are typically activated per token, and activation storage often grows roughly with the top-, increasing training memory requirements.
* **Lower arithmetic intensity:** Smaller experts perform less computation for the amount of weights and activation data that must be moved, reducing FLOPs per byte and making execution increasingly bandwidth-bound.
    * AI ~ 1 / (G + s)
* **Tile-level inefficiency:** Highly sparse MoEs often leave only a small number of tokens per expert. Grouped GEMMs must still execute at hardware tile granularity, causing partially utilized tiles and wasted computation.

## Challenge 3 – All-to-All Communication Bottlenecks

Expert parallelism requires tokens to be exchanged across GPUs before expert computation and returned afterward. This makes All-to-All (A2A) communication a major part of MoE execution time; reported measurements show it consuming about 34.1% of a training step on average. Although A2A is a bottleneck in both training and inference, the underlying causes are different.

* **Low GPU utilization during communication:** A2A is largely a data-movement phase, leaving much of the GPU compute capability idle. Reported average SM efficiency during A2A is only 3.7%.
* **Communication grows with the number of activated experts:** Increasing Top-(K) sends each token to more experts and therefore increases the amount of data transferred. In reported experiments, A2A time increased from 33.4% to 44.5% of the step time as the communication volume increased.
    * **Training** – contention with gradient communication: During backward propagation, expert-parallel A2A can execute concurrently with data-parallel AllReduce operations. Because these independent communication streams share the same network resources, background AllReduce traffic can reduce the bandwidth available to the blocking A2A and directly increase training time.
    * **Inference** – skewed expert popularity: During inference, routing is determined by the input workload and can be highly uneven. Some experts receive significantly more tokens than others, causing the GPUs hosting popular experts to experience both heavier communication traffic and more computation. These GPUs become stragglers, increasing the latency of the entire MoE layer.

| # Experts / GPUs | Model (#Layers & Params) | Training All-to-All (ms) | Training Ratio | Inference All-to-All (ms) | Inference Ratio |
|---:|---|---:|---:|---:|---:|
| 4 | 12L + 117M | 259 | 36.7% | 73 | 27.4% |
| 4 | 24L + 233M | 589 | 35.4% | 103 | 26.2% |
| 4 | 36L + 349M | 979 | 38.2% | 153 | 28.3% |
| 16 | 12L + 419M | 333 | 39.5% | 102 | 32.5% |
| 16 | 24L + 838M | 715 | 37.6% | 177 | 31.7% |
| 16 | 36L + 1.2B | 1145 | 36.8% | 243 | 27.4% |

## Challenge 4 – A2A Dependencies and Limited Communication–Computation Overlap

MoE introduces direct dependencies between All-to-All communication and expert computation: tokens must reach their destination experts before they can be processed, and expert outputs must be communicated back before subsequent computation can proceed. This makes hiding A2A latency fundamentally difficult.

* **Limited opportunities for overlap:** Some computation can proceed concurrently with A2A—for example, suitably partitioned non-MoE computation in the forward pass and independent weight-gradient computation during backward—but much of the MoE execution remains directly dependent on communication.
* **Partitioning hurts compute efficiency:** Increasing overlap generally requires expert computation to be divided into smaller chunks. As these chunks become smaller, GEMMs become less efficient and may underutilize GPU compute resources.
* **Unavoidable communication bubbles:** Coarse-grained pipelining still leaves exposed communication at the beginning, while data is first being received, and at the end, while the final results are being sent. These regions cannot be hidden by expert computation.
* **Communication and computation operate at different granularities:** Communication becomes useful as routed tokens arrive, while GPUs execute expert GEMMs in tiles. This mismatch makes it difficult to start computation immediately as data becomes available without sacrificing GEMM efficiency.
* **Dynamic workloads complicate resource sharing:** The number of tokens assigned to each expert varies at runtime. Consequently, the optimal balance of GPU resources devoted to communication and computation also changes dynamically.
* **Cluster topology can introduce redundant traffic:** A token routed to multiple experts located on different GPUs of the same remote node may be transmitted multiple times across the slower inter-node fabric. Thus, logical expert-level communication can translate into unnecessary physical network traffic.

## Challenge 5 – Large-Scale MoE Inference Serving

MoE inference combines two very different workloads—attention and expert FFNs—whose optimal hardware and scaling requirements often conflict.

* **Conflicting resource requirements:** During decode, attention is largely memory-intensive because every new token must access its request-specific KV cache. In contrast, MoE FFNs require sufficiently large token batches to achieve good weight reuse and GPU utilization.
* **Batch size is constrained by attention:** The maximum serving batch is often limited by KV-cache capacity and latency SLAs. This can leave too few tokens per expert for efficient FFN execution, especially as the number of experts increases.
* **Coupled scaling leads to poor utilization:** When attention and FFN share the same serving instance, they must scale together even though their compute, memory, and bandwidth requirements differ. A configuration that is efficient for one may be inefficient for the other.
* **Disaggregation introduces a new communication problem:** Separating attention and FFN resources allows them to scale independently and potentially use different hardware, but hidden states must then move between the two pools at every MoE layer.
* **Communication must be hidden to avoid idle resources:** Because attention and FFN remain sequentially dependent, either side can sit idle while waiting for data. Efficient serving therefore requires enough concurrent work to cover these communication and dependency gaps.
* **Independent scaling creates irregular M-to-N communication:** Different numbers of attention and FFN workers naturally produce  and  traffic patterns, which are more complex than conventional fixed-size collective communication.

As a system architect, these challenges are part of what makes ML so exciting. The rapid evolution keeps throwing new challenges for system design, with plenty of interesting problems for everyone working across the stack.
