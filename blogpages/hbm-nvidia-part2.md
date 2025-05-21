** **
## Part 2 - Exploring HBM in NVIDIA GPUs

In this part of the series, we will delve into the intricacies of HBM access in NVIDIA GPUs. We will begin by establishing a foundational understanding of HBM, covering its architecture, physical characteristics, and key attributes. From there, we will examine different memory access mechanisms, beginning with straightforward approaches and progressing to more optimized methods for accessing HBM from CUDA programs. Along the way, we’ll discuss the pros and cons of each approach. Furthermore, we will analyze specific NVIDIA NSight profiles to identify potential performance bottlenecks and examine HBM bandwidth utilization through illustrative plots. Our primary focus will be on the NVIDIA A100 and H100 GPUs, which utilize HBM2 and HBM3 respectively, allowing for a comparative analysis of their memory access characteristics.

### HBM Architecture Overview

As the name implies, HBM (High Bandwidth Memory) is a specialized type of Dynamic RAM (DRAM) designed and optimized for applications that demand exceptionally high memory bandwidth. While the fundamental operational principles of HBM are akin to those of standard Double Data Rate (DDR) DRAM devices – sharing core concepts with technologies like DDR4 or DDR5 – a key distinction lies in its architecture. However, traditional DDR DRAM is constrained by the physical limitations of I/O signals, resulting in relatively narrow I/O widths. In contrast, HBM significantly expands the I/O width by employing a much larger number of I/O bits. This is made possible because HBM is typically integrated into the same package as the compute SoC (System on Chip)—whether it’s a GPU, CPU, or another type of accelerator. This integration allows HBM to overcome the bandwidth limitations of traditional DRAM architectures, making it a powerful choice for high-performance computing applications.

To analyze HBM effectively, it’s crucial to understand some architectural details. HBM (High Bandwidth Memory) development began around 2010, with the first generation released in 2013. This was followed by HBM2 in 2015, HBM2E (an extended version) around 2020, HBM3 in 2022, and HBM3E in 2023. Across generations, HBM has consistently improved in density, capacity, and bandwidth. However, the total I/O width has remained constant at 1024 bits.

Similar to other DRAM devices, HBM operates with multiple independent interfaces known as channels. A single memory die contains a specific number of these channels. To form an HBM stack, multiple dies are vertically stacked, aggregating the total number of available channels. The illustration below depicts this organization showing HBM2 on the left and HBM3 on the right.

![](/images/HBM-Die.png "HBM Die")

The architectural differences between HBM2 and HBM3 can be seen in their channel configurations:
* In HBM2, each memory die contains 2 channels, each 128 bits wide.
* In HBM3, each memory die contains 4 channels, each 64 bits wide.

In both cases, each die maintains a total width of 256 bits. At the stack level a single HBM stack typically consists of 4 stacked dies, resulting in:
* 8 channels for HBM2.
* 16 channels for HBM3.

An HBM device is constructed using these individual stacks, as illustrated in the figure below. The figure depicts three common configurations: a single-stack, a two-stack, and a three-stack device, all of which are supported by both HBM2 and HBM3 architectures. These are often referred to as 4-High, 8-High or 12-High devices, respectively, based on the number of memory dies within each stack.

![](/images/hbm-device-stack.png "HBM Device")

It's important to note that increasing the number of stacks primarily enhances the overall memory capacity and expands the number of bank groups and banks within the device, which can improve bank-level parallelism. However, adding more stacks does not inherently increase the peak memory bandwidth of the HBM device itself.

A natural question arises: how do HBM architectures achieve generational improvements in both bandwidth and memory capacity? As you might have guessed, these advancements are driven by increases in clock speed and device density, made possible through improvements in manufacturing processes. The table below summarizes the evolution of HBM, from its inception to HBM4, detailing the key parameters that have changed with each generation to enhance the architecture. Please refer the resources provided below if you want to dig deeper in these areas.

![](/images/hbm-arch-generation.png "HBM Generations")

An essential aspect of understanding HBM is how a compute engine—whether a CPU or GPU—integrates with an HBM device within the same package. It’s equally important to grasp how software running on these compute engines perceives and utilizes HBM. The following figure illustrates a typical HBM2 integration with a GPU via a silicon interposer.

![](/images/hbm-gpu-packaging.png "HBM GPU Package")

On the HBM device side, as we’ve discussed, there are 1024 signals spread across 8 channels for HBM2 or 16 channels for HBM3. These signals are routed through the silicon interposer and connected to a SERDES (Serializer/Deserializer), depicted as a PHY in the figure, located on the GPU die. The PHY then interfaces with individual memory controllers, each managing a channel. From the software perspective, these memory channels are accessed similarly to traditional memory systems. The below figure shows an actual HBM2 in the Pascal P100 package.

![](/images/P100-HBM2-Package.png "HBM2 in P100")

However, achieving optimal memory performance in systems with multiple HBM devices, each containing numerous channels is a complex undertaking. It necessitates significant empirical analysis and experimentation, an aspect we will explore in more detail later. For now, with this foundational understanding, we will proceed to examine the specific implementation and utilization of HBM in NVIDIA's A100 and H100 GPUs.

### HBM Implementation in Ampere & Hooper GPUs

NVIDIA's recent Ampere and Hopper GPU architectures both incorporate six HBM devices in their design. However, it's noteworthy that in many configurations, only five of these six devices are actively utilized on the platform.

![](/images/Nvidia-HBM.png "HBM Device in A100 and H100")

The table below outlines the specific HBM configurations for NVIDIA’s A100 (Ampere), H100, and H200 (Hopper) GPUs. The A100 utilizes HBM2E, while the Hopper architecture features HBM3 in the H100 and the advanced HBM3E in the H200. As previously discussed, each individual HBM stack is equipped with a 1024-bit-wide bus. The total HBM capacity spans from 40 GB to 144 GB, with bandwidth ranging from 300 GB/s to an impressive 4.8 TB/s. Examining the table, you can observe how architectural advancements in HBM—such as higher device densities, increased clock speeds, and improved fabrication processes—have significantly enhanced NVIDIA’s memory capabilities. These improvements are crucial for delivering the memory bandwidth and capacity demanded by modern AI workloads, granting NVIDIA GPUs a distinct performance advantage.

![](/images/Nvidia-HBM-Parameters.png "HBM Configuration")

Given the intricate hierarchy of multiple HBM devices, each with numerous channels, and the further subdivision of channels into rows and columns, a natural question arises: how does a memory access initiated by a thread in a Streaming Multiprocessor (SM) ultimately reach its target location within HBM? In the following section, I will attempt to shed light on this aspect of the architecture.

**However, let me clarify upfront: the precise microarchitectural details, design, and mechanisms employed by NVIDIA remain proprietary and undisclosed. The explanation provided here is based on informed speculation and logical deduction, and while it is grounded in fundamental principles, the exact implementation details might differ significantly. With this in mind, consider the explanations as a plausible representation of how memory access routing may work in NVIDIA’s GPUs. While alternative schemes could exist, the broad concepts described here should align with the overarching functionality of the architecture.**

Let’s take a step back. As explored in detail in Part 1, memory accesses initiated by a thread traverse the L1 and L2 cache hierarchy before reaching the HBM. The figure below illustrates how the two portions of the L2 cache might be structured. Each portion consists of numerous physical L2 slices—individual memory blocks that store data—and together, these slices form a logically unified L2 cache. This organization is typical of modern CPU and GPU architectures with large caches. An on-chip interconnect serves as the backbone, connecting Streaming Multiprocessors (SMs), L2 slices, and memory controllers. In the simplified depiction here, a single interconnect is dedicated to each portion of the L2 cache, encompassing half of the L2 slices, SMs, and HBM controllers. However, other configurations are possible. For example, the interconnect could be divided into a top and bottom half, or there might be a separate interconnect exclusively for memory controllers, which then links to another interconnect responsible for L2 slices and SMs. The fundamental principle, however, remains consistent: a collection of SMs, memory controllers, and L2 slices are interconnected in a hierarchical fashion, with this interconnected portion repeating to form the complete GPU's memory hierarchy and architecture.

![](/images/L2-NOC.png "L2 Interconnect")

Using the figure as a reference, the journey of a memory access follows a structured path. The first step is to determine the appropriate L2 partition to which the access belongs. Once identified, the memory request is routed to the corresponding L2 partition. Within that partition, the system must decide which specific L2 slice the access maps to. After arriving at the correct slice, a tag search is performed to determine whether the access results in a hit or a miss.

In the case of a miss, the process continues by identifying the appropriate HBM device and the memory controller responsible for handling the request. Upon reaching the memory controller, the operation proceeds like any standard DRAM device. The memory controller determines the target channel, and within that channel, it identifies the bank, row, and column corresponding to the access.

On the return path, the retrieved data is written back into the L2 slice where the request originated. Finally, the data is forwarded back to the requesting SM, completing the memory access journey.

![](/images/Address-Mapping.png "Device Address Mapping")

While these steps appear straightforward in determining the final memory destination, the precise mapping mechanisms are often far more complex. Typically, all mapping steps are functions of the physical memory address bits, but they can also incorporate other parameters like SM ID or thread ID to achieve a more fine-grained, distributed mapping. These intricacies present several fascinating and open research questions regarding the optimal mechanisms to achieve overall system performance across diverse workloads.

1. Partitioning the Device Memory Address Space:<br>
   First and foremost, a critical decision lies in how the entire device memory address space should be partitioned across multiple HBM devices. Consider an example with the A100 GPU, which has 40 GB of HBM memory divided across five devices (8 GB each).
- One approach could assign the first 8 GB of the address space to the first HBM device, the next 8 GB to the second, and so forth.
- Another approach could interleave memory at a fine granularity, such as at a cache-line level (e.g., 128 bytes). In this case, the first 128 bytes could go to one HBM device, the next 128 bytes to another, cycling through all devices before looping back.<br>

   Each approach has distinct pros and cons regarding locality, load balancing, and contention. A practical solution likely lies in between, involving block-sized address ranges mapped to specific HBMs, with interleaving at a coarser granularity.

1. Mapping within an HBM Device:<br>
Extending this further, once an HBM device is selected, the exact mapping scheme for a particular channel, and subsequently, the bank and rows within that channel, becomes another crucial consideration. This includes architectural choices such as whether an open-page or closed-page policy should be adopted, each impacting access latency and throughput.

1. L2 Slice Partition Mapping:<br>
Another pivotal aspect involves deciding the optimal L2 slice and partition to which an address should be mapped. Similar to HBM mapping, various granularities for mapping address ranges to L2 slices present interesting scenarios. This decision directly influences cache hit rates and interconnect traffic.

1. Cross-Partition Data Access in L2 Cache:<br>
Finally, there's the consideration of coherence and data placement when an SM accesses an address mapped to an L2 partition it is not directly connected to. In such cases, should the data exclusively reside in that remote L2 partition, or would it be more beneficial to also copy the data to the local L2 partition? Understanding the conditions under which such copying is advantageous, especially concerning L2 coherence traffic (as briefly noted in Part 1), is vital for maximizing performance.

With this foundational background established, let's now transition to examining the real-world performance achievable from HBM.

### HBM Bandwidth Under the Lens: Threads, SMs, and Beyond

![A100 HBM BW Profile](/images/A100-HBM-BW-Chart1.png "A100 HBM2E BW")
![H200 HBM BW Profile](/images/H200-HBM-BW-Chart1.png "H200 HBM3E BW")
![Single SM HBM BW](/images/HBM-BW-1SM-Chart.png "HBM BW Single SM")
![HBM BW Grid Size](/images/HBM-BW-GS-Increase-Chart.png "HBM BW Grid Size")

### Resources to explore
#### HBM Architecture
[Wikipedia](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)
[Rambus](https://www.rambus.com/blog_category/hbm-and-gddr6/)
[Micron](https://www.micron.com/products/memory/hbm)
[SKHynix](https://product.skhynix.com/products/dram/hbm.go)
