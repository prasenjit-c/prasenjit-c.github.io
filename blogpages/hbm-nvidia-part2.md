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

On the HBM device side, as we’ve discussed, there are 1024 signals spread across 8 channels for HBM2 or 16 channels for HBM3. These signals are routed through the silicon interposer and connected to a SERDES (Serializer/Deserializer), depicted as a PHY in the figure, located on the GPU die. The PHY then interfaces with individual memory controllers, each managing a channel. From the software perspective, these memory channels are accessed similarly to traditional memory systems. However, achieving optimal memory performance in systems with multiple HBM devices, each containing numerous channels is a complex undertaking. It necessitates significant empirical analysis and experimentation, an aspect we will explore in more detail later. For now, with this foundational understanding, we will proceed to examine the specific implementation and utilization of HBM in NVIDIA's A100 and H100 GPUs.
