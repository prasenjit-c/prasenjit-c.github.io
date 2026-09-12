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
* n - Expern FFN dimension

The standard MoE computation for an expert ‘e’ with SwiGLU activation can be broken down into the following components:
* Input: Xe[Te x D]
* UpProjection He: Xe[Te x D] x W1e[D x 2n] -> [Te x 2n] -> [Ge | Ue]
* Activation Ae: SwiGLU(He) -> SiLU(Ge) x Ue -> [Te x n]
* DownProjection Ye: Ae[Te x n] x W2e[n x D] -> [Te x D]

* Expert Capacity (C) — Maximum number of token assignments an expert is provisioned to process in a batch
* C = k * f * (T/E)
* Expert Granularity (G) — Measures how small an individual expert is relative to the Transformer hidden dimension
* G = D / n
* Higher means smaller, finer-grained experts
* MoE Sparsity (s) — Fraction of the available experts activated for each token
* s = E / K
* Higher s means a sparser MoE
