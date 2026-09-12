---
layout: post
title: "Mixture of Experts (MoE): A Blessing for Models, a Challenge for Systems"
date: 2026-09-12
categories: jekyll update
---

Mixture of Experts (MoE) has emerged as one of the most effective ways to scale modern LLMs without increasing computation in proportion to model size. Instead of activating the entire parameter set for every token, an MoE model routes each token to only a small subset of specialized experts. This allows the model to grow to trillions of parameters while keeping the amount of computation performed per token relatively modest. From a model-design perspective, this is an extremely attractive proposition: significantly more model capacity without paying the full computational cost of a similarly sized dense model.

What makes MoE a boon for model designers, however, is precisely what turns it into a headache for system architects. This article focuses on that paradox from the perspective of a hardware and systems architect. Rather than discussing how MoE models are designed or proposing particular optimizations, the goal is to examine where the systems challenges come from. We will look at MoE inference serving through the three fundamental system dimensions - compute, memory, and communication and examine why sparse scaling breaks many of the assumptions that make dense Transformer execution efficient.

# Key MoE shapes and notations
