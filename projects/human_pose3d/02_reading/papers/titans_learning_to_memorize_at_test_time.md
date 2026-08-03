# Titans: Learning to Memorize at Test Time

> 2025 · id: arxiv:2501.00663 · arXiv: 2501.00663 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and
attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows
attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling
of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new
neural long-term memory module that learns to memorize historical context and helps an attention to attend to the
current context while utilizing long past information. We show that this neural memory has the advantage of a fast
parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its
limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its
ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce
a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate
memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics,
and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models.
They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks
compared to baselines.
1

## introduction
“The true art of memory is the art of attention!"
— Samuel Johnson, 1787
T
ransformers, pure attention-based architectures (Vaswani et al. 2017), have been firmly established as state-of-
the-art models in sequence modeling, mainly due to their in-context learning and ability to learn at scale (Kaplan
et al. 2020). The primary building blocks of Transformers–attention modules—function as associative memory
blocks (Bietti et al. 2024), where they learn to store key-value associations and retrieve them by computing pairwise
similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer
is exclusively conditioned on the direct dependencies of tokens in the current context window. This accurate modeling of
dependencies, however, comes with quadratic time and memory complexity in terms of the context length. In complex
real-world tasks (e.g., language modeling (N. F. Liu et al. 2024), video understanding (C.-Y. Wu et al. 2019), long-term time
series forecasting (H. Zhou et al. 2021)), the context window can become extremely large, making the applicability of
Transformers challenging in these downstream tasks.
To overcome the scalability issue of Transformers, recent studies aim to design different variants of linear Transform-
ers (Kacham, Mirrokni, and P. Zhong 2024; Katharopoulos et al. 2020; S. Yang, B. Wang, Shen, et al. 2024), where softmax is
replaced by a kernel function in the attention (see §2.1 for details), resulting in a significant drop in memory consumption.
Despite efficiency and the ability to scale to longer context, linear Transformers do not show competitive performance
compared to Transformers as the kernel trick makes the model a linear recurrent network, in which the data is compressed
into a matrix-valued states (Katharopoulos et al. 2020). This, however, brings a contradictory fact about linear recurrent (or
linear Transformers) models: On one hand, we use these linear models to enhance scalability and efficiency (linear vs.
quadratic complexity), whose advantages is appeared for very long context; On the other hand, a very long context cannot
be properly compressed in a small vector-valued or matrix-valued states (S. Wang 2024).
1
arXiv:2501.00663v1  [cs.LG]  31 Dec 2024

Furthermore, beyond efficiency, most existing architectures–ranging from Hopfield Networks (Hopfield 1982) to LSTMs (Jür-
gen Schmidhuber and Hochreiter 1997) and Transformers (Vaswani et al. 2017)–face challenges when dealing with general-
ization, length extrapolation, and/or reasoning (Anil et al. 2022; Qin, Y. Zhong, and Deng 2024), all of which are inseparable
parts of many hard real-world tasks. Although these architectures draw inspiration from the human brain, each of which
are missing: (1) a crucial component for learning process—such as short-term memory, long-term memory, meta-memory,
attending to current context, etc. (Cowan 2008); (2) how these components are interconnected systems that can operate
independently; and/or (3) the ability to actively learn from data and memorize the abstraction of past history. We argue
that in an effective learning paradigm, similar to human brain, there are distinct yet interconnected modules, each of which
is responsible for a component crucial to the learning process.
Memory Perspective
Memory is a fundamental mental process and is an inseparable component of human learning (Terry 2017). Without
a properly functioning memory system, humans and animals would be restricted to basic reflexes and stereotyped
behaviors. Accordingly, memory has been the inspiration for many seminal research in machine learning literature; e.g.,
Hopfield Networks (Hopfield 1982), LSTMs (Jürgen Schmidhuber and Hochreiter 1997), and Transformers (Vaswani et al.
2017).
Taking inspiration from the common definitions of memory and learning in neuropsychology literature (Okano, Hirano,
and Balaban 2000), most existing architectures consider memory as a neural update caused by an input, and define learning
as a process for acquiring effective and useful memory, given an objective. In this perspective, Recurrent Neural Networks
(RNNs) (Williams and Zipser 1989) can be defined as models with a vector-valued memory module M (also called hidden
state) with two main steps: Given a new input 𝑥𝑡at time 𝑡, the model (1) updates the memory using a function 𝑓(M𝑡−1,𝑥𝑡)
(with compression); and (2) retrieves the corresponding memory of input using a function 𝑔(M𝑡,𝑥𝑡) (see §2.1 for details).
Similarly, Transformers can be seen as architectures with a growing memory and two similar steps. That is, the pair of key
and value matrices acts as the model’s memory, and the model: (1) updates the memory by appending the key and value to
the memory (without compression), and (2) retrieves query vectors’ corresponding memory by finding the similarity of
query and key vectors, which is then used to weight the value vectors for the output.
This perspective, can help us better understand existing paradigms, their critical differences, and design more effective
architectures. For example, the main difference between Transformers (Vaswani et al. 2017) and linear Transform-
ers (Katharopoulos et al. 2020) is the memory structure as well as the memory updating step, in which linear Transformers
compress the historical data into a fixed-size matrix-valued memory while Transformers keep all historical data (within
the context length) without any compression. While both linear Transformers and linear RNNs (including state space
models) compress the information in memory update step, the critical difference lies in the structure of the memory,
where linear RNNs (vs. linear Transformers) use a vector-valued memory (vs. matrix-valued memory). Therefore, this
perspective motivates us to ask: (Q1) What constitute a good structure for the memory? (Q2) What is a proper memory
update mechanism? and (Q3) What is a good memory retrieval process?
Revisiting our understanding of human memory, it is neither a unitary process nor it serves a single function (Cowan
2008). In fact, memory is a confederation of systems–e.g., short-term, working, and long-term memory–each serving a
different function with different neural structures, and each capable of operating independently (Willingham 1997). This
fact motivates us to ask: (Q4) How to design an efficient architecture that incorporates different interconnected memory
modules. Finally, storing a memory is a neural process that requires to encode and store the abstraction of the past. It can
be over-simplification to assume a single vector or a matrix, whose parameters are encoding the data in a linear manner,
are enough for storing long-term history. (Q5) Is a deep memory module needed to effectively store/remember long
past?
Contributions and Roadmap
In this paper, we aim to answer the above five questions by designing a long-term neural memory module, that can
efficiently and effectively learn to memorize at test time. Building upon its design, we discuss how it can be incorporated
into an architecture.
Neural Memory (§3). We present a (deep) neural long-term memory that (as a meta in-context model) learns how to
memorize/store the data into its parameters at test time. Inspired by human long-term memory system (Mandler 2014),
2

we design this memory module so an event that violates the expectations (being surprising) is more memorable. To this
end, we measure the surprise of an input with the gradient of the neural network with respect to the input in associative
memory loss (see §3.1 for details). To better handle the limited memory, we present a decaying mechanism that consider the
proportion of memory size and the amount of data surprise, resulting in better memory management. We show that this
decay mechanism is in fact the generalization of forgetting mechanism in modern recurrent models (Dao and Gu 2024; Gu
and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2024). Interestingly, we find that this mechanism is equivalent to optimizing
a meta neural network with mini-batch gradient descent, momentum, and weight decay. Building upon tensorizing
mini-batch gradient descent to use more matmul operations (Yu Sun et al. 2024), we present a fast and parallelizable
algorithm to train our deep neural long-term memory.
Titans Architectures (§4). After designing the long-term neural memory, an important remaining question is how to
effectively and efficiently incorporate memory into a deep learning architecture. We present Titans, a family of deep models
that consists of three hyper-heads: (1) Core: this module consists of the short-term memory, and is responsible for the main
flow of processing the data (we use attention with limited window size); (2) Long-term Memory: this branch is our neural
long-term memory module that is responsible to store/remember long past; (3) Persistent Memory: this is a set of learnable
but date-independent parameters that encodes the kn

## method
S-NIAH-PK
S-NIAH-N
S-NIAH-W
2K
4K
8K
16K
2K
4K
8K
16K
2K
4K
8K
16K
TTT
98.4
98.8
98.0
88.4
60.2
36.6
10.2
4.4
78.8
28.0
4.4
0.0
Mamba2
98.6
61.4
31.0
5.4
98.4
55.8
14.2
0.0
42.2
4.2
0.0
0.0
DeltaNet
96.8
98.8
98.6
71.4
47.2
15.4
12.8
5.4
46.2
20.0
1.6
0.0
Titans (LMM)
99.8
98.4
98.2
96.2
100.0
99.8
93.4
80.2
90.4
89.4
85.8
80.6
Titans (MAC)
99.2
98.8
99.0
98.4
99.6
98.2
97.6
97.4
98.2
98.2
95.6
95.2
Titans (MAG)
99.4
98.0
97.4
97.4
99.2
98.8
97.2
98.6
98.0
98.0
90.2
88.2
Titans (MAL)
98.8
98.6
98.8
97.8
99.8
98.1
96.8
96.4
98.0
97.4
92.0
90.4
(a) Few-shot Setup
(b) Fine-Tuning Setup
Figure 6: Performance of Titans and baselines on BABILong benchmark. Titans (MAC) outperforms all baselines, including
extremely large models, e.g., GPT4.
we can see a significant drop in performance when increasing the sequence length; (3) Compared to DeltaNet, although it
is capable of removing memory using delta rule, it cannot erase the memory, lacking forgetting mechanism. Finally, As
expected we can see on par or better results when using Titans variants, where the best results correspond to MAC.
5.4
BABILong Benchmark
In the previous section we discussed the results on a simple NIAH tasks where a single needle needs to be retrieved.
Although Titans showed better performance compared to baselines, their true advantage over very long sequences is still
hidden. To this end, in this section, we use a harder task from BABILong benchmark (Yuri Kuratov et al. 2024), in which
the model needs to reason across facts distributed in extremely long documents. We follow the original experimental setup
and training process in the benchmark. There are two settings: (1) Few-shot setting, in which we use large pre-trained
models, and (2) fine-tuning setting, where we fine-tune the MAC variant of Titans to compare it with other fine-tuned
baselines. The results for few-shot setting are reported in Figure 6a. In this setup, we can see Titans outperform all
baselines–i.e., Mamba2.8B (Gu and Dao 2024), RWKV-6-7B (Peng, Goldstein, et al. 2024), RecurrentGemma-9B (Botev et al.
2024), Gemma-9B (Team et al. 2024), Llama3.1-8B (Touvron et al. 2023), GPT-4, and GPT4o-mini (Achiam et al. 2023). These
results are achieved while Titans (MAC) is having much less number of parameters than baselines.
In the fine-tuning setup, we compare the small fine-tuned version of Titans (MAC) with: (i) the fine-tuned version of small
models (almost the same number of parameters as Titans) such as Mamba (Gu and Dao 2024), RMT (Bulatov, Yury Kuratov,
and Burtsev 2022), (ii) large models with Retrieval-Augmented Generation (RAG) (P. Lewis et al. 2020) such as Llama3.1-
8B (Touvron et al. 2023), and (iii) extremely large models such as GPT-4 (Achiam et al. 2023), GPT4o-mini, Qwen2.5-72B (A.
Yang et al. 2024), and Llama3.1-70B (Touvron et al. 2023). Baseline results are reported by (Yuri Kuratov et al. 2024). The
results of Titans and baselines are reported in Figure 6b. Titans outperform all models even extremely large models like
GPT4. Also, compared to Transformer-based with memory models like RMT, Titans show better performance mainly due
to their powerful memory. That is, RMT compress the historical data into 16 size vector-valued memory, while Titans with
in-context online memory learner are capable of encoding the past into the parameters of the model. Interestingly, even
14

(a) 170M Parameters
(b) 360M Parameters
(c) 760M Parameters
Figure 7: The effect of memory depth on the perplexity. Deeper long-term memory results in better scaling in longer
sequences.
Table 3: Performance on long-term forecasting. The best results are highlighted .
Neural Memory
Simba
iTransformer
RLinear
PatchTST
Crossformer
TiDE
TimesNet
DLinear
MSE
MAE
MSE
MAE
MSE
MAE
MSE
MAE
MSE
MAE
MSE
MAE
MSE
MAE
MSE
MAE
MSE
MAE
ETTm1
0.358
0.387
0.383
0.396
0.407
0.410
0.414
0.407
0.387
0.400
0.513
0.496
0.419
0.419
0.400
0.406
0.403
0.407
ETTm2
0.261
0.309
0.271
0.327
0.288
0.332
0.286
0.327
0.281
0.326
0.757
0.610
0.358
0.404
0.291
0.333
0.350
0.401
ETTh1
0.420
0.421
0.441
0.432
0.454
0.447
0.446
0.434
0.469
0.454
0.529
0.522
0.541
0.507
0.458
0.450
0.456
0.452
ETTh2
0.336
0.382
0.361
0.391
0.383
0.407
0.374
0.398
0.387
0.407
0.942
0.684
0.611
0.550
0.414
0.427
0.559
0.515
ECL
0.162
0.261
0.169
0.274
0.178
0.270
0.219
0.298
0.205
0.290
0.244
0.334
0.251
0.344
0.192
0.295
0.212
0.300
Traffic
0.415
0.289
0.493
0.291
0.428
0.282
0.626
0.378
0.481
0.304
0.550
0.304
0.760
0.473
0.620
0.336
0.625
0.383
Weather
0.231
0.265
0.255
0.280
0.258
0.278
0.272
0.291
0.259
0.281
0.259
0.315
0.271
0.320
0.259
0.287
0.265
0.317
augmenting Llama3.1-8B model with RAG performs worse than Titans with about ×70 less parameters.
5.5
The Effect of Deep Memory
In this section, we evaluate the effect of deep memory in both wall-clock training time and model performance2. To this
end, we focus on different variants of our neural memory module, where 𝐿M = 1, 2, 3, 4. We also use Mamba as a baseline
for the model performance. For a fair comparison, we use the same training process for all models and train them on a
subset of the Pile dataset (L. Gao et al. 2020).
We report the perplexity of our models and baselines as the function of the sequence length in Figure 7. Interestingly, with
the increase of memory depth, 𝐿M, the model can achieve better perplexity over all sequence length. Also, deeper memory
modules are more robust to the sequence length when the model has less number of parameters. With the increase of the
number of parameters, all models show better performance on longer sequences.
Figure 8: The effect of memory depth on
training throughput
We also evaluate the effect of memory depth (𝐿M = 1, 2, 3, 4) on the training
throughput. We report the training throughput (the number of tokens per
second) as the function of sequence length in Figure 8. All models scale linearly
with respect to the context length (i.e., constant trend in the number of tokens
per second with respect to sequence length). Also, by increasing the memory
depth, as expected, we can see a linear trend that a deeper memory results in
a slower training. Therefore, it is not always efficient to use deeper memory
modules, showing a trade-off between effectiveness and efficiency.
5.6
Time Series Forecasting
To show the effectiveness of our memory module in a broader tasks, we also evaluate its performance in time series
forecasting tasks. To this end, we use Simba framework (Patro and Agneeswaran 2024) for time series forecasting, and
2Note that, in this experiment, we only focus on the neural memory module to evaluate the effect of memory depth in the memorization process.
Combining neural memory with attention as we do in Titans variants, can additionally enhance the performance of the model over long sequences.
15

Table 4: Downstream evaluation of pre-trained DNA models on GenomicsBenchmarks (Grešová et al. 2023). We report
top-1 classification accuracy (%).

## experiments
N
ext, we evaluate the performance of Titans and its variants in language modeling, commonsense reasoning, needle
in haystack, DNA modeling, and time series forecasting tasks1. In more details, in this section, we answer the
following empirical questions: (1) How do Titans perform compared to baselines in downstream tasks? (see §5.2,
1In the first version of the work, we aim to provide insights/evidences about why the learning paradigms of Titans are effective. We are working on
finalizing the results of larger models and will report them in the next version.
11

§5.6, and §5.7); (2) What is the actual context length of Titans? (see §5.3 and §5.4); (3) How do Titans scale with respect to
context length? (see §5.8); (4) How the depth of memory can affect both performance and efficiency? (see §5.5); and (5)
What is the contribution of each Titans’ component in its performance? (see §5.9).
5.1
Experimental Setup
Models. In our experiments, we focus on the three variants of Titans, which we refer to as: Titans with (1) Memory as a
Context (MAC), (2) Memory as a Gate (MAG), and (3) Memory as a Layer (MAL) as well as (4) neural memory module
alone. The reason behind using our long-term memory as a separate module is based on our definition of learning. As
discussed in Section 1, we define learning a process for acquiring effective and useful memory. Accordingly, we expect our
long-term memory to effectively learn from data, even without attention. For each of these models, we consider four scales
with: (i) 170M, (ii) 340M, (iii) 400M, and (iv) 760M parameters. While the first three are trained on 15B tokens sampled
from FineWeb-Edu dataset (Penedo et al. 2024), the last one is trained on 30B tokens from the same dataset.
Baselines. We compare our models with the state-of-the-art linear recurrent models, Transformers, and hybrid models
(recurrent + attention). More specifically in language tasks, we compare with Transformer++ (Touvron et al. 2023),
RetNet (Yutao Sun et al. 2023), Gated Linear Attention (GLA) (S. Yang, B. Wang, Shen, et al. 2024), Mamba (Gu and Dao
2024), Mamba2 (Dao and Gu 2024), DeltaNet (S. Yang, B. Wang, Yu Zhang, et al. 2024), TTT (Yu Sun et al. 2024), and Gated
DeltaNet (S. Yang, Kautz, and Hatamizadeh 2024). In needle in haystack tasks, we also compare with GPT4 (Achiam et al.
2023), Llama3 with RAG (Touvron et al. 2023), RecurrentGemma2-9B (Botev et al. 2024), and Mistral (Jiang et al. 2023)
models, all of which are provided in the benchmark (Yuri Kuratov et al. 2024). In time series tasks, we compare with
Mamba-based (Behrouz, Santacatterina, and Zabih 2024), Transformer-based (Y. Liu et al. 2023; Nie et al. 2022; Yunhao
Zhang and Yan 2023), and linear models (Das et al. 2023; Z. Li et al. 2023; H. Wu et al. 2023; Zeng et al. 2023).
Training. In the training, we follow the training procedure of S. Yang, Kautz, and Hatamizadeh (2024), and use LLama 2
tokenizer with a vocabulary size of 32K and use training length of 4K tokens. We employ AdamW optimizer with learning
rate of 4𝑒-4 with cosine annealing schedule with batch size of 0.5M tokens, and weight decay of 0.1.
5.2
Language Modeling
We first focus on the perplexity in language modeling and also commonsense reasoning tasks. The results for Titans’
variants and also baselines with three different sizes of 340M, 400M, and 760M are reported in Table 1. Among non-hybrid
models, including Transformer++, our neural memory module achieves the best performance in both perplexity and
accuracy measures. Comparing our neural memory module and TTT, which is also a gradient-based recurrent model can
show us the importance of our weight decay as well as the momentum. As discussed earlier, the weight decay can be
interpreted as a gating mechanism to forget the past data, when it is needed. Also, momentum can help us better manage
the memory by providing additional memory for the surprise metric. While some baselines also take advantage of gating
mechanism, e.g., Mamba, Mamba2, and Gated DeltaNet, the superior performance of our neural memory module shows
the importance of both our surprise mechanism and having deep and non-linear memory. We further discuss the later in
Section 5.5.
Comparing the hybrid models, we found that all three variants of Titans (MAC, MAG, and MAL) outperform both Samba
(Mamba + attention) and Gated DeltaNet-H2 (Gated DeltaNet + atttention). We attribute the superior performance of Titans
(MAL) to the power of neural memory module as the architecture design and used attention are all the same. Comparing
Titans (MAG) and (MAC), we find that while their performance are close, MAC performs better when dealing with longer
dependencies in the data. Interestingly, both MAG and MAC outperform MAL variant, which due to using the same
modules, we attribute this to the architecture design of these models. This finding is particularly important as the current
hybrid models (except Hymba (X. Dong et al. 2024)) in the literature are using MAL-style combination of recurrent models
and attention.
5.3
Needle in a Haystack
Scaling a model to longer context window is not always equivalent to being effective for very long sequences (Hsieh
et al. 2024). The needle-in-a-haystack (NIAH) task is designed to measure the actual effective context length of models.
In this task, we evaluate the model on retrieving a piece of information (i.e., the “needle”) from long distractor texts (i.e.,
12

Table 1: Performance of Titans and recurrent- and Transformer-based baselines on language modeling and common-sense
reasoning tasks. Hybrid models are marked with ∗. The best results among simple and hybrid models are highlighted.

## related_work
I
n this section, we discuss the notation and some background concepts that we use though the paper. We let
𝑥∈R𝑁×𝑑in be the input, M be a neural network (neural memory module), Q, K, V be the query, key and value
of the attention mechanism, and M be the attention mask. When segmenting the sequence, we use S(𝑖) to refer to
the 𝑖-th segment. Through the paper, we abuse the notation and use subscripts to refer to a specific element of a matrix,
vector, or segments. For example, we let S(𝑖)
𝑗
be the 𝑗-th token in the 𝑖-th segment. The only exception is subscripts with 𝑡,
which we reserved to index recurrence over time, or the state of a neural network at time 𝑡. Given a neural network N and
a data sample 𝑥, we use N (𝑥) (resp. N∗(𝑥)) to refer to the forward pass with (resp. without) weight adjustment. Also, we
abuse the notation and use N (𝑘) to refer to the 𝑘-th layer of the neural network. In the following, we first, discuss the
backgrounds for attention and its efficient variants followed by a review of modern linear RNNs. Finally, we discuss a
memory perspective of these architectures that motivates us to design Titans.
2.1
Backgrounds
Attention. Transformers (Vaswani et al. 2017) as the de facto backbone for many deep learning models are based on
attention mechanism. Given input 𝑥∈R𝑁×𝑑in, causal attention computes output y ∈R𝑁×𝑑in based on softmax over input
dependent key, value, and query matrices:
Q = 𝑥WQ,
K = 𝑥WK,
V = 𝑥WV,
(1)
y𝑖=
𝑖∑︁
𝑗=1
exp

Q⊤
𝑖K𝑗/√𝑑in

V𝑗
Í𝑖
ℓ=1 exp

Q⊤
𝑖Kℓ/√𝑑in
 ,
(2)
where WQ, WK, and WV ∈R𝑑in×𝑑in are learnable parameters. Despite the power and effectiveness in recall, transformers
need at least 𝑁× 𝑑operators to calculate the output, resulting in larger memory consumption and lower-throughput for
longer sequences.
Efficient Attentions. To improve the memory consumption and throughput of softmax attention for longer sequences,
various studies focused on I/O aware implementations of attention (Dao 2024; Dao, D. Fu, et al. 2022), designing more
3

efficient attention mechanisms by sparsifying the attention matrix (B. Chen et al. 2021; Choromanski et al. 2021; Dai et al.
2019), approximating the softmax (Arora et al. 2024), or developing kernel-based (linear) attentions (Aksenov et al. 2024;
Kacham, Mirrokni, and P. Zhong 2024; Schlag, Irie, and Jürgen Schmidhuber 2021; S. Yang, B. Wang, Shen, et al. 2024). In
this part, we focus on the later, i.e., linear attentions, where the softmax in standard attention is replaced with an alternative
kernel function 𝜙(., .), such that 𝜙(𝑥,𝑦) = 𝜙(𝑥)𝜙(𝑦). Accordingly, the attention can be written as:
y𝑖=
𝑖∑︁
𝑗=1
𝜙(𝑄⊤
𝑖𝐾𝑗)
Í𝑖
ℓ=1 𝜙(𝑄⊤
𝑖𝐾ℓ)
𝑉𝑗=
𝑖∑︁
𝑗=1
𝜙(𝑄𝑖)⊤𝜙(𝐾𝑗)
Í𝑖
ℓ=1 𝜙(𝑄𝑖)⊤𝜙(𝐾ℓ)
𝑉𝑗=
𝜙(𝑄𝑖)⊤Í𝑖
𝑗=1 𝜙(𝐾𝑗)𝑉𝑗
𝜙(𝑄𝑖)⊤Í𝑖
ℓ=1 𝜙(𝐾ℓ)
,
(3)
resulting in a higher-throughput as terms Í𝑖
𝑗=1 𝜙(𝐾𝑗) and Í𝑖
ℓ=1 𝜙(𝐾ℓ) are re-using in each step. When choosing the kernel
as identity matrix (Yutao Sun et al. 2023), the above formulation can also be written in a recurrent format:
M𝑡= M𝑡−1 + 𝐾⊤
𝑡𝑉𝑡,
(4)
y𝑡= 𝑄𝑡M𝑡,
(5)
which allows efficient inference for linear attentions.
Modern Linear Models and Their Memory Perspective. As discussed earlier, one can define learning as a process for
acquiring effective and useful memory. Building upon this, one can see the hidden state of Recurrent Neural Networks
(RNNs) as a memory unit, which the model aims to compress the information into. Accordingly, in a general form of
recurrent neural network, the hidden state can be treated as a memory unit and the recurrence process can be split into the
read and write operations in the memory unit. That is, we let 𝑥∈R𝑁×𝑑in be the input, M ∈R𝑑is the memory unit, and
y ∈R𝑑in is the output, then the general form of the recurrent neural network is defined as:
M𝑡= 𝑓(M𝑡−1,𝑥𝑡),
Write Operation
(6)
y𝑡= 𝑔(M𝑡,𝑥𝑡),
Read Operation
(7)
where 𝑓(., .) is the read and 𝑔(., .) is the write corresponding functions. Note that here the subscript of M𝑡shows the state
of the memory at time 𝑡.
In this perspective, the recurrence formula of linear Transformers (see Equation 4) is equivalent to additively compress
and write keys and values, (𝐾𝑡,𝑉𝑡), into a matrix-valued memory unit M𝑡. Therefore, when dealing with long context
data, this additive nature of the process results in memory overflow, significantly damaging the performance of the model.
To address this, studies have focused on two promising directions: (1) Adding forget mechanism: several studies have
presented adaptive (data-dependent) forgetting gate mechanisms for linear models, where it can erase the memory when it
is needed. As examples of such models, we refer to GLA (S. Yang, B. Wang, Shen, et al. 2024), LRU (Orvieto et al. 2023),
Griffin (De et al. 2024), xLSTM (Beck et al. 2024), and Mamba2 (Dao and Gu 2024), which the later is also connected to the
discretized version of traditional state space models (Gu and Dao 2024).(2) Improving the write operation: To overcome the
additive nature of memory write operation in traditional recurrent models, Widrow and Hoff (1988) presented Delta Rule,
in which before adding a memory (i.e., a pair of key and value), the model first removes its past value. To enhance the
parallelizable training and scaling, S. Yang, B. Wang, Yu Zhang, et al. (2024) present a fast paralellizable algorithm. Finally,
very recently, S. Yang, Kautz, and Hatamizadeh (2024) improved the DeltaNets by adding a forget gate.
Memory Modules. Memory has always been one of the core parts of the neural network designs (Graves, Wayne,
and Danihelka 2014; JH Schmidhuber 1992; Jürgen Schmidhuber and Hochreiter 1997; J. Zhang et al. 2024). The idea of
seeing linear layers as the key-value (associative) memory system backs to fast weight programs, in which dynamic fast
programs are incorporated into recurrent neural networks to serve as writable memory (JH Schmidhuber 1992). The two
learning rules of Hebbian (Hebb 2005) and delta (Prados and Kak 1989) are the most popular learning rules for fast weight
programs, which have been extensively explored in various studies (Irie, Schlag, et al. 2021; Munkhdalai, Sordoni, et al.
2019; Munkhdalai and H. Yu 2017; Schlag, Irie, and Jürgen Schmidhuber 2021; JH Schmidhuber 1992; S. Yang, Kautz, and
Hatamizadeh 2024; S. Yang, B. Wang, Yu Zhang, et al. 2024). All these models, however, are based on momentary surprise,
missing the token flow in the sequences (see Section 3.1), and most of them lacks a forgetting gate, resulting in a poor
memory management.
We further discuss the connection of our architectures with recent models in Appendix C. Additional related work are
discussed in Appendix A.
4

3
Learning to Memorize at Test Time
T
o overcome the lack of long-term memory and to enable the model to learn, forget, and retrieve information, in
this section, we present a neural long-term memory module, which is a meta models that learns to memorize at
test time. In Section 3.1, we first discuss the motivation and the design of the neural memory. In Section 3.2, we
discuss how our architecture design can benefit from a fast and parallelizable training. Finally, in Section 3.3, we augment
our architecture using persistent memory module, in which we use learnable but data-independent parameters to learn
meta information about the task.
3.1
Long-term Memory
To design a neural long-term memory module, we need a model that can encode the abstraction of the past history into its
parameters. An example of this can be LLMs that are shown to be memorizing their training data (Leybzon and Kervadec
2024; Schwarzschild et al. 2024; Staab et al. 2024). Therefore, a simple idea is to train a neural network and expect it to
memorize its training data. Memorization, however, has almost always been known as an undesirable phenomena in
neural networks as it limits the model generalization (Bayat et al. 2024), causes privacy concerns (Staab et al. 2024), and
so results in poor performance at test time. Moreover, the memorization of the training data might not be helpful at test
time, in which the data might be out-of-distribution. We argue that, we need an online meta-model that learns how to
memorize/forget the data at test time. In this setup, the model is learning a function that is capable of memorization, but it
is not overfitting to the training data, resulting in a better generalization at test time.
Learning Process and Surprise Metric. The key idea to train a long-term memory is to treat its training as an online
learning problem, in which we aim to compress the past information 𝑥1, . . . ,𝑥𝑡−1 into the parameters of our long-term
neural memory module M𝑡. As discussed earlier, an event that violates the expectations (i.e., is surprising) is more
memorable for humans (Mandler 2014). Inspired by this, a simple definition of surprise for a model can be its gradient with
respect to the input. The larger the gradient is, the more different the input data is from the past data. Accordingly, using
this surprise score, we can update the memory as:
M𝑡