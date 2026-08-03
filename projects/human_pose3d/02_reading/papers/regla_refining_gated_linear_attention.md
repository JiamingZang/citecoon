# ReGLA: Refining Gated Linear Attention

> 2025 · id: arxiv:2502.01578 · arXiv: 2502.01578 · pdf: https://arxiv.org/pdf/2502.01578 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent advancements in Large Language Mod-
els (LLMs) have set themselves apart with their
exceptional performance in complex language
modelling tasks. However, these models are
also known for their significant computational
and storage requirements, primarily due to the
quadratic computation complexity of softmax
attention. To mitigate this issue, linear atten-
tion has been designed to reduce the quadratic
space-time complexity that is inherent in stan-
dard transformers. In this work, we embarked
on a comprehensive exploration of three key
components that substantially impact the per-
formance of the Gated Linear Attention mod-
ule: feature maps, normalization, and the gat-
ing mechanism. We developed a feature map-
ping function to address some crucial issues
that previous suggestions overlooked. Then we
offered further rationale for the integration of
normalization layers to stabilize the training
process. Moreover, we explored the satura-
tion phenomenon of the gating mechanism and
augmented it with a refining module. We con-
ducted extensive experiments and showed our
architecture outperforms previous Gated Linear
Attention mechanisms in extensive tasks includ-
ing training from scratch and post-linearization
with continual pre-training.
1

## introduction
In the rapidly evolving field of Natural Lan-
guage Processing (NLP), Transformer models have
emerged as a groundbreaking innovation. These
models have demonstrated unparalleled success
across a wide array of tasks, revolutionizing our
approach to understanding and generating natural
language. They have proven their mettle in ana-
lyzing intricate documents, executing professional
writing, and performing sophisticated reasoning
tasks, thereby setting new benchmarks in the realm
†Corresponding author.
*Work conducted while at Huawei Noah’s Ark Lab
of NLP (OpenAI, 2023; Touvron et al., 2023a,b;
Jiang et al., 2024; Xie et al., 2024).
The cornerstone of these Transformer models is
the softmax attention mechanism. This mechanism,
an extension inspired by the attention mechanism
employed in Recurrent Neural Network (RNN)
systems, has played a pivotal role in the success
of Transformer models (Bahdanau et al., 2015;
Vaswani et al., 2017). The softmax attention has
outperformed RNN models in terms of paralleliz-
ability and the stability of gradient propagation
over time, making it a preferred choice for many
NLP tasks.
However, the softmax attention mechanism is
not without its challenges.
It requires substan-
tial computational resources and high memory us-
age, which can be a significant hurdle in practi-
cal applications. As the length of the input in-
creases, the required computation grows quadrati-
cally. This growth restricts the context window size
and complicates the deployment of these models in
real-world scenarios (Kwon et al., 2023). In addi-
tion to the issue of computational complexity, sev-
eral studies have highlighted the limited length ex-
trapolation capability of self-attention-based mod-
els (Press et al., 2022; Ruoss et al., 2023). Specif-
ically, transformer models tend to underperform
during inference if the sequence length of the test
data exceeds that of the training data. As an order-
invariant encoding mechanism, the self-attention-
based encoder heavily depends on Position Embed-
dings (PEs) to model input orders. However, these
studies reveal that the inability of transformers to
handle long sequences can be attributed to the lim-
ited length generalization ability of these position
embedding methods (Press et al., 2022; Zhao et al.,
2024; Wang et al., 2024). This finding underscores
the need to explore alternative architectures to ad-
dress the challenges associated with long-sequence
processing.
Numerous studies have been conducted with the
1
arXiv:2502.01578v3  [cs.CL]  8 Aug 2025

aim of mitigating this drawback by introducing lin-
ear attention operator (Choromanski et al., 2021;
Peng et al., 2021; Katharopoulos et al., 2020; Belt-
agy et al., 2020; Tay et al., 2020; Zhang et al.,
2024; Nahshan et al., 2023; Chen et al., 2024; Ka-
sai et al., 2021). Unfortunately, existing linear at-
tention mechanisms frequently struggle to match
the modeling quality of softmax attention. Some
work introduce gating mechanisms to improve the
performance of linear attention (Schlag et al., 2021;
Mao, 2022; Yang et al., 2024a). In this work, we
delve into the different components of the Gated
Linear Attention mechanism with the goal of op-
timizing the training process while ensuring rapid
inference.
Our contribution can be summarized as follows:
First, we find that previous suggestions overlook
some crucial aspects. We address the instability
issue of feature mapping functions by proposing
a normalized exponential solution. Additionally,
we introduce a variance reduction scaling factor to
enhance its performance. Then we revisit the nor-
malization layer, emphasizing its role in stabilizing
the training process. Finally, we investigate the
saturation phenomenon of the Gating Mechanism
and enhance it with a refining module. By integrat-
ing our findings, we propose a novel architecture
that outperforms previous Gated Linear Attention
mechanisms across various tasks.
2

## method
BoolQ
PIQA
HellaSwag
Winogrande
Truth_QA1
Truth_Qa2
Avg.
0-shot
Pythia-160m
54.6
62.0
30.1
51.0
24.9
44.5
44.5
ReLU
55.5
56.5
26.6
48.6
23.5
47.2
43.0
Hedgehog
60.5
60.4
27.7
50.2
24.4
46.0
44.9
Scalar Gate
55.5
56.5
26.6
48.6
23.5
46.2
42.8
Fast Decay
58.7
59.6
27.1
50.1
25.2
48.4
44.9
REGLA
62.0
58.9
26.9
50.0
25.3
48.8
45.3
5-shot
Pythia-160m
50.6
62.4
30.7
51.4
24.9
44.5
44.1
ReLU
56.5
58.4
26.0
50.2
24.2
45.5
43.5
Hedgehog
61.4
55.6
27.0
50.8
25.7
49.6
45.0
Scalar Gate
57.7
59.8
26.8
51.8
26.4
50.1
45.4
Fast Decay
58.7
60.6
27.1
51.0
25.3
49.5
45.4
REGLA
62.1
60.5
26.8
50.8
25.3
48.8
45.7
Table 5: Results of zero-shot and few-shot evaluation of Post-linearized Pythia-160m models.
models. However, our Refining Gated Linear Atten-
tion (REGLA) method significantly narrows this
performance gap when compared to other methods,
both with and without gating. This underscores
the effectiveness of our design. We also imple-
mented a hybrid architecture that mixes softmax
attention layers with our REGLA layers. In our ex-
periments, the replacement is conducted in a layer-
wise manner. Specifically, for post-linearization,
we replace 50% softmax attention layers (6 out of
12) in a Pythia-160m model with randomly initial-
ized ReGLA modules and do continual training, for
training from scratch, the architecture is the same,
but both softmax attention and ReGLA modules
are randomly initialized. We found this hybrid vari-
ant of REGLA outperforms the softmax attention
method.
In addition to the aforementioned experiments,
we also conducted continual pretraining exper-
iments using pre-trained model checkpoints on
WikiText. These experiments were carried out in a
setting that aligns with those described in previous
studies (Kasai et al., 2021; Mao, 2022). Specif-
ically, we replaced the softmax attention of the
Pythia-160m model with different linear attention
mechanisms and applied continual pre-training to
the entire model on the WikiText-103 dataset.
Our results underscore the versatility of our over-
all design. Not only is it effective when learning
from scratch, but it also offers benefits for post-hoc
linearization. This demonstrates the potential of
our approach to enhance the performance of swap-
ping existing SA models to their linear variants
through continual pretraining.
We further evaluate the zero-shot and few-shot
ability of the post-linearized models on common
sense reasoning tasks, including BoolQ (Clark
et al., 2019), PIQA (Bisk et al., 2020), Hel-
laSwag (Zellers et al., 2019), Winogrande (Sak-
aguchi et al., 2021), TruthfulQA 1 and 2 (Lin
et al., 2022).
The checkpoint of Pythia model
is obtained from HuggingFace2 and we use lm-
evaluation-harness tool (Gao et al., 2023) to per-
form the 0-shot and 5-shot evaluation3. Since our
REGLA also shares the outer product gating for-
mulation as GLA (Yang et al., 2024a), we imple-
mented it based on the Flash Linear Attention4. We
replace the softmax attention layer with our method
and other variants of linear attention. To recover
the performance of the pre-trained model, we per-
form continual pre-training to the post-linearized
model on the SlimPajama dataset (Soboleva et al.,
2023) 50k steps with batch size 8 and maximum
input length 2048.
Results.
Table 5 presents the performance of
various methods across six commonsense reason-
ing datasets. Following continual pretraining, our
model effectively narrows the performance gap on
most benchmarks, with PIQA and Hellaswag be-
ing the notable exceptions. Furthermore, our ap-
proach outperforms all baseline methods on av-
erage, demonstrating its superior performance in
commonsense reasoning tasks.
5
Analysis and Discussion
In this section, we delve into a comprehensive dis-
cussion of our REGLA method. This includes an
evaluation of the effectiveness of the gating mecha-
nism, an analysis of speed and memory usage and
an ablation study to understand the impact of each
component. All of these aspects are examined in
a controlled manner to ensure the reliability of our
2https://huggingface.co/EleutherAI/pythia-160m
3https://github.com/EleutherAI/lm-evaluation-harness
4https://github.com/sustcsonglin/flash-linear-attention
7

findings.
5.1
Gating Analysis
0.2
0.4
0.6
0.8
0
2
4
6
Before (ReGLA)
0.3
0.4
0.5
0.6
0
1
2
3
4
5
After (ReGLA)
0.0
0.2
0.4
0.6
0.8
1.0
0
2
4
6
8
Before (GLA)
0.2
0.4
0.6
0.8
0
1
2
3
4
5
After (GLA)
Figure 4: Distribution of gate activations before and
after the training. We initialize the gate function with
large and small biases to push two methods have very
extreme gate activation values.
In addition to the aforementioned evaluations,
we also conducted a detailed analysis of our refin-
ing gate mechanism. As depicted in Figure 4, we
examined the distributions of the forget gate acti-
vations for both the Gated Linear Attention (GLA)
and our Refining Gated Linear Attention (REGLA)
methods, both before and after the training process.
To validate the effectiveness of our refining gate,
we initialized the gate function with extremely
large and small biases. This was done to push the
initial activation values close to the boundary. The
distribution after training revealed that the vanilla
gating found it challenging to escape the extreme
region. In contrast, our refined gate was able to
learn a diverse range of activation distributions.
Besides, we observed that the gate tended to con-
centrate on values significantly different from 1.0.
This observation suggests that the language model
may have a propensity to favor local information.
5.2
Memory and Speed Analysis
Next, we give an analysis of the inference speed
and peak memory usage of our Refining Gated Lin-
ear Attention (REGLA) mechanism, comparing
it with other methods, notably the Gated Linear
Attention (GLA) with Fast Decay rule and soft-
max attention. Our experiments were conducted
using 6-layer architectures. To ensure a more real-
istic comparison, we employed a Key-Value (KV)
cache for softmax attention. All our experiments
were carried out on a Nvidia V100 32GB GPU.
0
1000
2000
3000
4000
5000
6000
7000
8000
2000
4000
6000
8000
Memory Usage (MB)
GLA-6layer
SA-6layer
ReGLA-6layer
0
1000
2000
3000
4000
5000
6000
7000
8000
Maximum Generation Length
0
100000
200000
300000
400000
Time (ms)
GLA-6layer
SA-6layer
ReGLA-6layer
Figure 5: Plot of memory usage and the total prompt
processing + decoding time of our REGLA, Fast Decay
(GLA) and softmax attention (6-layer) when generating
the next token at various sequence lengths on Nvidia
V100 GPU. Our method and Fast Decay rule consume
approximately the same peak memory and time (over-
lapped in plot).
We maintained a consistent prompt length of 5 and
controlled the maximum generation length from
26 to 213. Figure 5 shows that softmax attention
significantly consumes GPU memory as the output
length increases, leading to a substantial slowdown
in speed. In contrast, our REGLA, when compared
to the Fast Decay rule, achieves nearly the same
speed and memory footprints, demonstrating its
efficiency and practicality.

## experiments
In this section, we evaluate our method with other
linear attention and the conventional transformer.
This comparison spans autoregressive language
modeling training from scratch and finetuning pre-
trained language models after replacing its softmax
attention with linear variants. To justify our design
choices for REGLA, we conduct a comprehensive
ablation study and efficiency analysis.

## related_work
We first briefly revisit the linear attention. Our
method is grounded on these works by analyzing
the essential components of them.
2.1
Softmax Attention
The softmax attention (SA) is the key component
of the state-of-the-art transformer architectures.
Given a sequence of N query vectors {qi}, which
attend to M key and value vectors. The atten-
tion module aggregates the values with the nor-
malized outputs of a softmax function (Vaswani
et al., 2017):
SA(qi, {kj}, {vj}) =
X
j
exp(q⊤
i kj/
√
d)
P
j′ exp(q⊤
i kj′/
√
d)
vj,
(1)
where qi, ki, vi are d dimensional vectors. For
a given input query qi, computing the attention
necessitates time and space complexity of O(M),
leading to a memory footprint of O(MN) for full
N queries. This bottleneck makes attention-based
LLMs difficult to scale in terms of context window
size since growing input length not only substan-
tially escalates GPU computation but also compli-
cates the management of Key-Value (KV) cache,
particularly for decoder-based LLMs (Kwon et al.,
2023).
2.2
Linear Attention
Linear Attention (LA) (Katharopoulos et al., 2020;
Choromanski et al., 2021; Peng et al., 2021; Zheng
et al., 2022; Qin et al., 2022; Nahshan et al., 2023)
exchanges the computation order by decomposing
the softmax function with randomized or learnable
feature functions. Eq.1 can then be rewritten as
hi =
P
j vjϕ(kj)⊤ϕ(qi)
P
j′ ϕ(kj′)⊤ϕ(qi) ,
(2)
where ϕ : Rd →Rm is a m dimensional fea-
ture mapping function. Such an order exchang-
ing enables to avoid computing the attention ma-
trix of size RN×M for the full sequence and re-
duces the time complexity to O(N). Existing meth-
ods generally utilize different functions to approxi-
mate softmax kernels. For example, Choromanski
et al. (2021) propose a positive Orthogonal Ran-
dom features approach (Favor+) and Peng et al.
(2021) leverages random Fourier features to ap-
proximate attention functions, Katharopoulos et al.
(2020) adopt a learnable linear transformation with
1 + elu(·) activation as the feature map and Ka-
sai et al. (2021) propose to use a learned ReLU
function: ϕ(x) = ReLU(W x + b) as the feature
map.
Another benefit of this feature map-based atten-
tion is that Eq. 2 can be further regrouped as a linear
recurrence formulation because of the associative
property of the matrix product as:
St = St−1 + vtϕ(kt)⊤,
(3)
ct = ct−1 + ϕ(kt),
(4)
ht = Stϕ(qt)
c⊤
t ϕ(qt),
(5)
where St ∈Rd×m is the recurrent state matrix
and ct ∈Rm is the normalization vector. This
linear recurrence can be regarded as a variant of
fast weight additive outer products (Schmidhuber,
2

1992; Schlag et al., 2021). These techniques con-
centrate on either estimating or modifying the soft-
max operator, thus maintaining its original char-
acteristics. When contrasted with the softmax at-
tention, these techniques frequently sacrifice per-
formance for efficiency, typically leading to dimin-
ished task performance.
2.3
Linear Attention with Gating Mechanisms
(GLA)
Instead of approximating self-attention rigorously,
recent works focus on improving the hidden
state representation by introducing different gat-
ing mechanisms (Peng et al., 2021; Schlag et al.,
2021; Mao, 2022).
Peng et al. (2021) propose
to add a gated update rule to Linear Attention
which is inspired by gated recurrent neural net-
works (Hochreiter and Schmidhuber, 1997; Cho
et al., 2014; Chung et al., 2014) to forget distant
input with a recency bias. The state updating rule
is as follows:
St = gtSt−1 + (1 −gt)vtϕ(kt)⊤,
(6)
ct = gtct−1 + (1 −gt)ϕ(kt),
(7)
where gt = Sigmoid(Wgx) ∈R is a function with
learnable parameters Wg ∈R1×d. Schlag et al.
(2021) propose a way to improve the vanilla gating
method as Fast Weight Programmer (Schmidhuber,
1992) to forget information related to the current
write key:
St = St−1 −gtSt−1ϕ(kt)ϕ(kt)⊤+ gtvtϕ(kt)⊤.
(8)
Mao (2022) investigates various update rule config-
urations and proposes a fast decaying rule inspired
by Ba et al. (2016) and removes feature maps. The
update rule is as:
St = Gt ⊙St−1 + vtϕ(kt)⊤,
(9)
Gt = σ(Wzxt + bz)σ(Wfxt + bf)⊤,
(10)
where Wz ∈Rd×d, Wf ∈Rm×d, bz ∈Rd,
bf ∈Rm are trainable parameters, ⊙is Hadamard
product, and σ is the Sigmoid function. This gated
rule learns to output a gating matrix instead of a
scalar, thus leading to a more fine-grained informa-
tion control. This mechanism is also adopted in a
recent work (Yang et al., 2024a) which develops
a chunked parallel formulation for gated linear at-
tention to achieve more hardware-friendly training
for large-scale models. Pramanik et al. (2023) also
X
+
X
safe exp
safe exp
X
+
X
NORM
Figure 1: The overall model architecture of our REGLA.
The right side depicts the regular linear attention with
our safe exp feature maps and normalization layer and
the left side depicts the refining gate mechanism.
utilize this fast decay rule and evaluate their recur-
rent linear transformer in reinforcement learning
problems.
Compared to softmax attention’s implicit un-
bounded memory footprint requirement:
KV
cache (Kwon et al., 2023), linear attention has
bounded memory size during the inference, which
is much easier to deploy and manage for language
models in service. However, both the memory
size of hidden states and the mechanism of up-
dating rule have a great impact on the performance
of these Linear models. For example,
Schlag
et al. (2021) develop the Deterministic Parameter-
Free Projection (DPFP) to expand the outer prod-
uct dimension and use delta rule to edit the for-
get/write mechanism of hidden states, but Mao
(2022) demonstrates this underperforms the gating
method. All these findings show that it’s more cru-
cial to concentrate on creating an expressive update
rule for gate linear attention. It is not conclusive
which architecture: softmax attention or linear At-
tention is superior. Also, techniques developed
by efficient attention can be directly or indirectly
adapted to various modern large language models
to improve the deployment, i.e., Qin et al. (2023a)
develop the first large-scale linear attention-based
LLM and Slide Window Attention (SWA) (Beltagy
et al., 2020) is reported being used in Mistral (Jiang
et al., 2023) to achieve context extension for long
input sequences.
3

## conclusion
In this study, our primary focus is on auto-
regressive tasks. We believe that a concentrated ex-
amination of these tasks allows us to delve deeper
into the nuances and intricacies involved, thereby
providing more insightful and meaningful findings.
Furthermore, our method is designed to investigate
the fundamental components of linear attention
methods. We aim to understand the underlying
principles and mechanisms that drive the perfor-
mance of these architectures. This approach allows
us to identify potential areas for improvement and
propose innovative solutions to enhance their ef-
9

fectiveness. We have not conducted large-scale
experiments in this study. Our decision to limit the
scale of our experiments is intentional. We believe
that by focusing on a smaller, more manageable
scale, we can maintain a high level of control and
precision in our experiments. This approach en-
sures the reliability of our results and allows us to
draw more accurate conclusions.