# Test-time regression: a unifying framework for designing sequence models with associative memory

> 2025 · id: arxiv:2501.12352 · arXiv: 2501.12352 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Sequence models lie at the heart of modern deep learning. However, rapid advancements have
produced a diversity of seemingly unrelated architectures, such as Transformers and recurrent alter-
natives. In this paper, we introduce a unifying framework to understand and derive these sequence
models, inspired by the empirical importance of associative recall, the capability to retrieve contextu-
ally relevant tokens. We formalize associative recall as a two-step process, memorization and retrieval,
casting memorization as a regression problem. Layers that combine these two steps perform associa-
tive recall via “test-time regression” over its input tokens. Prominent layers, including linear attention,
state-space models, fast-weight programmers, online learners, and softmax attention, arise as special
cases defined by three design choices: the regression weights, the regressor function class, and the
test-time optimization algorithm. Our approach clarifies how linear attention fails to capture inter-
token correlations and offers a mathematical justification for the empirical effectiveness of query-key
normalization in softmax attention. Further, it illuminates unexplored regions within the design space,
which we use to derive novel higher-order generalizations of softmax attention. Beyond unification,
our work bridges sequence modeling with classic regression methods, a field with extensive literature,
paving the way for developing more powerful and theoretically principled architectures.
1

## introduction
Sequences play a vital role in modern machine learning by providing a powerful abstraction: any compu-
tational task can be viewed as transforming one sequence into another (Sutskever et al., 2014). This se-
quential perspective has spread across diverse domains, including natural language processing (Sutskever
et al., 2014; Devlin et al., 2019; Brown et al., 2020), computer vision (Dosovitskiy et al., 2021; Bertasius
et al., 2021), time series analysis (Salinas et al., 2020; Gruver et al., 2023; Ansari et al., 2024), and compu-
tational biology (Jumper et al., 2021; Zhou and Troyanskaya, 2015; Nguyen et al., 2024), highlighting the
importance of building generically applicable sequence layers (Vaswani et al., 2017).
This development has produced a diversity of architectures, each with its own unique characteris-
tics and performance trade-offs. While these architectures have achieved considerable success, they have
largely emerged through separate lines of investigation. Such a fragmented and often empirically-driven
approach to model development limits our ability to systematically understand and improve design choices.
Moreover, the idiosyncratic notations of each architecture obscures their underlying connections (Rush,
2024). Given the wide variety of sequence models, a natural question to ask is whether there is an under-
lying principle that explains why some sequence models work better than others.
∗Correspondence to alxwang@cs.stanford.edu
†Now at Google Deepmind
1
arXiv:2501.12352v3  [cs.LG]  2 May 2025

One empirical discovery that ties together disparate architectures is the strong correlation between an
architecture’s associative recall ability and its language modeling performance (Olsson et al., 2022; Arora
et al., 2023a). Associative recall is the act of retrieving contextually relevant information based on an
association with a query. (Arora et al., 2023a) gives the following example of how associative memory
and retrieval can improve language modeling. Consider the example sentence from Arora et al. (2023b):
“Hakuna Matata, it means no worries for the rest of your days. Hakuna
”. To predict the next word,
we can first memorize the previous occurrence of “Hakuna” and its associated value “Matata”; the next time
we encounter “Hakuna” again we can retrieve its associated value from memory as our prediction. Notice
that we can perform this task using only in-context information and we can make an accurate prediction
even if we have never encountered this strange phrase before. Indeed, transformer-based language models
have been discovered to exhibit this kind of behavior via “induction heads” that emerge during training
(Olsson et al., 2022).
Given the empirical importance of associative recall, how can we systematically design neural
network layers that can perform associative recall (AR)? In this paper, we introduce a simple but
principled framework for deriving sequence layers designed to perform associative recall, which we call
“test-time regression layers”. From the “Hakuna Matata” example, we see that associative recall has
two-steps: memorization and retrieval. Our crucial observation is that we can implement the memo-
rization step by solving a weighted regression problem. We can then generate an output by applying the
regressor to a cue/query token, retrieving the most relevant token from our associative memory. Combin-
ing both memorization and retrieval into the forward pass of a sequence layer results in a layer that performs
regression over the input tokens, a procedure that we call “test-time regression”, visualized in Figure 1.
Our terminology reflects that the regressor is regenerated with each forward pass and depends only on
the input tokens rather than a fixed training dataset.
Since any regression method can be used, our framework provides a general recipe to derive a large
class of sequence layers. Under our framework, an AR-based sequence layer is a mathematical
consequence of choosing (1) the regression weights, (2) the regressor parameterization, and (3)
the optimization algorithm for finding the regressor. In this paper, we consider a few choices of these
three “ingredients”, and show how to derive many recently proposed classes of sequence layers, illustrating
the generality of our framework. Our derivations reveal that linear attention (Katharopoulos et al., 2020),
its feature-mapped variants (e.g. (Peng et al., 2020; Qin et al., 2021; Kasai et al., 2021; Zhang et al., 2023a;
Aksenov et al., 2024; Chen et al., 2024)), its gated variants (Sun et al., 2023; Orvieto et al., 2023; Katsch, 2024;
De et al., 2024; Qin et al., 2024; Peng et al., 2024; Yang et al., 2024b; Beck et al., 2024), state-space model
layers (Gu and Dao, 2024; Dao and Gu, 2024), fast-weights layers (Schlag et al., 2021; Yang et al., 2024a,a),
online learning layers (Liu et al., 2024; Sun et al., 2024; Yang et al., 2024a; Behrouz et al., 2024), and softmax
self-attention (Vaswani et al., 2017) are all test-time regression layers, implicitly performing memorization
followed by retrieval in their forward passes, despite being developed from different perspectives. Figure 1
previews how existing architectures are instantiations of test-time regression layers within our framework.
Our derivations also lead to new understandings of existing sequence layers. We show that layers based
on linear attention underperform because they fail to account for the correlation between tokens. We also
show that query-key normalization, an important technique in stabilizing the training of large language
models (Dehghani et al., 2023; Wortsman et al., 2023), is mathematically necessary to ensure that softmax
self-attention is a proper local constant regressor. Finally, we propose a higher order generalization of
softmax attention, motivated by local linear regression.
Outline of our paper.
We start in Section 2 by examining a few of the most prominent classes of se-
quence layers. Although each class of layers was developed from distinct motivations, their similar com-
putation pattern hints at a common unifying theme that underlies their effectiveness. We then introduce
2

Parametric regression via
batch gradient descent
Linear Attention
DeltaNet
Mesa-layer
Intention
TTT
Longhorn
Gated DeltaNet
Titan-LMM
Delta Product
Intention (kernelized)
Skyformer
SOFT
Softmax attention
Performer
cosFormer
RFA
Hedgehog
...
Based
Rebased
DiJiang
Gated Linear Attention
Mamba
HGRN
Gateloop
mLSTM
LRU
RWKV-6
RetNet
Uniform weights
+ adaptive step size
+ multiple updates
Step 1: Memorize key-value pairs
Step 2: Retrieve a value from memory
+ L2 regularization
+ Momentum
Standard SGD
Kernel regression
Local constant regression
Decaying weights
Parametric regression via 
stochastic gradient descent
Compatible with feature maps for nonlinear regression
Parametric regression via
exact solution
Nonparametric regression
Associative recall as a forward pass
Memorization via test-time regression
A unified perspective on sequence layers
For
Linear associative memory
(less expressive)
Nonlinear associative memory
(more expressive)
Figure 1: Our framework provides a systematic way to derive sequence models that can perform asso-
ciative recall, following a two step process of memorization and retrieval. The memorization step can be
formalized as a solving a regression problem at test-time. Our perspective results in a “recipe” for derv-
ing sequence layers by making three choices: the importance of each association {γ(t)
i }t
i=1, the regressor
function class M, and the optimization algorithm. Parametric regression layers tend to have an efficient
recurrence for updating the memory mt, while non-parametric layers like softmax-attention do not. For
simplicity, we discuss causal sequence models where prefix key-value pairs are memorized, but the same
principles also apply to non-causal ones (e.g. generically masked attention).
our test-time regression framework in Section 3 to formalize the connection between associative mem-
orization and regression. Our perspective provides a systematic approach to designing sequence layers
that produce its outputs via associative recall. In Section 4, we show that indeed all of the aforementioned
classes of sequence layers can be understood from a single unified perspective using the principles of test-
time regression and associative recall. We demonstrate the generality of our framework by deriving these
layers simply by varying how we minimize the regression objective. Section 5 empirically validates that
test-time regression layers implicitly perform regression over its input tokens with a single forward pass.
Then, in Section 6 we examine how to construct effective key-value pairs for associative recall in next-
token prediction tasks. We discuss prior work related to associative recall and memory in Section 7. We
finish in Section 8 by discussing the broader implications of viewing sequence models through the lens of
regression, inc