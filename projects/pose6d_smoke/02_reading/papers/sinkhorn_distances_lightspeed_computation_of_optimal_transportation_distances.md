# Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances

> 2013 · id: arxiv:1306.0895 · arXiv: 1306.0895 · pdf: https://arxiv.org/pdf/1306.0895 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Optimal transportation distances (Villani, 2009, §6) – also known as Earth
Mover’s following the seminal work of Rubner et al. (1997) and their application
to computer vision – hold a special place among other distances in the probability
simplex.
Compared to other classic distances or divergences, such as Hellinger,
χ2, Kullback-Leibler or Total Variation, they are the only ones to be parameter-
ized.
This parameter – the ground metric – plays an important role to handle
high-dimensional histograms: the ground metric provides a natural way to han-
dle redundant features that are bound to appear in high-dimensional histograms
(think synonyms for bags-of-words), in the same way that Mahalanobis distances
can correct for statistical correlations between vector coordinates.
The central role played by histograms and bags-of-features in most data analysis
tasks and the good performance of optimal transportation distances in practice has
generated ample interest, both from a theoretical point of view (Levina and Bickel,
2001; Indyk and Thaper, 2003; Naor and Schechtman, 2007; Andoni et al., 2009)
and a pracical aspect, mostly to compare images (Grauman and Darrell, 2004;
Ling and Okada, 2007; Gudmundsson et al., 2007; Shirdhonkar and Jacobs, 2008).
Optimal transportation distances have, however, a very clear drawback. No matter
what the algorithm employed – network simplex or interior point methods – their
cost scales at least in O(d3log(d)) when computing the distance between a pair of
histograms of dimension d, in the general case where no restrictions are placed upon
the ground metric parameter (Pele and Werman, 2009, §2.1). This speed can be
improved by ensuring that the ground metric observes certain constraints and/or
by accepting some approximation errors. However, when these restrictions do not
apply, computing a single distance between a pair of histograms of dimension in
1
arXiv:1306.0895v1  [stat.ML]  4 Jun 2013

2
MARCO CUTURI
the few hundreds can take more than a few seconds. This issue severely hinders
the applicability of optimal transportation distances in large-scale data analysis
and goes as far as putting into question their relevance within the ﬁeld of machine
learning.
Our aim in this paper is to show that the optimal transportation problem can
be regularized by an entropic term, following the maximum-entropy principle. We
argue that this regularization is intuitive given the geometry of the optimal trans-
portation problem and has, in fact, been long known and favored in transportation
theory (Erlander and Stewart, 1990). From an optimization point of view, this
regularization has multiple virtues, among which that of turning this LP into a
strictly convex problem that can be solved extremely quickly with the Sinkhorn-
Knopp matrix scaling algorithm (Sinkhorn and Knopp, 1967; Knight, 2008). This
algorithm exhibits linear convergence and can be trivially parallelized – it can be
vectorized. It is therefore amenable to large scale executions on parallel platforms
such as GPGPUs. From a practical perspective, we show that, on the benchmark
task of classifying MNIST digits, Sinkhorn distances perform better than the EMD
and can be computed several orders of magnitude faster over a large sample of
dimensions without making any assumption on the ground metric. We believe this
paper contains all the ingredients that are required for optimal transportation dis-
tances to be at last applied on high-dimensional datasets and attract again the
attention of the machine learning community.
This paper is organized as follows: we provide reminders on optimal transporta-
tion theory in Section 2, introduce Sinkhorn distances in Section 3 and provide
algorithmic details in Section 4. We follow with an empirical study in Section 5
before concluding.
2. Reminders on Optimal Transportation
2.1. Transportation Tables and Joint Probabilities. In what follows, ⟨·, ·⟩
stands for the Frobenius dot-product. For two histograms r and c in the simplex
Σd
def
= {x ∈Rd
+ : xT 1d = 1}, we write U(r, c) for the transportation polytope of r
and c, namely the polyhedral set of d × d matrices:
U(r, c)
def
= {P ∈Rd×d
+
| P1d = r, P T 1d = c},
where 1d is the d dimensional vector of ones. U(r, c) contains all nonnegative d × d
matrices with row and column sums r and c respectively. U(r, c) has a probabilistic
interpretation: for X and Y two multinomial random variables taking values in
{1, · · · , d}, each with distribution r and c respectively, the set U(r, c) contains
all possible joint probabilities of (X, Y ). Indeed, any matrix P ∈U(r, c) can be
identiﬁed with a joint probability for (X, Y ) such that p(X = i, Y = j) = pij. Such
joint probabilities are also known as contingency tables. We deﬁne the entropy h
and the Kullback-Leibler divergences of these tables and their marginals as
r ∈Σd,
h(r) = −
d
X
i=1
ri log ri,
P ∈U(r, c),
h(P) = −
d
X
i,j=1
pij log pij
P, Q ∈U(r, c),
KL(P∥Q) =
X
ij
pij log pij
qij
.

SINKHORN DISTANCES
3
2.2. Optimal Transportation. Given a d×d cost matrix M, the cost of mapping
r to c using a transportation matrix (or joint probability) P can be quantiﬁed as
⟨P, M ⟩. The following problem:
dM(r, c)
def
=
min
P ∈U(r,c)⟨P, M ⟩.
is called an optimal transportation problem between r and c given cost M. An
optimal table P ⋆for this problem can be obtained with the network simplex (Ahuja
et al., 1993, §9) as well as other approaches (Orlin, 1993). The optimum of this
problem, dM(r, c), is a distance (Villani, 2009, §6.1) whenever the matrix M is itself
a metric matrix, namely whenever M belongs to the cone of distance matrices (Avis,
1980; Brickell et al., 2008):
M = {M ∈Rd×d
+
: ∀i ≤d, mii = 0; ∀i, j, k ≤d, mij ≤mik + mkj}.
For a general matrix M, the worst case complexity of computing that optimum
with any of the algorithms known so far scales in O(d3 log d) and turns out to be
super-cubic in practice as well (Pele and Werman, 2009, §2.1). Much faster speeds
can be obtained however when placing all sorts of restrictions on M and accepting
approximated solutions, albeit at a cost in performance (Grauman and Darrell,
2004) and a loss in applicability.
3. Sinkhorn Distances
We consider in this section a family of optimal transportation distances whose
feasible set is the not the whole of U(r, c), but a parameterized restricted set of
joint probability matrices.
3.1. Entropic Constraints on Joint Probabilities. We recall a basic informa-
tion theoretic inequality (Cover and Thomas, 1991, §2) which applies to all joint
probabilities:
(1)
∀r, c ∈Σd, ∀P ∈U(r, c), h(P) ≤h(r) + h(c).
This bound is tight, since the table rcT – known as the independence table (Good,
1963) – has an entropy of h(rcT ) = h(r) + h(c). By the concavity of entropy, we
can introduce the convex set Uα(r, c) ⊂U(r, c) as
Uα(r, c)
def
= {P ∈U(r, c) | KL(P∥rcT ) ≤α} = {P ∈U(r, c) | h(P) ≥h(r) + h(c) −α}
These deﬁnitions are indeed equivalent, since one can easily check that
KL(P∥rcT ) = h(r) + h(c) −h(P),
a quantity which is also the mutual information I(X∥Y ) of two random variables
(X, Y ) should they follow the joint probability P (Cover and Thomas, 1991, §2).
Hence, all tables P whose Kullback-Leibler divergence to the table rcT is con-
strained to lie below a certain threshold can be interpreted as the set of tables
P in U(r, c) which have suﬃcient entropy with respect to h(r) and h(c), or joint
probabilities which display a small enough mutual information.
As a classic result of linear optimization, the optimum of classical optimal trans-
portation distances is achieved on vertices of U(r, c), that is d × d matrices with
only up to 2d −1 non-zero elements (Brualdi, 2006, §8.1.3). Such plans can be
interpreted as quasi-deterministic joint probabilities, since if pij > 0, then very few
values pij′ will have a non-zero probability. By mitigating the transportation cost

4
MARCO CUTURI
M
dM,α(r, c) = ⟨P ⋆, M⟩
U(r, c)
rcT
P ⋆
Uα(r, c) = {P ∈U(r, c)| KL(P ∥rcT) ≤α}
Figure 1. Schematic view of the transportation polytope and the
Kullback-Leibler ball of level α that surrounds the independence
table rcT . The Sinkhorn distance is the dot product of M with
the optimal transportation table in that ball.
objective with an entropic constraint, which is equivalent to following the max-
entropy principle (Jaynes, 1957; Dud´ık and Schapire, 2006) and thus for a given
level of the cost look for the most smooth joint probability, we argue that we can
provide a more robust notion of distance between histograms. Indeed, for a given
pair (r, c), ﬁnding plausible transportation plans with low cost (where plausibility
is measured by entropy) is more informative than ﬁnding extreme plans that are
extremely unlikely to appear in nature.
We note that the idea of regularizing the transportation problem was also con-
sidered recently by Ferradans et al. (2013). In their work, Ferradans et al. also
argue that an optimal matching may not be suﬃciently regular in vision applica-
tions (co

## experiments
5.1. MNIST Digits. We test the performance of Sinkhorn distances on the MNIST
digits3 dataset, on which the ground metric has a natural interpretation in terms of
pixel distances. Each digit is provided as a vector of intensities on a 20 × 20 pixel
grid. We convert each image into a histogram by normalizing each pixel intensity
by the total sum of all intensities . We consider a subset of N points in the training
set of the database, where N ranges within {3, 5, 12, 17, 25} × 103 datapoints.
5.1.1. Experimental setting. For each subset of size N, we provide mean and stan-
dard deviation of classiﬁcation error using a 4 fold (3 test, 1 train) cross validation
scheme repeated 6 times, resulting in 24 diﬀerent experiments. We study the per-
formance of diﬀerent distances with the following parameter selection scheme: for
each distance d, we consider the kernel e−d/t, where t > 0 is chosen by cross val-
idation individually for each training fold within the set {1, q10(d), q20(d), q50(d)},
where qs is the s% quantile of a subset of distances observed in the training fold.
We regularize non-positive deﬁnite kernel matrices resulting from this computation
by adding a suﬃciently large diagonal term. SVM’s were run with libsvm (one-vs-
one) for multiclass classiﬁcation, the regularization constant C being selected by 2
folds/2 repeats cross-validation on the training fold in the set 10−2:2:4
3http://yann.lecun.com/exdb/mnist/

8
MARCO CUTURI
Figure 2. Average test errors with shaded conﬁdence intervals.
Errors are computed using 1/4 of the dataset for train and 3/4 for
test. Errors are averaged over 4 folds × 6 repeats = 24 experiments.
5.1.2. Distances. The Hellinger, χ2, Total Variation and squared Euclidean (Gauss-
ian kernel) distances are used as such. We set the ground metric M to be the
Euclidean distance between the 20 × 20 points in the grid, resulting in a 400 × 400
distance matrix. We also tried to use Mahalanobis distances on this example with
a positive deﬁnite matrix equal to exp(-tM.^2), t>0, as well as its inverse, with
varying values of t but none of the results proved competitive. For the Indepen-
dence kernel, since any Euclidean distance matrix is valid, we consider [ma
ij] where
a ∈{0.01, 0.1, 1} and choose a by cross-validation on the training set.
Smaller
values of a seem to be preferable. We select the entropic penalty λ of Sinkhorn dis-
tances so that the matrix e−λM is relatively diagonally dominant and the resulting
transportation not too far from the classic optimal transportation. We select λ for
each training fold by internal cross-validation within {5, 7, 9, 11} × 1/q50(M) where
q50(M) is the median distance between pixels on the grid. We set the number of
ﬁxed-point iterations to an arbitrary number of 20 iterations. In most (though not
all) folds, the value λ = 9 comes up as the best setting. The Sinkhorn distance
beats by a safe margin all other distances, including the EMD.
5.2. Does the Sinkhorn Distance Converge to the EMD?. We study in this
section the convergence of Sinkhorn distances towards classical optimal transporta-
tion distances as λ gets bigger. Because of the additional penalty that appears in
(2) program, dλ
M(r, c) is necessarily larger than dM(r, c), and we expect this gap to
decrease as λ increases. Figure 3 illustrates this by plotting the boxplot of distri-
butions of (dλ
M(r, c) −dM(r, c))/dM(r, c) over 402 pairs of distinct points taken in
the MNIST database. As can be observed, even with large values of λ, Sinkhorn
distances hover above the values of EMD distances by about 10%. For practical
values of λ such as λ = 9 selected above we do not expect the Sinkhorn distance to
be numerically close to the EMD, nor believe it to be a desirable property.

SINKHORN DISTANCES
9
1
3
5
7
9
11
13
15
17
19
21
23
25
0.2
0.4
0.6
0.8
1
1.2
1.4
h
Distribution of (Sinkhorn−EMD)/EMD
Deviation of Sinkhorn’s Distance
to EMD on subset of MNIST Data
Figure 3. Decrease of the gap between the Sinkhorn distance and
the EMD on the MNIST dataset.
64
128
256
512
1024
2048
4096
10
−6
10
−4
10
−2
10
0
10
2
10
4
Histogram Dimension
Avg. Execution Time per Distance (in s.)
Computational Speed for Histograms of
Varying Dimension Drawn Uniformly on the Simplex
(log log scale)
 
 
FastEMD
Rubner’s emd
Sink. CPU h=9
Sink. GPU h=9
Sink. CPU h=1
Sink. GPU h=1
Figure 4. Average computational time required to compute a
distance between two histograms sampled uniformly in the d di-
mensional simplex for varying values of d. Sinkhorn distances are
run both on a single CPU node and on a GPU card, until the
variation in x becomes smaller than ϵ = 0.01 in Euclidean norm.
5.3. Several Orders of Magnitude Faster. We measure in this section the com-
putational speed of classic optimal transportation distances vs. that of Sinkhorn

10
MARCO CUTURI
Figure 5. The inﬂuence of λ on the number of iterations required
to converge on histograms uniformly sampled from the simplex.
distances using Rubner et al.’s (1997)4 and Pele and Werman’s (2009)5 publicly
available implementations. We generate points uniformly in the d-simplex (Smith
and Tromble, 2004) and generate random distance matrices M by selecting d points
distributed with a spherical Gaussian in dimension d/10 to obtain enough vari-
ability in the distance matrix.
M is then divided by the median of its values,
M=M/median(M(:)). Sinkhorn distances are implemented in matlab code (see Al-
gorithm 1) while emd mex, emd hat gd metric are mex/C ﬁles. The emd distances
and Sinkhorn CPU are run on a matlab session with a single working core (2.66
Ghz Xeon). Sinkhorn GPU is run on an NVidia Quadro K5000 card. Following the
experimental ﬁndings of Section 5.1, we consider two parameters for λ, λ = 1 and
λ = 9. λ = 1 results in a relatively dense matrix K = e−λM, with results compara-
ble to that of the Independence kernel, while λ = 9 results in a matrix K = e−λM
with mostly negligible values and therefore a matrix with low entropy that is closer
to the optimal transportation solution. Rubner et al.’s implementation cannot be
run for histograms larger than d = 512. For large dimensions and on the same
CPU, Sinkhorn distances are more than 100.000 faster than EMD solvers given a
threshold of 0.01. Using a GPU results in a speed-up of a supplementary order of
magnitude.
5.4. Empirical Complexity. To provide an accurate picture of the actual num-
ber of steps required to guarantee the algorithm’s convergence, we replicate the
experiments of Section 5.3 but focus now on the number of iterations of the loop
described in Algorithm 1. We use a tolerance of 0.01 on the norm of the diﬀerence
of two successive iterations of x ∈Rd. As can be seen in Figure 5, the number of
iterations required so that ∥x −x′∥2 ≤0.01 increases as e−λM becomes diagonally
dominant. From a practical perspective, and because keeping track of the change
4http://robotics.stanford.edu/ rubner/emd/default.htm
5http://www.cs.huji.ac.il/ ofirpele/FastEMD/code/, we use emd hat gd metric in these

## conclusion
We have shown that regularizing the optimal transportation problem with an
intuitive entropic penalty opens the door for new research directions and poten-
tial applications at the intersection of optimal transportation theory and machine
learning. This regularization guarantees speed-ups that are eﬀective whatever the
structure of the ground metric M. Based on preliminary evidence, it seems that
Sinkhorn distances do not perform worse than the EMD, and may in fact perform
better in applications. Sinkhorn distances are parameterized by a regularization
weight λ which should be tuned having both computational and performance ob-
jectives in mind, but we have not observed a need to establish a trade-oﬀbetween
both. Indeed, reasonably small values of λ seem to perform better than large ones.
7. Appendix: Proofs
Proof of Property 1. The set U1(r, c) contains all joint probabilities P for which
h(P) = h(r)+h(c). In that case (Cover and Thomas, 1991, Theorem 2.6.6) applies
and U1(r, c) can only be equal to the singleton {rcT }. If M is negative deﬁnite, there
exists vectors (ϕ1, · · · , ϕd) in some Euclidean space Rn such that mij = ∥ϕi −ϕj∥2
2
through (Berg et al., 1984, §3.3.2). We thus have that
rT Mc =
X
ij
ricj∥ϕi −ϕj∥2 = (
X
i
ri∥ϕi∥2 +
X
i
ci∥ϕi∥2) −2
X
ij
⟨riϕi, cjϕj ⟩
= rT u + cT u −2rT Kc
where ui = ∥φi∥2 and Kij = ⟨ϕi, ϕj ⟩. We used the fact that P ri = P ci = 1 to go
from the ﬁrst to the second equality. rT Mc is thus a n.d. kernel because it is the
sum of two n.d. kernels: the ﬁrst term (rT u + cT u) is the sum of the same function
evaluated separately on r and c, and thus a negative deﬁnite kernel (Berg et al.,
1984, §3.2.10); the latter term −2rT Ku is negative deﬁnite as minus a positive
deﬁnite kernel (Berg et al., 1984, Deﬁnition §3.1.1).
Remark. The proof above suggests a faster way to compute the Independence
kernel. Given a matrix M, one can indeed pre-compute the vector of norms u as
well as a Cholesky factor L of K above to preprocess a dataset of histograms by
premultiplying each observations ri by L and only store Lri as well as precomputing
its diagonal term rT
i u. Note that the independence kernel is positive deﬁnite on
histograms with the same 1-norm, but is no longer positive deﬁnite for arbitrary
vectors.
Proof of Lemma 1. Let T be the a probability distribution on {1, · · · , d}d whose
coeﬃcients are deﬁned as
(4)
tijk
def
= pijqjk
yj
,

12
MARCO CUTURI
for all indices j such that yj > 0. For indices j such that yj = 0, all values tijk are
set to 0.
Let S
def
= [P
j tijk]ik. S is a transportation matrix between x and z. Indeed,
X
i
X
j
sijk =
X
j
X
i
pijqjk
yj
=
X
j
qjk
yj
X
i
pij =
X
j
qjk
yj
yj =
X
j
qjk = zk (column sums)
X
k
X
j
sijk =
X
j
X
k
pijqjk
yj
=
X
j
pij
yj
X
k
qjk =
X
j
pij
yj
yj =
X
j
pij = xi (row sums)
We now prove that h(S) ≥h(x)+h(z)−α. Let (X, Y, Z) be three random variables
jointly distributed as T. Since by deﬁnition of T in Equation (4)
p(X, Y, Z) = p(X, Y )p(Y, Z)/p(Y ) = p(X)p(Y |X)p(Z|Y ),
the triplet (X, Y, Z) is a Markov chain X →Y →Z (Cover and Thomas, 1991,
Equation 2.118) and thus, by virtue of the data processing inequality (Cover and
Thomas, 1991, Theorem 2.8.1), the following inequality between mutual informa-
tions applies:
I(X; Y ) ≥I(X; Z), namely
h(X, Z)−h(X)+h(Z) ≥h(X, Y )−h(X)+h(Y ) ≥−α.