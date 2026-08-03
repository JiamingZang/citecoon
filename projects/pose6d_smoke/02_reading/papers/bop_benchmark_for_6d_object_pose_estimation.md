# BOP: Benchmark for 6D Object Pose Estimation

> 2018 · id: W2888752296 · arXiv: 1808.08319 · pdf: https://arxiv.org/pdf/1808.08319 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Estimating the 6D pose, i.e. 3D translation and 3D rotation, of a rigid object
has become an accessible task with the introduction of consumer-grade RGB-D
sensors. An accurate, fast and robust method that solves this task will have a
big impact in application ﬁelds such as robotics or augmented reality.
Many methods for 6D object pose estimation have been published recently,
e.g. [34,24,18,2,36,21,27,25], but it is unclear which methods perform well and
in which scenarios. The most commonly used dataset for evaluation was created
by Hinterstoisser et al. [14], which was not intended as a general benchmark and
has several limitations: the lighting conditions are constant and the objects are
easy to distinguish, unoccluded and located around the image center. Since then,
some of the limitations have been addressed. Brachmann et al. [1] added ground-
truth annotation for occluded objects in the dataset of [14]. Hodaˇn et al. [16]
created a dataset that features industry-relevant objects with symmetries and
similarities, and Drost et al. [8] introduced a dataset containing objects with
reﬂective surfaces. However, the datasets have diﬀerent formats and no standard
evaluation methodology has emerged. New methods are usually compared with
only a few competitors on a small subset of datasets.
∗Authors have been leading the project jointly.
arXiv:1808.08319v1  [cs.CV]  24 Aug 2018

2
Hodaˇn, Michel et al.
LM/LM-O [14,1]
IC-MI [34]
IC-BIN [7]
T-LESS [16]
RU-APC [28]
TUD-L - new
TYO-L - new
1
2
3
4
5
6
7
8
9
10
11 12 13 14 15 16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
T-LESS
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
1
2
3
4
5
6
LM/LM-O
IC-MI/IC-BIN
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17 18
19
20
21
TYO-L
1
2
3
1
2
3
4
5
6
7
8
9
10
11
12
13
14
TUD-L
RU-APC
Fig. 1. A collection of benchmark datasets. Top: Example test RGB-D images where
the second row shows the images overlaid with 3D object models in the ground-truth
6D poses. Bottom: Texture-mapped 3D object models. At training time, a method is
given an object model or a set of training images with ground-truth object poses. At
test time, the method is provided with one test image and an identiﬁer of the target
object. The task is to estimate the 6D pose of an instance of this object.
This work makes the following contributions:
1. Eight datasets in a uniﬁed format, including two new datasets focusing
on varying lighting conditions, are made available (Fig. 1). The datasets con-
tain: i) texture-mapped 3D models of 89 objects with a wide range of sizes,
shapes and reﬂectance properties, ii) 277K training RGB-D images showing
isolated objects from diﬀerent viewpoints, and iii) 62K test RGB-D images
of scenes with graded complexity. High-quality ground-truth 6D poses of the
modeled objects are provided for all images.
2. An evaluation methodology based on [17] that includes the formulation
of an industry-relevant task, and a pose-error function which deals well with
pose ambiguity of symmetric or partially occluded objects, in contrast to the
commonly used function by Hinterstoisser et al. [14].
3. A comprehensive evaluation of 15 methods on the benchmark datasets
using the proposed evaluation methodology. We provide an analysis of the
results, report the state of the art, and identify open problems.
4. An online evaluation system at bop.felk.cvut.cz that allows for con-
tinuous submission of new results and provides up-to-date leaderboards.

BOP: Benchmark for 6D Object Pose Estimation
3
1.1

## experiments
Accuracy. Tab. 2 and 3 show the recall scores of the evaluated methods per
dataset and per object respectively, for the misalignment tolerance τ = 20 mm
and the correctness threshold θ = 0.3. Ranking of the methods according to the
recall score is mostly stable across the datasets. Methods based on point-pair
features perform best. Vidal-18 is the top-performing method with the average
recall of 74.6%, followed by Drost-10-edge, Drost-10, and the template matching
method Hodaˇn-15, all with the average recall above 67%. Brachmann-16 is the
best learning-based method, with 55.4%, and Buch-17-ppfh is the best method
based on 3D local features, with 54.0%. Scores of Buch-16-si and Buch-16-shot
are inferior to the other variants of this method and not presented.
Fig. 4 shows the average of the per-dataset recall scores for diﬀerent values
of τ and θ. If the misalignment tolerance τ is increased from 20 mm to 80 mm,
the scores increase only slightly for most methods. Similarly, the scores increase
only slowly for θ > 0.3. This suggests that poses estimated by most methods are
either of a high quality or totally oﬀ, i.e. it is a hit or miss.
Speed. The average running times per test target are reported in Tab. 2. How-
ever, the methods were evaluated on diﬀerent computers3 and thus the presented
running times are not directly comparable. Moreover, the methods were opti-
mized primarily for the recall score, not for speed. For example, we evaluated
Drost-10 with several parameter settings and observed that the running time
can be lowered by a factor of ∼5 to 0.5 s with only a relatively small drop of
the average recall score from 68.1% to 65.8%. However, in Tab. 2 we present the
result with the highest score. Brachmann-14 could be sped up by sub-sampling
the 3D object models and Hodaˇn-15 by using less object templates. A study of
such speed/accuracy trade-oﬀs is left for future work.
Open Problems. Occlusion is a big challenge for current methods, as shown by
scores dropping swiftly already at low levels of occlusion (Fig. 4, right). The big
gap between LM and LM-O scores provide further evidence. All methods per-
form on LM by at least 30% better than on LM-O, which includes the same but
partially occluded objects. Inspection of estimated poses on T-LESS test images
conﬁrms the weak performance for occluded objects. Scores on TUD-L show that
varying lighting conditions present a serious challenge for methods that rely on
3 Speciﬁcations of computers used for the evaluation are on the project website.

BOP: Benchmark for 6D Object Pose Estimation
13
# Method
LM
LM-O
IC-MI
IC-BIN
T-LESS
RU-APC
TUD-L
Average
Time (s)
1. Vidal-18
87.83
59.31
95.33
96.50
66.51
36.52
80.17
74.60
4.7
2. Drost-10-edge
79.13
54.95
94.00
92.00
67.50
27.17
87.33
71.73
21.5
3. Drost-10
82.00
55.36
94.33
87.00
56.81
22.25
78.67
68.06
2.3
4. Hodan-15
87.10
51.42
95.33
90.50
63.18
37.61
45.50
67.23
13.5
5. Brachmann-16
75.33
52.04
73.33
56.50
17.84
24.35
88.67
55.44
4.4
6. Hodan-15-nopso
69.83
34.39
84.67
76.00
62.70
32.39
27.83
55.40
12.3
7. Buch-17-ppfh
56.60
36.96
95.00
75.00
25.10
20.80
68.67
54.02
14.2
8. Kehl-16
58.20
33.91
65.00
44.00
24.60
25.58
7.50
36.97
1.8
9. Buch-17-si
33.33
20.35
67.33
59.00
13.34
23.12
41.17
36.81
15.9
10. Brachmann-14
67.60
41.52
78.67
24.00
0.25
30.22
0.00
34.61
1.4
11. Buch-17-ecsad
13.27
9.62
40.67
59.00
7.16
6.59
24.00
22.90
5.9
12. Buch-17-shot
5.97
1.45
43.00
38.50
3.83
0.07
16.67
15.64
6.7
13. Tejani-14
12.10
4.50
36.33
10.00
0.13
1.52
0.00
9.23
1.4
14. Buch-16-ppfh
8.13
2.28
20.00
2.50
7.81
8.99
0.67
7.20
47.1
15. Buch-16-ecsad
3.70
0.97
3.67
4.00
1.24
2.90
0.17
2.38
39.1
Table 2. Recall scores (%) for τ = 20 mm and θ = 0.3. The recall score is the percentage
of test targets for which a correct object pose was estimated. The methods are sorted
by their average recall score calculated as the average of the per-dataset recall scores.
The right-most column shows the average running time per test target.
# Method
LM
LM-O
TUD-L
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
1
5
6
8
9
10
11
12
1
2
3
1. Vidal-18
89
96
91
94
92
96
89
89
87
97
59
69
93
92
90
66
81
46
65
73
43
26
64
79
88
74
2. Drost-10-edge
77
97
94
40
98
94
83
96
45
94
68
66
72
88
79
47
82
46
75
42
44
36
57
85
88
90
3. Drost-10
86
83
89
84
93
87
86
92
66
96
53
67
79
91
80
62
75
39
70
57
46
26
57
73
90
74
4. Hodan-15
91
97
79
97
91
97
73
69
90
97
81
79
99
74
95
54
66
40
26
73
37
44
68
27
63
48
5. Brachmann-16
92
93
76
84
86
90
44
72
85
79
46
67
94
60
66
64
65
44
68
71
3
32
61
81
95
91
6. Hodan-15-nr
91
57
40
89
66
87
59
49
92
90
65
63
71
54
79
47
35
24
12
63
9
32
53
12
52
20
7. Buch-17-ppfh
77
65
0
94
84
60
24
59
75
67
24
39
75
47
62
59
63
18
35
60
17
5
30
55
89
63
8. Kehl-16
60
52
81
25
79
68
17
68
42
91
45
42
78
83
46
39
47
24
30
48
14
13
49
0
23
0
9. Buch-17-si
40
43
1
63
81
47
12
8
36
43
18
3
46
19
43
54
63
11
2
16
9
1
3
2
74
48
10. Brachmann-14
74
70
77
75
88
66
11
81
69
66
50
75
92
75
49
50
48
27
44
60
6
30
62
0
0
0
11. Buch-17-ecsad
31
2
2
19
66
3
3
0
9
49
1
0
3
7
6
29
29
0
0
7
8
1
0
1
62
10
12. Buch-17-shot
3
4
11
9
9
4
1
3
2
10
1
0
10
12
14
2
7
0
0
1
1
1
0
1
33
17
13. Tejani-14
36
0
36
0
1
0
1
11
1
70
27
0
0
0
0
26
2
0
1
0
0
10
0
0
0
0
14. Buch-16-ppfh
11
0
1
22
3
7
2
7
18
12
4
3
9
12
14
4
0
0
2
11
1
1
1
2
0
0
15. Buch-16-ecsad
2
0
0
9
5
0
0
4
5
8
0
0
17
3
5
1
3
0
2
2
0
0
0
0
1
0
IC-MI
-BIN
T-LESS
1
2
3
4
5
6
2
4
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
1. Vidal-18
80 100 100 98 100 94 100 93
43
46
68
65
69
71
76
76
92
69
68
84
55
47
54
85
82
79
2. Drost-10-edge
78 100 100 100 90
96 100 84
53
44
61
67
71
73
75
89
92
72
64
81
53
46
55
85
88
78
3. Drost-10
76 100 98 100 96
96 100 74
34
46
63
63
68
64
54
48
59
54
51
69
43
45
53
80
79
68
4. Hodan-15
100 100 100 74
98 100 100 81
66
67
72
72
61
60
52
61
86
72
56
55
54
21
59
81
81
79
5. Brachmann-16
42
98
70
88
64
78
84
29
8
10
21
4
46
19
52
22
12
7
3
3
0
0
0
5
3
54
6. Hodan-15-nr
100 100 92
62
60
94
93
59
64
67
71
73
62
57
49
56
85
70
57
55
60
23
60
82
81
77
7. Buch-17-ppfh
88 100 94 100 100 88 100 50
1
7
0
5
25
16
4
35
37
48
4
10
4
0
0
12
34
49
8. Kehl-16
22 100 70
72
96
30
71
17
7
10
18
24
23
10
0
2
11
17
5
1
0
9
12
56
52
22
9. Buch-17-si
62 100 94
62
52
34
97
21
0
1
17
17
9
3
1
4
0
8
2
0
0
0
0
20
26
12
10. Brachmann-14
96 100 66
72
46
92
28
20
0
0
1
0
0
0
0
0
1
0
0
0
0
0
0
0
1
2
11. Buch-17-ecsad
66
88
0
56
34
0
95
23
0
0
0
0
0
1
1
0
0
0
0
0
0
0
0
1
0
8
12. Buch-17-shot
52
88
38
36
40
4
66
11
0
0
1
0
1
5
0
2
1
0
0
1
0
1
1
2
1
3
13. Tejani-14
42
36
0
40
26
74
4
16
0
1
1
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
14. Buch-16-ppfh
28
34
20
6
24
8
4
1
1
6
3
1
24
4
10
13
10
13
3
8
1
0
0
5
32
13
15. Buch-16-ecsad
4
4
8
4
2
0
5
3
0
1
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
2
T-LESS
RU-APC
19
20
21
22
23
24
25
26
27
28
29
30
1
2
3
4
5
6
7
8
9
10
11
12
13
14
1. Vidal-18
57
43
62
69
85
66
43
58
62
69
69
85
39
38
42
54
53
43
4
82
32
0
48
47
20
8
2. Drost-10-edge
55
47
55
56
84
59
47
69
61
80
84
89
0
20
35
47
35
39
0
89
28
0
48
21
15
3
3. Drost-10
53
35
60
61
81
57
28
51
32
60
81
71
0
11
29
45
33
29
26
71
10
0
47
9
0
0
4. Hodan-15
59
27
57
50
74
59
47
72
45
73
74
85
4
36
59
24
47
46
52
97
28
28
34
52
17
0
5. Brachmann-16
38
1
39
19
61
1
16
27
17
13
6
5
6
64
25
21
32
41
47
37
1
0
18
40
0
5
6. Hodan-15-nr
58
27
55
50
73
60
49
72
40
72
76
85
4
39
50
24
41
15
43
91
25
33
31
39
16
1
7. Buch-17-ppfh
31
25
36
35
71
46
64
51
4
44
49
58
16
5
17
51
27
6
57
24
8
10
55
5
11
0
8. Kehl-16
35
5
26
27
71
36
28
51
34
54
86
69
19
14
46
38
54
40
4
80
3
5
3
37
7
5
9. Buch-17-si
11
21
18
11
37
4
52
53
3
35
32
53
24
49
16
39
3
4
32
54
14
9
43
15
17
5
10. Brachmann-14
0
0
0
0
1
0
1
1
0
0
0
0
6
80
42
19
31
33
52
89
19
1
0
40
7
0
11. Buch-17-ecsad
16
11
16
8
27
20
51
31
0
32
22
3
1
2
0
1
3
8
23
34
5
8
2
0
3
1
12. Buch-17-shot
6
6
8
2
28
3
17
13
0
11
7
6
0
0
0
0
0
0
0
1
0
0
0
0
0
0
13. Tejani-14
0
0
0
0
0
0
0
0
0
0
0
2
1
0
0
3
9
0
0
5
0
0
0
3
0
0
14. Buch-16-ppfh
3
3
8
8
16
2
24
4
5
11
6
1
0
0
6
19
2
12
34
8
0
0
38
2
5
0
15. Buch-16-ecsad
2
1
3
0
10
0
12
1
2
4
1
1
0
3
5
0
1
1
11
13
0
0
3
2
0
1
Table 3. Recall scores (%) per object for τ = 20 mm and θ = 0.3.

14
Hodaˇn, Michel et al.
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
recall
= 20
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
recall
= 80
0
2000
4000
6000
8000
10000
test targets
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
visibility [%]
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
recall
 
Vidal-18
Drost-10-edge
Drost-10
Hodan-15
Brachmann-16
Hodan-15-nr
Buch-17-ppfh
Kehl-16
Buch-17-si
Brachmann-14
Buch-17-ecsad
Buch-17-shot
Tejani-14
Buch-16-ppfh
Buch-16-ecsad
Fig. 4.
Left, middle: Average of the per-dataset recall scores for the misalignment
tolerance τ ﬁxed to 20 mm and 80 mm, and varying value of the correctness threshold θ.
The curves do not change much for τ > 80 mm. Right: The recall scores w.r.t. the visible
fraction of the target object. If more instances of the target object were present in the
test image, the largest visible fraction was considered.
synthetic tra

## related_work
The progress of research in computer vision has been strongly inﬂuenced by
challenges and benchmarks, which enable to evaluate and compare methods and
better understand their limitations. The Middlebury benchmark [31,32] for depth
from stereo and optical ﬂow estimation was one of the ﬁrst that gained large
attention. The PASCAL VOC challenge [10], based on a photo collection from
the internet, was the ﬁrst to standardize the evaluation of object detection and
image classiﬁcation. It was followed by the ImageNet challenge [29], which has
been running for eight years, starting in 2010, and has pushed image classiﬁcation
methods to new levels of accuracy. The key was a large-scale dataset that enabled
training of deep neural networks, which then quickly became a game-changer for
many other tasks [23]. With increasing maturity of computer vision methods,
recent benchmarks moved to real-world scenarios. A great example is the KITTI
benchmark [11] focusing on problems related to autonomous driving. It showed
that methods ranking high on established benchmarks, such as the Middlebury,
perform below average when moved outside the laboratory conditions.
Unlike the PASCAL VOC and ImageNet challenges, the task considered in
this work requires a speciﬁc set of calibrated modalities that cannot be easily
acquired from the internet. In contrast to KITTY, it was not necessary to record
large amounts of new data. By combining existing datasets, we have covered
many practical scenarios. Additionally, we created two datasets with varying
lighting conditions, which is an aspect not covered by the existing datasets.
2
Evaluation Methodology
The proposed evaluation methodology formulates the 6D object pose estimation
task and deﬁnes a pose-error function which is compared with the commonly
used function by Hinterstoisser et al. [13].
2.1
Formulation of the Task
Methods for 6D object pose estimation report their predictions on the basis
of two sources of information. Firstly, at training time, a method is given a
training set T = {To}n
o=1, where o is an object identiﬁer. Training data To may
have diﬀerent forms, e.g. a 3D mesh model of the object or a set of RGB-D
images showing object instances in known 6D poses. Secondly, at test time, the
method is provided with a test target deﬁned by a pair (I, o), where I is an
image showing at least one instance of object o. The goal is to estimate the 6D
pose of one of the instances of object o visible in image I.
If multiple instances of the same object model are present, then the pose
of an arbitrary instance may be reported. If multiple object models are shown
in a test image, and annotated with their ground truth poses, then each object
model may deﬁne a diﬀerent test target. For example, if a test image shows three
object models, each in two instances, then we deﬁne three test targets. For each
test target, the pose of one of the two object instances has to be estimated.

4
Hodaˇn, Michel et al.
This task reﬂects the industry-relevant bin-picking scenario where a robot
needs to grasp a single arbitrary instance of the required object, e.g. a component
such as a bolt or nut, and perform some operation with it. It is the simplest
variant of the 6D localization task [17] and a common denominator of its other
variants, which deal with a single instance of multiple objects, multiple instances
of a single object, or multiple instances of multiple objects. It is also the core of
the 6D detection task, where no prior information about the object presence in
the test image is provided [17].
2.2
Measuring Error
A 3D object model is deﬁned as a set of vertices in R3 and a set of polygons that
describe the object surface. The object pose is represented by a 4 × 4 matrix
P = [R, t; 0, 1], where R is a 3 × 3 rotation matrix and t is a 3 × 1 translation
vector. The matrix P transforms a 3D homogeneous point xm in the model
coordinate system to a 3D point xc in the camera coordinate system: xc = Pxm.
Visible Surface Discrepancy. To calculate the error of an estimated pose ˆP
w.r.t. the ground-truth pose ¯P in a test image I, an object model M is ﬁrst
rendered in the two poses. The result of the rendering is two distance maps1 ˆS
and ¯S. As in [17], the distance maps are compared with the distance map SI of
the test image I to obtain the visibility masks ˆV and ¯V , i.e. the sets of pixels
where the model M is visible in the image I (Fig. 2). Given a misalignment
tolerance τ, the error is calculated as:
eVSD( ˆS, ¯S, SI, ˆV , ¯V , τ) =
avg
p∈ˆV ∪¯V
(
0
if p ∈ˆV ∩¯V ∧| ˆS(p) −¯S(p)| < τ
1
otherwise.
(1)
Properties of eVSD. The object pose can be ambiguous, i.e. there can be
multiple poses that are indistinguishable. This is caused by the existence of
multiple ﬁts of the visible part of the object surface to the entire object surface.
The visible part is determined by self-occlusion and occlusion by other objects
and the multiple surface ﬁts are induced by global or partial object symmetries.
Pose error eVSD is calculated only over the visible part of the model surface
and thus the indistinguishable poses are treated as equivalent. This is a desirable
property which is not provided by pose-error functions commonly used in the
literature [17], including eADD and eADI discussed below. As the commonly used
pose-error functions, eVSD does not consider color information.
Deﬁnition (1) is diﬀerent from the original deﬁnition in [17] where the pixel-
wise cost linearly increases to 1 as | ˆS(p)−¯S(p)| increases to τ. The new deﬁnition
is easier to interpret and does not penalize small distance diﬀerences that may
be caused by imprecisions of the depth sensor or of the ground-truth pose.
1 A distance map stores at a pixel p the distance from the camera center to a 3D point
xp that projects to p. It can be readily computed from the depth map which stores
at p the Z coordinate of xp and which can be obtained by a Kinect-like sensor.

BOP: Benchmark for 6D Object Pose Estimation
5
RGBI
SI
ˆS
ˆV
¯S
¯V
S∆
Fig. 2. Quantities used in the calculation of eVSD. Left: Color channels RGBI (only for
illustration) and distance map SI of a test image I. Right: Distance maps ˆS and ¯S are
obtained by rendering the object model M at the estimated pose ˆP and the ground-
truth pose ¯P respectively. ˆV and ¯V are masks of the model surface that is visible in I,
obtained by comparing ˆS and ¯S with SI. Distance diﬀerences S∆(p) = ˆS(p) −¯S(p),
∀p ∈ˆV ∩¯V , are used for the pixel-wise evaluation of the surface alignment.
a: 0.04
3.7/15.2
b: 0.08
3.6/10.9
c: 0.11
3.2/13.4
d: 0.19
1.0/6.4
e: 0.28
1.4/7.7
f: 0.34
2.1/6.4
g: 0.40
2.1/8.6
h: 0.44
4.8/21.7
i: 0.47
4.8/9.2
j: 0.54
6.9/10.8
k: 0.57
6.9/8.9
l: 0.64
21.0/21.7
m: 0.66
4.4/6.5
n: 0.76
8.8/9.9
o: 0.89
49.4/11.1
p: 0.95
32.8/10.8
Fig. 3. Comparison of eVSD (bold, τ = 20 mm) with eADI/θAD (mm) on example pose
estimates sorted by increasing eVSD. Top: Cropped and brightened test images overlaid
with renderings of the model at i) the estimated pose ˆP in blue, and ii) the ground-
truth pose ¯P in green. Only the part of the model surface that falls into the respective
visibility mask is shown. Bottom: Diﬀerence maps S∆. Case (b) is analyzed in Fig. 2.
Criterion of Correctness. An estimated pose ˆP is considered correct w.r.t.
the ground-truth pose ¯P if the error eVSD < θ. If multiple instances of the
target object are visible in the test image, the estimated pose is compared to the
ground-truth instance that minimizes the error. The choice of the misalignment
tolerance τ and the correctness threshold θ depends on the target application.
For robotic manipulation, where a robotic arm operates in 3D space, both τ and
θ need to be low, e.g. τ = 20 mm, θ = 0.3, which is the default setting in the
evaluation presented in Sec. 5. The requirement is diﬀerent for augmented reality
applications. Here the surface alignment in the Z dimension, i.e. the optical axis
of the camera, is less important than the alignment in the X and Y dimension.
The tolerance τ can be therefore relaxed, but θ needs to stay low.

6
Hodaˇn, Michel et al.
Comparison to Hinterstoisser et al. In [14], the error is calculated as the
average distance from vertices of the model M in the ground-truth pose ¯P to
vertices of M in the estimated pose ˆP. The distance is measured to the position
of the same vertex if the object has no indistinguishable views (eADD), otherwise
to the position of the closest vertex (eADI). The estimated pose ˆP is considered
correct if e ≤θAD = 0.1d, where e is eADD or eADI, and d is the object diameter,
i.e. the largest distance between any pair of model vertices.
Error eADI can be un-intuitively low because of many-to-one vertex match-
ing established by the search for the closest vertex. This is shown in Fig. 3,
which compares eVSD and eADI on example pose estimates of objects that have
indistinguishable views. Overall, (f)-(n) yield low eADI scores and satisfy the
correctness criterion of Hinterstoisser et al.

## conclusion
We have proposed a benchmark for 6D object pose estimation that includes eight
datasets in a uniﬁed format, an evaluation methodology, a comprehensive evalu-
ation of 15 recent methods, and an online evaluation system open for continuous
submission of new results. With this benchmark, we have captured the status
quo in the ﬁeld and will be able to systematically measure its progress in the fu-
ture. The evaluation showed that methods based on point-pair features perform
best, outperforming template matching methods, learning-based methods and
methods based on 3D local features. As open problems, our analysis identiﬁed
occlusion, varying lighting conditions, and object symmetries and similarities.
Acknowledgements
We gratefully acknowledge Manolis Lourakis, Joachim Staib, Christoph Kick,
Juil Sock and Pavel Haluza for their help. This work was supported by CTU
student grant SGS17/185/OHK3/3T/13, Technology Agency of the Czech Re-
public research program TE01020415 (V3C – Visual Computing Competence
Center), and the project for GAˇCR, No. 16-072105: Complex network methods
applied to ancient Egyptian data in the Old Kingdom (2700–2180 BC).

BOP: Benchmark for 6D Object Pose Estimation
15