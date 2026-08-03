# HumanEva: Synchronized Video and Motion Capture Dataset and Baseline Algorithm for Evaluation of Articulated Human Motion

> 2009 · id: W2099333815 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

HumanEva: Synchronized Video and Motion Capture Dataset
and Baseline Algorithm for Evaluation of Articulated Human
Motion
Leonid Sigal
University of Toronto
Department of Computer Science
ls@cs.toronto.edu
Alexandru O. Balany
Brown University
Department of Computer Science
alb@cs.brown.edu
Michael J. Black
Brown University
Department of Computer Science
black@cs.brown.edu
July 27, 2009
Abstract
While research on articulated human motion and pose estimation has progressed rapidly in the last few years,
there has been no systematic quantitative evaluation of competing methods to establish the current state of the art.
We present data obtained using a hardware system that is able to capture synchronized video and ground-truth 3D
motion. The resulting HUMANEVA datasets contain multiple subjects performing a set of predened actions with
a number of repetitions. On the order of 40; 000 frames of synchronized motion capture and multi-view video
(resulting in over one quarter million image frames in total) were collected at 60 Hz with an additional 37; 000 time
instants of pure motion capture data. A standard set of error measures is dened for evaluating both 2D and 3D
pose estimation and tracking algorithms. We also describe a baseline algorithm for 3D articulated tracking that uses
a relatively standard Bayesian framework with optimization in the form of Sequential Importance Resampling and
Annealed Particle Filtering. In the context of this baseline algorithm we explore a variety of likelihood functions, prior
models of human motion and the effects of algorithm parameters. Our experiments suggest that image observation
models and motion priors play important roles in performance, and that in a multi-view laboratory environment,
where initialization is available, Bayesian ltering tend s to perform well. The datasets and the software are made
available to the research community. This infrastructure will support the development of new articulated motion and
pose estimation algorithms, will provide a baseline for the evaluation and comparison of new methods, and will help
establish the current state of the art in human pose estimation and tracking.
1
Introduction
The recovery of articulated human motion and pose from video has been studied extensively in the past 20 years
with the earliest work dating to the early 1980’s [28, 53]. A variety of statistical [1, 2, 7, 17, 30, 74, 75, 76] as well
as deterministic methods [46, 83, 69] have been developed for tracking people from single [1, 2, 21, 30, 36, 45, 46,
58, 59, 60, 63, 75] or multiple [7, 17, 26, 74] views. All these methods make different choices regarding the state
space representation of the human body and the image observations required to infer this state from the image data.
Despite clear advances in the eld, evaluation of these meth ods remains mostly heuristic and qualitative. As a result,
 The rst two authors contributed equally to this work.
yThe work was conducted at Brown University.
1

it is difcult to evaluate the current state of the art with an y certainty or even to compare different methods with any
rigor.
Quantitative evaluation of human pose estimation and tracking is currently limited due to the lack of common
datasets containing ground truth with which to test and co mpare algorithms. Instead qualitative tests are still widely
used and evaluation often relies on visual inspection of results. This is usually achieved by projecting the estimated 3D
body pose into the image (or set of images) and visually assessing how the estimates explain the image [17, 21, 60].
Another form of inspection involves applying the estimated motion to a virtual character to see if the movements
appear natural [76]. The lack of the quantitative experimentation at least in part can be attributed to the difculty of
obtaining 3D ground-truth data that specify the true pose of the body observed in video sequences.
To obtain some form of ground truth, previous approaches have resorted to custom action-specic schemes;
e.g. motion of the arm along a circular path of known diameter [34]. Alternatively, synthetic data have been ex-
tensively used [1, 2, 26, 69, 76] for quantitative evaluation. With packages such as POSER (e frontier, Scotts Valley,
CA) or MAYA (Autodesk, San Rafael, CA), semi-realistic images of humans can be rendered and used for evaluation.
Such images, however, typically lack realistic camera noise, often contain very simple backgrounds and provide sim-
plied types of clothing. While synthetic data allow quanti tative evaluation, current datasets are still too simplistic to
capture the complexities of natural images of people and scenes.
In the last few years, there have been a few successful attempts [23, 35, 47, 65] to simultaneously capture video
and ground truth 3D motion data (in the form of marker-based tracking); some groups were also able to capture 2D
motion ground truth data in a similar fashion [89]. Typically hardware systems similar to the one proposed here have
been employed [35] where the video and motion capture data were captured either independently (and synchronized
in software off-line) or with hardware synchronization. While this allowed some quantitative analysis of results [23,
35, 47, 65, 89], to our knowledge none of the synchronized data captured by these groups (with the exception of [89],
discussed in Section 2) has been made available to the community at large, making it hard for competing approaches
to compare performance directly. For 2D human pose/motion estimation, quantitative evaluation is more common and
typically uses hand-labeled data [30, 58, 59]. Furthermore, for both 2D and 3D methods, no standard error measures
exist and results are reported in a variety of ways which prevent direct comparison; e.g. average root-mean-squared
(RMS) angular error [1, 2, 76], normalized error in joint angles [69], silhouette overlap [58, 59], joint center distance
[7, 26, 36, 39, 41, 74, 75], etc.
Here we describe two datasets containing human activity with associated ground truth that can be used for quanti-
tative evaluation and comparison of both 2D and 3D methods. We hope that the creation of these datasets, which we
call HUMANEVA, will advance the state of the art in human motion and pose estimation by providing a structured,
comprehensive, development dataset with support code and quantitative evaluation measures. The motivation behind
the design of the HUMANEVA datasets is that, as a research community, we need to answer the following questions:
 What is the state-of-the art in human pose estimation?
 What is the state-of-the art in human motion tracking?
 What algorithm design decisions affect human pose estimation and tracking performance and to what extent?
 What are the strengths and weaknesses of different pose estimation and tracking algorithms?
 What are the main unsolved problems in human pose estimation and tracking?
In answering these questions, comparisons must be made across a variety of different methods and models to nd
which choices are most important for a practical and robust solution. To support this analysis, the HUMANEVA
datasets contain a number of subjects performing repetitions (trials) of a varied set of predened actions. The dataset s
are broken into training, validation, and test sub-sets. For the testing subset, the ground truth data are withheld and a
web-based evaluation system is provided. A set of error measures is dened and made available as part of the dataset.
These error measures are general enough to be applicable to most current pose estimation and tracking algorithms
and body models. Support software for manipulating the data and evaluating results is also made available as part
of the HUMANEVA datasets. This support code shows how the data and error measures can be used and provides an
easy-to-use Matlab (The Mathworks, Natick, MA) interface to the data. This allows different methods to be fairly
compared using the same data and the same error measures.
2

In addition we provide a baseline algorithm for 3D articulated tracking in the form of simple Bayesian ltering.
We analyze the performance of the baseline algorithm under a variety of parameter choices and show how these
parameters affect the performance. The reported results on the HUMANEVA-II dataset are intended to be the baseline
against which future algorithms that use the dataset can be compared. In addition, this Bayesian ltering software
is freely available, and can serve as a foundation for new algorithm development and experimentation with image
likelihood models and new prior models of human motion.
In systematically addressing the problems of articulated human pose estimation and tracking using the HUMAN-
EVA datasets, other related research areas may benet as well, s uch as foreground/background segmentation, appear-
ance modeling and voxel carving. It is worth noting that similar efforts have been made in related areas including the
development of datasets for face detection [55, 56], human gait identication [27, 67], dense stereo vision [68] and
optical ow [4]. These efforts have helped advance the state -of-the-art in their respective elds. Our hope is that the
HUMANEVA datasets will lead to similar advances in articulated human pose and motion estimation. In the short time
that the dataset has been made available to the research community, it has already helped with the development and
evaluation of new approaches for articulated motion estimation [8, 9, 38, 40, 41, 50, 62, 84, 88, 91]. The dataset has
also served as a basis for a series of workshops on Evaluation of Human Motion and Pose Estimation (EHuM)1 set
forth by the authors.
2
Related work
Articulated Pose and Motion Estimation.
Classically the solutions to articulated human motion estimation fall into
two categories: pose estimation and tracking. Pose estimation is usually formulated as the inference of the articulated
human pose from a single image (or in a multi-view setting, from multiple images captured at the same time). Tracking,
on the other hand, is formulated as inference of the human pose over a set of consecutive image frames throughout an
image sequence. Tracking approaches often assume knowledge of the initial pose of the body in the rst frame and
focus on the evolution of this pose over time. These approaches can be combined [74, 76], such that tracking benets
from automatic initialization and failure recovery in the form of static pose estimation and pose estimation benets
from temporal coherence constraints.
It is important to note that both tracking and pose estimation can be performed in 2D, 2.5D, or 3D, corresponding
to different ways of modeling the human body. In each case, the body is typically represented by an articulated
set of parts corresponding naturally to body parts (limbs, head, hands, feet, etc.). Here 2D refers to models of the
body that are dened directly in the image plane while 2.5D ap proaches also allow the model to have relative depth
information. Finally 3D approaches typically model the human body using simplied 3-dimensional parts such as
cylinders or superquadrics. A short summary of different approaches with evaluation and error measures employed
(when appropriate) can be seen in Table 1; for a more complete taxonomy, particularly of older work, we refer readers
to [24] and [44].
Common Datasets.
While HUMANEVA is the most extensive dataset for evaluation of human pose and motion
estimation, there have been several related efforts. A similar approach was employed by Wang et al. [89] where syn-
chronized motion capture and monocular video was collected. The dataset, used by the authors to analyze performance
of 2D articulated tracking algorithms, is available to the public5. The dataset, however, only contains 4 sequences (2
of which come from old movie footage and required manual labeling); only 2D ground truth marker positions are
provided. The INRIA Perception Group also employed a similar approach for collection of ground truth data [35],
however, only the multi-view video data is currently made available to the public.
The CMU Graphics Lab Motion Capture Database [15] is by far the most extensive dataset of publicly available
motion capture data. It has been used by many researchers within the community to build prior models of human
motion. The dataset, however, is not well suited for evaluating video-based tracking performance. While, for many
of the motion capture sequences, low-resolution monocular videos are available, the calibration information required
1While the workshops did not have any printed proceedings, submissions can be viewed on-line:
http://www.cs.brown.edu/people/ls/ehum/
http://www.cs.brown.edu/people/ls/ehum2/
5http://www.cc.gt.atl.ga.us/grads/w/Ping.Wang/Projec
t/FigureTracking.html
3

Table 1: Short survey of the human motion and tracking algorithms. Methods are listed in the chronological order
by the rst author.
Type refers to the type of the approach, where (P) corresponds to the pose-estimation and (T)
to tracking. Approaches that employ (?) and (??) evaluation measures are consistent with the evaluation measures
proposed in this paper.
Year
First Author
Model Type
Parts
Dim
Type
Evaluation
Measure
1983
Hogg [28]
Cylinders
14
2.5
T
Qualitative
1996
Ju [33]
Patches
2
2
T
Qualitative
1996
Kakadiaris [34]
D Silhouettes
2
3
T
Quantitative
1998
Bregler [11]
Ellipsoids
10
3
T
Qualitative*
2000
Rosales [64]
Stick-Figure
10
3
P
Synthetic
? 2
2000
Sidenbladh [73]
Cylinders
2/10
3
T
Qualitative
2002
Ronfard [63]
Patches
15
2
P
Hand Labeled
2002
Sidenbladh [71]
Cylinders
2/10
3
T
Qualitative
2003
Grauman [26]
Mesh
N/A
3
P
Synthetic/POSER
?
2003
Ramanan [59]
Rectangles
10
2
T,P
Hand Labeled

2003
Shakhnarovich [69]
Mesh
N/A
3
P
Synthetic/POSER
z
2003
Sminchisescu [78, 79]
Superquadric Ellip.
15
3
T
Qualitative3
2004
Agarwal [1, 2]
Mesh
N/A
3
P
Synthetic/POSER
y
2004
Deutscher [17]
R-Elliptical Cones
15
3
T
Qualitative
2004
Lan [37]
Rectangles
10
2
T,P
Qualitative
2004
Mori [46]
Stick-Figure
9
3
P
Qualitative
2004
Roberts [61]
Prob. Template
10
2
P
Qualitative
2004
Sigal [74]
R-Elliptical Cones
10
3
T,P
Motion Capture
??
2005
Balan [7]
R-Elliptical Cones
10
3
T
Motion Capture
??
2005
Felzenszwalb [21]
Rectangles
10
2
P
Qualitative
2005
Hua [30]
Quadrangular
10
2
P
Hand Labeled
\
2005
Lan [36]
Rectangles
10
2
P
Motion Capture
?
2005
Ramanan [58]
Rectangles
10
2
T,P
Hand Labeled

2005
Ren [60]
Stick-Figure
9
2
P
Qualitative
2005
Sminchisescu [76]
Mesh
N/A
3
T,P
Synthetic/POSER
y
2006
Gall [23]
Mesh
N/A
3
T
Motion Capture
y
2006
Lee [39]
R-Elliptical Cones
5/10
3
T,P
Hand Labeled
?? 4
2006
Li [41]
R-Elliptical Cones
10
3
T
HUMANEVA
??
2006
Rosenhahn [65]
Free-form surface patches
N/A
3
T
Motion Capture
y
2006
Sigal [75]
Quadrangular
10
2
P
Motion Capture
?
2006
Urtasun [85]
Stick-gure
15
3
T
Qualitative
2006
Wang [89]
SPM + templates
10
2
T
Motion Capture
? and 
2007
Lee [38]
Joint centers
N/A
3
T
HUMANEVA
??
2007
Mundermann [47]
SCAPE
15
3
T
Motion Capture
?? and 
2007
Navaratnam [48]
Mesh
N/A
3
P
Motion Capture
y
2007
Srinivasan [82]
Exemplars
6
2
P
Hand Labeled
? and 
2007
Xu [91]
Cylinders
10
3
T
HUMANEVA
??
2008
Bo [9]
Joint centers
N/A
3
P
HUMANEVA
??
2008
Ning [50]
Stick-gure
10
3
P
HUMANEVA
y
2008
Rogez [62]
Joint centers
10
2/3
P
HUMANEVA
?
2008
Urtasun [84]
Joint centers
N/A
3
P
HUMANEVA
??
2008
Vondrak [88]
Ellipsoids + prisms
13
3
T
HUMANEVA
??
? - Mean squared distance in 2D between the set of M = 15 (or fewer) virtual markers corresponding to the joint
centers and limb ends. Measured in pixels (pix).
D (x ; ^x ) =
1
M
P
M
i =1
k m i (x )  m i ( ^x ) k, where m i (x ) 2 R2 is the location of 2D marker i with respect to pose x .
?? - Mean squared distance in 3D between the set of M = 15 virtual markers corresponding to the joint
centers and limb ends. Measured in millimeters (mm).
D (x ; ^x ) =
1
M
P
M
i =1
k m i (x )  m i ( ^x ) k, where m i (x ) 2 R3 is the location of 3D marker i with respect to pose x .
y - Root mean square (RMS) error in joint angle. Measured in degrees (deg).
D (; ^ ) =
1
N
P
N
i =1 j( i  ^ i )mod 
180  j, where  2 RN is the pose in terms of joint angles.
z - Normalized error in joint angle. Measured as a fraction from 0 to 1.
D (; ^ ) = P
N
i =1 1  cos ( i  ^ i ), where  2 RN is the pose in terms of joint angles.
 =  
- Pixel overlap / Pixel overlap based threshold resulting in binary 0/1 detection measure.
\ - Mean distance from 4 endpoints of quadrangular shape representing the limb.
2 Error units were in fractions of the subject’s height.
3 While only qualitative analysis of the overall tracking performance was presented, a quantitative analysis of the number of local minima
in the posterior was performed.
4 Additional per-limb weighting was applied to downweight the error proportionally with the size of the limb.
4

Table 2: Comparison of HUMANEVA to other datasets available and employed by the community.
HUMANEVA
Wang et al.
INRIA Perception [35]
CMU MoCap
CMU MoBo
Datasets
[89]
Multi-Cam Dataset
Dataset [15]
Dataset [27]
# of Subjects
4
3
Unknown
> 100
25
# of Frames

80; 000

450
Unknown
Unknown

200; 000
# of Sequences
56
4
13
2,605
100
Video Data
# of Cameras
4/7
1
8/34
1
6
Calib. Available
Yes
No
Yes
No
Yes
Dataset Content
Motion
Walk
Walk
Dance
Many
Walk
Jog
Dance
Exercise
Throw/Catch
Jumping Jacks
Gesture
Box
Combo
Appearance
Natural
Natural /
Natural /
MoCap Suit
Natural
MoCap Suit
MoCap Suit
Ground Truth
Content
3D
2D
None
3D
2D
Source
MoCap
MoCap /
None
MoCap
Manual Label [92]
Manual Label
to project the 3D models into the images is not. Nevertheless, the video data has proved useful for the analysis of
discriminative methods that do not estimate 3D body location e.g. [48]. In addition, the subjects are dressed in tight
tting motion capture suits and hence lack the realistic clo thing variations exhibited in less controlled environments.
The CMU Motion of Body (MoBo) Database [27], initially developed for gait analysis, has also proved useful in
analyzing the performance of articulated tracking algorithms [20, 92]. While the initial dataset, which contains an
extensive collection of walking motions, did not contain joint-level ground truth information, manually labeled data
has been made available6 by Zhang et al.
A more direct comparison of HUMANEVA to other datasets that are available to the community is given in Table 2.
3
HUMANEVA Datasets
To simultaneously capture video and motion information, our subjects wore natural clothing (as opposed to tight-
tting motion capture suits typically used for pure motion c apture sessions [15]) on which reective markers were
attached using invisible adhesive tape.7 Our motivation was to obtain natural looking image data th at contained all
the complexity posed by moving clothing. One negative outcome of this is that the markers tend to move more than
they would with a tight-tting motion capture suit. As a resu lt, our ground truth motion capture data may not always
be as accurate as that obtained by more traditional methods; we felt that the trade-off of accuracy for realism here
was acceptable. We have applied minimal post-processing to the motion capture data, steering away from the use of
complex software packages (e.g. Motion Builder) that may introduce biases or alter the motion data in the process.
As a result, motion capture data for some frames in some sequences are missing markers or are inaccurate. We made
an effort to detect such cases and exclude them from the quantitative comparison. Note that the presence of markers
on the body may also alter the natural appearance of the body. Given that the marker locations are known, it would
6http://www.cs.cmu.edu/
 zhangjy/
7Participation in the collection process was voluntary and each subject was required to read, understand, and sign an Institutional Review Board
(IRB) approved consent form for collection and distribution of data. A copy of the consent form for the Video and Motion C apture Project
is available by writing to the authors. Subjects were informed that the data, including video images, would be made available to the research
community and could appear in scientic publications.
5

be possible to provide a pixel mask in each image covering the marker locations; these pixels could then be excluded
from further analysis. We felt this was unnecessary since the markers are often barely noticeable at video resolution
and hence will likely have an insignicant impact on the perf ormance of image-based tracking algorithms.
We have developed two datasets that we call HUMANEVA-I and HUMANEVA-II. HUMANEVA-I was captured
earlier and is the larger of the two sets. HUMANEVA-II was captured using a more sophisticated hardware system that
allowed better quality motion capture data and hardware synchronization. The differences between these two datasets
are outlined in the Figure 1.
Since all the data was captured in a laboratory setting, the sequences do not contain any external occlusions or
signicant clutter, but do exhibit the challenges imposed b y strong illumination (e.g. strong shadows that tend to
confuse background subtraction); grayscale cameras used in the HUMANEVA-I dataset present additional challenges
when it comes to background subtraction and image features. Even at 60 Hz the images still exhibit a fair amount of
motion blur.
The split of the training and test data was specically desig ned to emphasize the ability of the pose and motion
estimation approaches to generalize to novel subjects and unobserved motions. To this end, one subject and one
motion for all subjects were withheld from the training and validation dataset for which ground truth is given out. We
believe the proposed datasets exhibit a moderately complex and varied set of motions under realistic indoor imaging
conditions that are applicable to most pose and motion estimation techniques proposed to date.
3.1
HumanEva-I
HUMANEVA-I contains data from 4 subjects performing a set of 6 predened actions in three repetitions (twice
with video and motion capture, and once with motion capture alone). A short description of the actions is provided in
Figure 1. Example images of a subject walking are shown in Figure 2 where data from 7 synchronized video cameras
is illustrated with an overlay of ground truth body pose.
3.1.1
Hardware
Ground truth motion of the body was captured using a commercial motion capture (MoCap) system from Vicon-
Peak8. The system uses reective markers and six 1M-pixel cameras to recover the 3D position of the markers and
thereby estimate the 3D articulated pose of the body.
Video data was captured using two commercial video capture systems. One from Spica Technology Corporation9
and one from IO Industries10. The Spica system captured video using four Pulnix 11 TM6710 grayscale cameras
(grayscale, progressive scan, 644x488 resolution, frame rate of up to 120 Hz). The IO Industries system used three
UniQ 12 UC685CL 10-bit color cameras with 659x494 resolution and a frame rate of up to 110 Hz. The raw frames
were re-scaled from 659x494 to 640x480 by IO Industries software. To achieve better image quality under natural
indoor lighting conditions both video systems were set up to capture at 60 Hz. The rough relative placement of
cameras is illustrated in Figure 1 (left).
The motion capture system and video capture systems were not synchronized in hardware, and hence a software
synchronization was employed. The synchronization and calibration procedures are described in Sections 3.3 and 3.4
respectively.
3.2
HumanEva-II
HUMANEVA-II contains only 2 subjects (both also appear in the HUMANEVA-I dataset) performing an extended
sequence of actions that we call Combo. In this sequence a subject starts by walking along an elliptical path, then
continues on to jog in the same direction and concludes with the subject alternatively balancing on each of the two feet
roughly in the center of the viewing volume. Unlike HUMANEVA-I, this later dataset contains a relatively small test
8http://www.vicon.com/
9http://www.spicatek.com/
10http://www.ioindustries.com/
11http://www.pulnix.com/
12http://www.uniqvision.com/
6

HUMANEVA-I
HUMANEVA-II
MoCap
Hardware System
Manufacturer
ViconPeak
ViconPeak
Number of cameras
6
12
Camera resolution
1M-pixel
MX13 1.3M-pixel
Frame rate
120 Hz
120 Hz
Video Capture System
Color Cameras
Number of cameras
3
4
Frame grabber
IO Industries
ViconPeak
Camera model
UniQ UC685CL
Basler A602fc
Sensor
Progressive Scan
Progressive Scan
Camera resolution
659 x 494 pixels
656 x 490 pixels
Frame rate
60 Hz
60 Hz
Grayscale Cameras
Number of cameras
4
Frame grabber
Spica Tech
Camera model
Pulnix TM6710
Sensor
Progressive Scan
Camera resolution
644 x 448 pixels
Frame rate
60 Hz
Synchronization
Software
Hardware
Data
Actions
(1) Walking, (2) Jogging, (3) Gesturing
Combo
(4) Throwing and Catching a ball,
(5) Boxing, (6) Combo
Number of subjects
4
2
Number of frames
Training (synchronized)
6,800 frames
Training (MoCap only)
37,000 frames
Validation
6,800 frames
Testing
24,000 frames
2,460 frames
Capture Space Layout
BW3
Control Station
Capture Space
C2
C3
C1
BW4
BW1
BW2
3 m
2 m
C1
Control Station
Capture Space
C4
C3
C2
3 m
2 m
Figure 1: HUMANEVA Datasets . The table illustrates the hardware system and conguratio n used to capture the two
datasets, HUMANEVA-I and HUMANEVA-II. The main difference between the hardware systems lies in hardware
synchronization employed in HUMANEVA-II. The contents of the two datasets in terms subjects, motion and amount
of data are also noted. The bird’s eye view sketch of the capture conguration is also shown with rough dimensions
of the capture space and placement of video and motion capture cameras. The color video cameras (C) are designated
by RGB stripped pattern, grayscale video cameras (BW) by the empty camera icon and motion capture cameras are
denoted by gray circles.
7

C1
C2
C3
BW1
BW2
BW3
BW4
Figure 2: Example data from the
HUMANEVA-I database.
Example images of walking subject (S1) from 7 syn-
chronized video cameras (three colored and four grayscale) are shown with overlaid synchronized motion capture
data.
set of synchronized frames ( 2; 500). The HUMANEVA-I training and validation data is intended to be shared across
the two datasets with test results primarily being reported on HUMANEVA-II.
3.2.1
Hardware
As with HUMANEVA-I, the ground truth motion capture data was acquired using a system from ViconPeak. How-
ever, here we used a more recent Vicon MX system with twelve 1.3M-pixel cameras. This newer system produced
more accurate motion capture data.
Video data was captured using a 4-camera reference system provided by ViconPeak which allowed for frame-
accurate synchronization (using the Vicon MX Control
module) of the video and motion capture data. Video was
8

C1
C2
C3
C4
Figure 3: Example data from the
HUMANEVA-II database. Example images of subject (S4) from 4 synchronized
color video cameras performing a combo motion (that includes jogging as shown).
captured using four Basler13 A602fc progressive scan cameras with 656x490 resolution operated at 60 Hz. The rough
relative placement of cameras is illustrated in Figure 1 (right). A calibration procedure to align the Vicon and Basler
coordinate systems is discussed in the next section.
3.3
Calibration
The motion capture system was calibrated using Vicon’s proprietary software and protocol. Calibration of the
intrinsic parameters for the video capture systems was done using a standard checker-board calibration grid and the
Camera Calibration Toolbox for Matlab [10]. Focal length (Fc 2 R2), principle point (Cc 2 R2) and radial distortion
coefcients ( K c 2 R5) were estimated for each camera c 2 C. We assume square pixels and let the skew  c = 0 for
all cameras c 2 C.
The extrinsic parameters corresponding to the rotation, Rc 2 SO(3), and translation, Tc 2 R3, of the camera
with respect to the global (shared) coordinate frame were solved for using a semi-automated procedure to align the
global coordinate axis of each video camera with the global coordinate axis of the Vicon motion capture system. A
single moving marker was captured by the video cameras and the motion capture system for a number of synchronized
frames (> 1000). The resulting 3D tracked position of the marker   (3 D )
t
, t 2 f 1 : : : T (3 D )g was recovered using the
Vicon software. The 2D position of the marker in the video,   (2 D )
t
, t 2 f 1 : : : T (2 D )g, was recovered using a Hough
circle transform [29] that was manually initialized in the  rst frame and subsequently tracked. The projection of the
13http://www.baslerweb.com/
9

3D marker position f (  (3 D )
t
; Rc; Tc) onto the image was then optimized directly for each camera by minimizing
min
R c ;T c ;A c ;B c
T (2 D )
X
t =1
 (t; Ac; B c)k  (2 D )
t
  f (  (3 D )
tA c + B c ; R c; Tc)k2
(1)
for the rotation, Rc, and translation, Tc. Note that the video cameras were calibrated with respect to the calibration
parameters of the Vicon system, as opposed to from the images directly.
In the HUMANEVA-I dataset, the video and motion capture systems were not temporally synchronized in hardware,
hence we also solved for the relative temporal scaling, Ac 2 R, between the video and Vicon cameras, and the temporal
offset Bc 2 R. In doing so we assumed that the temporal scaling was constant over the length of a capture sequence14
(i.e. no temporal drift). The 3D position f (  (3 D )
tA c + B c ; Rc; Tc) was linearly interpolated to cope with non-integer indices
tA c + Bc. Finally, in Eq. (1),  (t; Ac; Bc) is dened as:
 (t; Ac; Bc) =
8
<
:
0
if
tA c + Bc > T (3 D )
0
if
tA c + Bc < 1
1
otherwise.
(2)
The calibration accuracy of the video cameras appears most accurate in the center of the viewing volume (close to the
world origin).
For the HUMANEVA-II data, frame-accurate synchronization was achieved in hardware and we used xed values
Ac = 2 and Bc = 0 for the temporal scaling and offset.
3.4
Synchronization
While the extrinsic calibration parameters and temporal scaling, Ac, can be estimated once per camera (the Vicon
system was only re-calibrated when cameras moved15), without hardware synchronization, the temporal offset Bc was
different for every sequence captured. To temporally synchronize the motion capture and the video in software, for
HUMANEVA-I we manually labeled visible markers on the body for a small sub-set of images (6 images were used
with several marker positions labeled per frame). These labeled frames were subsequently used in the optimization
procedure above but with xed values for Rc, Tc, and Ac to recover a least squares estimate of the temporal offset Bc
for every sequence captured.
4
Evaluation Measures
Various evaluation measures have been proposed for human motion tracking and pose estimation. For example,
a number of papers have suggested using joint-angle difference as the error measure (see Table 1). This measure,
however, assumes a particular parameterization of the human body and cannot be used to compare methods where the
body models have different degrees of freedom or have different parameterizations of the joint angles. For this dataset
we introduce a more widely applicable error measure based on a sparse set of virtual markers that correspond to the
locations of joints and limb endpoints. This error measure was rst introduced for 3D pose estimation and tracking in
[74] and later extended in [7]. It has since been also used for 3D tracking in [41] and for 2D pose estimation evaluation
in [36, 75].
Let x represent the pose of the body. We dene M
= 15 virtual markers as f mi(x)g, i = 1 : : : M , where
mi(x) 2 R3 (or mi(x) 2 R2 if a 2D body model is used) is a function of the body pose that returns the position of the
i’th marker in the world (or image respectively). Notice that dening functions mi(x) for any standard representation
of the body pose x is trivial. The error between the estimated pose ^x and the ground truth pose x is expressed as the
average Euclidean distance between individual virtual markers:
D(x; ^x) =
1
M
M
X
i=1
jjmi(x)   mi(^x)jj:
(3)
14In practice Ac 
2 since the frame rate of motion capture system was roughly 120 Hz and video system is 60 Hz.
15Calibration of the Vicon motion capture system changes the global coordinate frame and hence requires re-calibration of extrinsic parameters
of the video cameras as well.
10

To ensure that we can compare algorithms that use different numbers of parts, we add a binary selection variable
per-marker ^ =
f ^ 1; ^ 2; :::; ^ M g and obtain the nal error function
D(x; ^x; ^) =
1
P M
j =1 ^ j
M
X
i=1
^ ijjmi(x)   mi(^x)jj;
(4)
where ^ i = 1 if the algorithm is able to recover marker i, and 0 otherwise.
For the sequence of T frames we compute the average performance using the following:
 seq = 1
T
T
X
t =1
D(xt ; ^xt ; ^
t ):
(5)
Since many tracking algorithms are stochastic in nature, an average error and the standard deviation computed over a
number of runs is most useful. As a convention from previous methods [7, 36, 74, 75] that have already used this error
measure, we compute the 3D error in millimeters (mm) and the 2D error directly in the image in pixels (pix).
The error measures formulated above are appropriate for measuring the performance of approaches that are able
to recover the full 3D articulated pose of the person in space or the 2D articulated pose of the person in an image.
Some approaches, however, are inherently developed to recover the pose but not the global position of the body (most
discriminative approaches fall into this category, e.g. [2, 48, 76]). To make the above error measures appropriate for
this class of approaches we employ a relative variant
~D(x; ^x) =
1
M
M
X
i=1
jj ~mi(x)  ~mi(^x)jj;
(6)
with ~mi(x) = mi(x)   m0(x), where mi(x) is dened as before and m0(x) is the position of the marker correspond-
ing to the origin of the root segment. The rest of the equations can also be modied accordingly. It is worth noting
that this measure assumes that the orientation of the body relative to the camera is recovered; this is typical of most
discriminative methods.
Note that the error measures assume that an algorithm returns a unique body pose estimate rather than a distribution
over poses. For algorithms that model the posterior distribution over poses as uni-modal, the mean pose is likely to give
a good estimate of x. Most recent methods, however, model multi-modal posterior distributions implicitly or explicitly.
Here the maximum-a-posteriori estimate may be a more appropriate choice for x. This is discussed in greater detail in
[7]. Alternative error measures that compute lower-bounds for sample- or kernel-based representations of the posterior
are discussed in [7].
5
Baseline Algorithm
In addition to the datasets and quantitative evaluation measures, we provide a baseline algorithm16 against which
future advances can be measured. While no standard algori thm exists in the community, we implemented a fairly
common Bayesian ltering method based on the methods of Deut scher and Reid [17] and Sidenbladh et al. [71].
Several variations on the base algorithm are explored with the goal of giving some insight into the important design
choices for human trackers. Quantitative results are presented in the following section.
5.1
Bayesian Filtering Formulation
We pose the tracking problem in a standard way as one of estimating the posterior probability distribution p(xt jy1:t )
for the state xt of the human body at time t given a sequence of image observations y1:t 
(y1; : : : ; yt ). Assuming a
rst-order Markov process
p(xt jx1:t   1) = p(xt jxt   1)
16The implementation is available for download from http://vision.cs.brown.edu/humaneva/baseline.html
11

with a sensor Markov assumption
p(yt jx1:t ; y1:t   1) = p(yt jxt );
a recursive formula for the posterior can be derived [3, 19]:
p(xt jy1:t ) / p(yt jxt )
Z
p(xt jxt   1)p(xt   1jy1:t   1)dxt   1:
(7)
where the integral in Eq. 7 computes the prediction using the previous posterior and the temporal diffusion model
p(xt jxt   1). The prediction is weighted by the likelihood p(yt jxt ) of the new image observation conditioned on the
pose estimate.
5.1.1
Optimization
Non-parametric approximate methods represent posterior distributions by a set of N random samples or particles
with associated normalized weights that are propagated over time using the temporal model and assigned new weights
according to the likelihood function. This is the basis of the Sequential Importance Resampling (SIR) algorithm, or
Condensation [3, 31]. A variation of SIR is the Annealed Particle Filter (APF) introduced for human tracking by
Deutscher and Reid [17]. An APF iterates these steps multiple times at each time instant in order to better localize the
modes of the posterior distribution, and relies on simulated annealing to avoid local optima.
We briey summarize our implementation of the Annealed Part icle Filter algorithm used here since this forms the
core of our baseline algorithm in the experiments that follow. The Sequential Importance Resampling algorithm is
also tested in the following section but is not described in detail as it is similar to APF.
At each time instant the APF algorithm proceeds in a set of la yers, from layer M down to layer 1, that update
the probability density over the state parameters. The state density at layer m + 1 is represented using a set of N
particles with associated normalized weights St;m +1  f x(i)
t;m +1 ;  (i)
t;m +1 gN
i=1 . For the prediction step at layer m, a
Gaussian diffusion model is implemented (section 5.1.4). Specically, hypotheses are drawn with replacement using
Monte Carlo sampling from the state probability density at the previous layer m + 1 using
f x(i)
t;m gN
i=1 
N
X
j =1
 (j )
t;m +1 N (x(j )
t;m +1 ;  M   m ) :
(8)
The sampling covariance matrix  controls the breadth of the search at each layer with a large  spreading sampled
particles more widely. From layer to layer we scale  by a parameter  . This parameter is used to gradually reduce
the diffusion covariance matrix  at lower layers in order to drive the particles towards the modes of the posterior
distribution. Typically  is set to 0:5.
Sampled poses that exceed the joint angle limits of the trained action model or result in inter-penetration of limbs
are rejected and not re-sampled within a layer. The remaining particles are assigned new normalized weights based on
an annealed version of the likelihood function (section 5 .1.3)
 (i)
t;m =
p(yt jx(i)
t;m ) m
P N
j =1 p(yt jx(j )
t;m ) m ; i 2 f 1; : : : ; N g ;
(9)
where  m is a temperature parameter optimized so that approximately half the particles get selected for propaga-
tion/diffusion to the next layer by the Monte-Carlo sampler (Eq. 8). The resulting particle set St;m  f x(i)
t;m ;  (i)
t;m gN
i=1
is then used to compute layer m   1 by re-applying Eqns. (8,9). In tracking, the top layer is initialized with the particle
set of the bottom layer at the previous time instant: St;M +1 = St   1;1.
The expected as well as the maximum a posteriori poses at frame t can be computed from the particle set St; 1 at
the bottom layer using:
^xt =
N
X
i=1
 (i)
t; 1x(i)
t; 1
(10)
^xMAP
t
= x(j )
t; 1 ;
 (j )
t; 1 = max
i
( (i)
t; 1) :
(11)
12

(a)
(b)
(c)
(d)
Figure 4: (a) Input image. (b) Body Model. The body is represented as a kinematic tree with 15 body parts. The red
spheres represent the joint locations where virtual markers are placed for computing 3D error: pelvis joint, hips, knees
and ankles, shoulders, elbows and wrists, neck and the top of the head. (c) Smoothed gradient edge map M e
t , with
values ranging from 0 (pure black) to 1 (pure white). Sparse points shown in red f  e
x t (j )g along the edges of the body
model are matched against the