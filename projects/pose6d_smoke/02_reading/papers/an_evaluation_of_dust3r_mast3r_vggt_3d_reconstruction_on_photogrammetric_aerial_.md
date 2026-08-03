# An Evaluation of DUSt3R/MASt3R/VGGT 3D Reconstruction on Photogrammetric Aerial Blocks

> 2025 · id: arxiv:2507.14798 · arXiv: 2507.14798 · pdf: https://arxiv.org/pdf/2507.14798 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
State-of-the-art 3D computer vision algorithms continue to improve on sparse, unordered image sets. 
Recently developed foundational models for 3D reconstruction, such as Dense and Unconstrained Stereo 
3D Reconstruction (DUSt3R), Matching and Stereo 3D Reconstruction (MASt3R), and Visual Geometry 
Grounded Transformer (VGGT), have attracted considerable attention due to their ability to handle very 
sparse image overlaps, as well as their generalization capability. In light of this contribution, evaluating 
DUSt3R/MASt3R/VGGT on typical aerial images is important, as these models may hold the potential to 
handle extremely low image overlaps, stereo occlusions, and textureless regions. For highly redundant 
collections, they can accelerate 3D reconstruction by using extremely sparsified image sets. Despite being 
tested on various computer vision benchmarks, their potential on photogrammetric aerial blocks remains 
unexplored. We present a comprehensive evaluation of the pre-trained DUSt3R/MASt3R/VGGT models 
on the aerial blocks of the UseGeo dataset for pose estimation and dense 3D reconstruction. The methods 
reconstruct dense point clouds from very sparse inputs (fewer than ten images, resized to a maximum 
dimension of 518 pixels), achieving reasonable accuracy and completeness gains up to 50% over 
COLMAP. VGGT further shows superior computational efficiency, scalability, and more reliable camera 
pose estimation. However, all three show limitations on high-resolution imagery and large image sets, 
with the camera pose estimation reliability significantly declining as the number of images and the 
geometric complexity of the scene increase. These findings indicate that while transformer-based method 
cannot replace traditional SfM and MVS methods entirely, they hold potential as complementary 
approaches, especially in challenging, low-resolution, and extremely sparse scenarios. 

2

## introduction
of 
key 
modules 
in 
traditional 
(COLMAP) 
and 
learning-based 
(DUSt3R/MASt3R/VGGT) 3D reconstruction pipelines. DLT: Direct Linear Transformation. 
 
3.3 Evaluation on Dense Point Clouds Generation 
Accuracy. Accuracy is measured using the quadratic height function in CloudCompare, which computes 
the vertical distance between each estimated point and the corresponding reference surface derived from 
the ground truth point cloud. This method provides a more reliable accuracy assessment by considering 
local surface variations rather than simple point‐to‐point Euclidean distances. The mean accuracy 
represents the average vertical deviation between the reconstructed point cloud and the ground truth 
Traditional Methods 
Feature 
Extraction 
Feature 
Matching 
Geometric 
Verification 
Image 
Registration 
Triangulation Robust 
Estimation 
Dense point 
cloud generation 
COLMAP SIFT (Lowe 
2004) 
Exhaustive 
search 
7‐Point F‐matrix 
(Hartley and 
Zisserman 2003) 
P3P (Gao et al. 
2003) 
Sampling‐
based DLT 
RANSAC 
Patch-based 
stereo 
(Schönberger et 
al. 2016) 
Learning-based Methods 
 
Encoder 
Decoder 
Heads 
Network Loss 
DUSt3R/ MASt3R ViT-Large (Dosovitskiy et al. 
2020) 
ViT-Base 
(Dosovitskiy 
et al. 2020) 
DPT (Ranftl et al. 2021) / 
CatMLP+DPT 
Simple regression loss 
VGGT 
ViT-Large (Dosovitskiy et al. 
2020) 
- 
 Task-specific heads 
Multi-task loss 

11
 
LiDAR data. We follow existing works (Ahmad Fuad et al. 2018; Xu et al. 2023) and use the mean C2C 
distance, 𝜎𝜎MEAN, as shown in Equation (1). 
 
𝜎𝜎MEAN = 𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀൫𝐷𝐷𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝_𝑡𝑡𝑡𝑡_𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙_𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠൯ 
(1) 
Completeness. Completeness is measured by reversing the process: the vertical distance between 
each ground truth point and the corresponding reference surface derived from the estimated point cloud is 
calculated, with an empirical threshold of 1 m applied. Completeness is defined as the ratio of ground 
truth points within this threshold (𝑁𝑁𝑤𝑤𝑤𝑤𝑤𝑤ℎ𝑖𝑖𝑖𝑖) to the total number of ground truth points (𝑁𝑁𝐺𝐺𝐺𝐺), where 𝑁𝑁𝑤𝑤𝑤𝑤𝑤𝑤ℎ𝑖𝑖𝑖𝑖 
is the number of ground truth points within the threshold, and 𝑁𝑁𝐺𝐺𝐺𝐺 is the total number of ground truth 
points. 
 
𝑁𝑁𝑤𝑤𝑤𝑤𝑤𝑤ℎ𝑖𝑖𝑖𝑖= ෍𝛿𝛿൫𝑑𝑑൫𝒑𝒑𝑗𝑗,𝐺𝐺𝐺𝐺, 𝑃𝑃𝐸𝐸൯≤𝜏𝜏൯
𝑁𝑁𝐺𝐺𝑇𝑇
𝑗𝑗=1
     
(2) 
Where 𝑑𝑑൫𝒑𝒑𝑗𝑗,𝐺𝐺𝐺𝐺, 𝑃𝑃𝐸𝐸൯ is the vertical distance from the ground truth point 𝒑𝒑𝑗𝑗,𝐺𝐺𝐺𝐺 to the corresponding 
reference surface derived from the estimated point cloud 𝑃𝑃𝐸𝐸. Here, 𝜏𝜏 is the threshold (e.g., 1 m); δ(·) is an 
indicator function that equals 1 if the condition inside is true, and 0 otherwise. The evaluation employs 
both accuracy and completeness to provide a comprehensive analysis of the results. 
3.4 Evaluation on Camera Poses Estimation 
The pose of each camera is compared against its corresponding ground truth, evaluating both position and 
orientation.   
 
3.4.1 Evaluation of Camera Position/Translation 
The camera position is assessed by calculating the Euclidean distance between the reconstructed position 
and the ground truth position, as shown below: 
 
𝛥𝛥𝛥𝛥= ฮ𝑪𝑪𝒑𝒑𝒑𝒑𝒑𝒑𝒑𝒑−𝑪𝑪𝒈𝒈𝒈𝒈ฮ 
(3) 
where 𝛥𝛥𝛥𝛥 is the camera center difference (in meters), 𝐶𝐶𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝 is the predicted camera center, 𝐶𝐶𝑔𝑔𝑔𝑔 is the 
ground truth camera center, and ∥ · ∥ denotes the Euclidean norm (distance). 
 
3.4.2 Evaluation of Camera Rotation/Orientation. 
Orientation differences are assessed by determining the angle of the rotation required to align the 
reconstructed camera’s orientation with the ground truth (Bianco et al. 2018; Xu et al. 2024). We 
represent orientations with unit quaternions and compute the error from the relative quaternion. The 
relative quaternion is calculated as follows: 
 
𝒒𝒒𝑹𝑹= 𝒒𝒒𝑬𝑬
−𝟏𝟏𝒒𝒒𝑮𝑮𝑮𝑮 
(4) 
 

12
 
Here, 𝐪𝐪𝑹𝑹 represents the quaternion describing the rotational transformation needed to align the estimated 
camera orientation (𝐪𝐪𝑬𝑬) with the ground truth orientation (𝐪𝐪𝑮𝑮𝑮𝑮), where 𝐪𝐪𝑬𝑬
−𝟏𝟏 denotes the inverse of the 
estimated orientation. The orientation error of camera poses is measured in terms of angle difference (𝛼𝛼), 
and can be computed from the scalar part 𝑤𝑤 of the quaternion, as shown in Equation (5).  
 
𝛼𝛼= cos−1(𝒒𝒒𝑅𝑅𝑅𝑅) 
(5) 
 
4 Experiment Results 
First, we assess the reconstructed point clouds, focusing on accuracy and completeness as key metrics, as 
shown in Section 4.1. Next, we compare methods by camera-center differences and camera-angle 
distances, as shown in Section 4.2. The scalability study on 191 images using VGGT and COLMAP 
appears in Section 4.3, and Section 4.4 reports runtime and computational resources. Finally, Section 4.5 
reviews the practical implications of learning-based reconstruction for aerial data.  
We use COLMAPHR for results from high-resolution inputs and COLMAPLR for results from low-
resolution inputs. COLMAP refers to the method family regardless of resolution. All experiments were 
conducted on a system running Ubuntu 22.04.5 LTS, equipped with an AMD Ryzen Threadripper PRO 
5955WX CPU (16 cores, 1.8–4.0 GHz), 512 GB RAM, and an NVIDIA RTX 6000 Ada Generation GPU 
(52 GB VRAM).  
4.1 Accuracy of Dense Point Clouds 
As Figure 3 illustrates, for the single-image case, DUSt3R, MASt3R, and VGGT reconstruct dense urban 
point clouds, whereas COLMAP fails because viewing angles are insufficient for triangulation. However, 
the reconstructed models still have flaws, exhibiting holes around buildings and failures on small towers, 
likely due to limited model understanding of tall structures in top‐down views and insufficient resolution. 
Similarly, when using two images with a large viewpoint difference, COLMAP often fails or produces 
low-quality models with sparse points, achieving an accuracy of up to 2.3 m. In contrast, DUSt3R, 
MASt3R, and VGGT produce reasonable point clouds, with MASt3R and VGGT showing similar 
performance and generally outperforming the others. These methods achieve higher accuracy (up to 0.4 m) 
and greater completeness (an increase of +10%), as shown in Table 3. 
MASt3R and VGGT outperform COLMAP in completeness in 87% of instances, achieving up to an 
additional 19% completeness in most scenarios. This is due to their ability to generate more points 
without geometric constraints, unlike COLMAP, which prioritizes higher accuracy by producing fewer 
points. Learning-based methods such as MASt3R employ a coarse-to-fine, one-versus-all strategy for 
point triangulation, while VGGT directly predicts near-accurate point or depth maps. Both approaches 
lack epipolar constraints and multi-view consistency, which leads to denser and more efficient, but less 
accurate point clouds. This trade-off yields higher completeness but lower accuracy in reconstructions.  
As the number of images increases, COLMAP leverages good viewing angle differences to 
reconstruct a model, with high-resolution input achieving significantly higher accuracy. The qualitative 

13
 
results for Dataset-3 using 38 images are presented in Figure 4. In this case, COLMAPHR achieves an 
accuracy of 0.2 m, corresponding to a 92% reduction in error compared to the other methods, which have 
errors around 2.0 m. One potential factor contributing to COLMAPHR’s superior accuracy is that it 
processes images at higher resolutions, allowing for more precise feature extraction and matching. 
However, when analyzing scenarios using rescaled images with a maximum dimension of 512 pixels, 
COLMAPLR’s accuracy fluctuates substantially, sometimes resulting in errors of 4 m in contrast to 
MASt3R’s 0.4 m, and COLMAPLR suffers from very low completeness due to the limited number of 3D 
points detected. 
Overall, COLMAPHR consistently achieves the highest accuracy when results are available and 
generally maintains acceptable completeness. Although its completeness is sometimes lower than that of 
VGGT, the difference is not substantial. Its performance is stable, especially as the number of images 
increases. However, MASt3R and VGGT demonstrate clear advantages in challenging scenarios with 
very limited images, where COLMAP often fails or cannot be applied. This suggests that, although 
MASt3R and VGGT are not yet a complete replacement for traditional methods in standard SfM and 
MVS pipelines, they can serve as a valuable supplement, particularly for improving completeness in 
sparse or difficult cases.   
The results of the low‐overlap reconstruction experiment using 38 images are presented in Table 4. 
Overall, these findings are consistent with previous observations: COLMAP achieves higher accuracy, 
whereas MASt3R and VGGT demonstrate comparable performance and superior completeness. 
Specifically, COLMAP achieves higher accuracy in 93% of cases, with accuracy up to 80% better than 
that of the others. In contrast, MASt3R and VGGT outperform both COLMAP variants in completeness 
in 80% of cases, with gains of up to +50%. Further, as the overlap decreases, the learning‐based methods 
maintain both accuracy and completeness, exhibiting robustness in extremely low‐overlap scenarios, 
whereas COLMAP

## method
Success Rate at Different Overlap Levels (%) 
70% 
55% 
40% 
25% 
10% 
DUSt3R 
100 (0) 
100 (0) 
100 (0) 
100 (0) 
100 (0) 
MASt3R 
100 (0) 
100 (0) 
100 (0) 
100 (0) 
100 (0) 
VGGT 
100 (10) 
100 (6) 
100 (0) 
100 (0) 
100 (0) 
COLMAPLR 
75 (27) 
84 (11) 
60 (2) 
20 (0) 
13 (0) 
COLMAPHR 
85 (64) 
61 (53) 
85 (35) 
85 (22) 
51 (11) 
4.3 Scalability Evaluation 
All four methods were evaluated on the standard 38-image dataset, but only VGGT and COLMAP can 
process larger image sets. Therefore, we conducted an additional scalability experiment with 191 images. 
Visualization results for Dataset-2 are presented in Figure 6. The VGGT reconstructions exhibit 
pronounced inconsistencies in point cloud alignment, such as overlapping buildings, repeated occurrences 
of the same structures at multiple locations, and road segments that are interpolated in ways inconsistent 
with the actual scene geometry. In comparison, COLMAP generates three separate models, but each 
reconstructed point cloud is internally consistent and does not display significant misalignment. Table 8 
presents the quantitative results for dense point cloud and camera pose accuracy. VGGT demonstrates 
higher point cloud errors, reaching up to 6 m, which represents approximately an 85% increase compared 
to COLMAPHR’s. Additionally, camera pose estimates produced by VGGT may exhibit drift of up to 42 
m. Substantial errors in both point cloud and camera pose estimation mean VGGT cannot yet deliver 
reliable or usable previews for the areas of interest, and it is still not suitable as a standalone solution for 
large-scale aerial photogrammetry, although VGGT demonstrates better scalability than the other end-to-
end approaches. 
4.4 Computation Time 
DUSt3R/MASt3R are significantly faster than COLMAP, and VGGT can be remarkably faster than 

20
 
DUSt3R/MASt3R as well. For instance, in the 38-image case (Table 9), MASt3R requires only 9% of 
COLMAPHR’s processing time, while VGGT operates at just 12% of MASt3R’s processing time, 
making VGGT particularly suitable for compute-constrained environments. The substantial reduction in 
processing time is likely due to VGGT's multi-image training paradigm, which enables the network to 
natively perform multiview triangulation. In contrast, DUSt3R relies on separate pairwise triangulations 
that are later averaged, resulting in less efficient alignment procedures.  
 
Figure 6. Reconstruction models for 191-image experiment on Dataset-2: (a) VGGT, (b) COLMAPHR. 
 
Table 8. Point cloud and camera pose evaluation of VGGT and COLMAPHR on three benchmark datasets. 
For camera poses, the values in parentheses are for inliers (center distance <1 m, angle difference <10°). 
Red cells indicate at least one valid inlier; white cells mean no inliers were found for that setting. 
Dataset

## related_work
Image-based 3D reconstruction has advanced rapidly in photogrammetry and computer vision. In this 
section, we review related work in 3D reconstruction, comparing traditional Structure-from-Motion (SfM) 
and Multi-View Stereo (MVS) with more recent learning-based approaches. We also examine existing 
evaluation studies and highlight their limitations. 
SfM and MVS. Camera orientation and dense image matching have been widely studied, leading to 
the development of various algorithms and open-source tools. SfM (Crandall et al. 2011; Hartley 2003; 
Schonberger and Frahm 2016) processes unordered images to recover camera parameters and produce a 
sparse point cloud. It uses correspondences between overlapping images to compute intrinsic and 
extrinsic parameters (Koutsoudis et al. 2014), followed by bundle adjustment to refine camera poses 
(Snavely et al. 2006). Bundler by Snavely et al. (Snavely et al. 2006) is one of the earliest open-source 
systems for image-based 3D reconstruction and point-cloud generation. It addresses the SfM problem by 
estimating camera parameters. Building on this foundation, later works extended these techniques to 
large-scale scene reconstruction (Agarwal et al. 2011). Further, Patch-based Multi-View Stereo (PMVS), 
introduced by Furukawa and Ponce (Furukawa and Ponce 2010, 20), performs for dense image matching 
to produce detailed reconstructions. More broadly, MVS reconstructs dense point clouds from a set of 
images, and the final 3D model is obtained fusing per-view depth maps into a single coherent 
representation. These tools have been widely adopted by researchers and practitioners (Furukawa et al. 
2015). Numerous frameworks and libraries have since been released, extending these techniques. 
Examples include the Multi-View Environment (MVE) (Furukawa et al. 2015), an end-to-end pipeline for 
image-based geometry reconstruction, and Open Multiple View Geometry (OpenMVG) by Moulon et al. 

5 
 
(Moulon et al. 2017), a library tailored to the multiple-view geometry community. More recently, full 3D 
reconstruction pipelines such as COLMAP and OpenMVS (Cernea 2020) provide comprehensive 
solutions for a broad audience. In parallel, advances in deep learning for computer vision and 
photogrammetry have increased the prominence of learning-based approaches (Hartmann et al. 2017; 
Kerbl et al. 2023; C. Wang et al. 2024), particularly in areas such as self-supervised methods for single-
image depth estimation (Knöbelreiter et al. 2018; Madhuanand et al. 2021). 
Direct RGB‐to‐3D. Unconstrained dense 3D reconstruction from multiple RGB images remains a 
long‐standing research problem in 3D modeling (Charles et al. 2017; Dame et al. 2013; Mildenhall et al. 
2021). In recent years, neural network‐based methods that predict depth from a single image or a very 
small number of images have gained significant attention. These approaches, used not only for matching 
(Ji et al. 2019), address many limitations of two‐view and multi‐view stereo depth estimation. Notably, 
they eliminate the sequential dependency of the SfM pipeline, which tends to accumulate errors and noise 
at each processing stage. Some methods use neural networks to learn robust geometric class-level priors 
or diffusion models (Liu et al. 2023). However, these approaches are primarily designed for object-centric 
reconstruction rather than large-scale scene reconstruction. Another line of research focuses on general 
scene reconstruction by using monocular depth estimation neural networks trained on large datasets. 
These methods can produce pixel‐aligned 3D point clouds (Ranftl et al. 2021; Wiles et al. 2020; Yin et al. 
2021), although depth quality can lack fidelity because of missing scale or out-of-distribution prediction. 
To address this limitation, multi-view neural networks for direct 3D reconstruction have been introduced, 
which enable end-to-end training and resolving scale ambiguity (Ummenhofer et al. 2017). More recently, 
DUSt3R has emerged as a notable advance, eliminating the need for ground truth camera intrinsics as 
input. This approach can directly generate point maps and global camera poses rather than relying on 
depth maps and relative poses. The promising results of DUSt3R and its sibling MASt3R have driven 
further progress, inspiring the development of more sophisticated methods such as VGGT (Visual 
Geometry Grounded Transformer) (J. Wang et al. 2025). VGGT is a feed-forward neural network built on 
a standard large transformer(Vaswani et al. 2017). It removes pairwise point cloud generation and can 
process more than two images simultaneously, enabling direct production of point clouds without post-
processing to fuse pairwise reconstructions. This design can yield more consistent point cloud results.  
As interest grows, new models are appearing rapidly. Fast3R (Yang et al. 2025) extends the DUSt3R 
family to a single forward pass designed for large N inputs, improving throughput from a handful of 
views to hundreds or more. Along the VGGT line, FastVGGT (Shen et al. 2025) and Faster VGGT (C.-S. 
B. Wang et al. 2025) identify global attention as the main bottleneck: the former uses token merging and 
the latter uses optimized block sparse attention to accelerate inference while keeping quality comparable. 
These methods are promising for aerial applications because they are efficient and can scale to hundreds 
or even thousands of images. In quick tests against the learning-based methods used in this paper, the 
newest models showed similar or slightly better point cloud and pose quality. Metrics include point cloud 
accuracy and completeness as well as pose center and orientation errors. These findings do not change our 
conclusions. Given the fast pace of the field and our scope, we proceed without adding these models and 
instead cite them because of their recent release. 

6 
 
Surveys, Reviews, and Evaluation. With the rise of open‐source 3D reconstruction solutions, 
evaluating these pipelines has become common in the research community. Reviews have analyzed 
methods, datasets, scenarios, and photogrammetric metrics (Alidoost and Arefi 2017; Georgopoulos et al. 
2016; Pepe et al. 2022). Moreover, Remondino et al. (Remondino et al. 2017) documented the 
development of diverse MVS algorithms for reconstructing different scenes. Stathopoulou et al. 
(Stathopoulou et al. 2019) examined widely used open-source image-based 3D reconstruction pipelines, 
while Jarahizadeh and Salehi (Jarahizadeh and Salehi 2024) presented a recent evaluation of popular 
photogrammetry software. However, these efforts are limited to traditional MVS solutions. Learning-
based methods have gained attention, and new evaluation practices have appeared because these 
approaches have the potential to surpass traditional methods in multiple domains. Unlike conventional 
techniques, they support end-to-end training, which removes the need for manually designed multi-stage 
processes. Several studies have surveyed key challenges, network architectures, and evaluation 
methodologies in 3D reconstruction (Fahim et al. 2021; Fu et al. 2021). However, their scope is limited to 
single-image 3D object reconstruction. Han et al. (Han et al. 2021) extend the scope by covering both 
single- and multi-image , but they do not include research published after 2019 and thus miss recent 
advances. Additionally, Samavati and Soryani (Samavati and Soryani 2023) take a broader perspective by 
exploring studies where 3D reconstruction serves as a downstream task for various objectives. Their 
survey mentions DUSt3R but does not provide experimental data to support its performance.  
The rapid progress of the field calls for regular reassessment of recent research. Evaluating new 
methods on updated benchmark datasets is essential to keep pace with ongoing advances. 
 
3 Material Preparation and Experiment Setup  
This section presents the benchmark dataset and our data preparation workflow, then outlines the 
evaluated approaches for 3D reconstruction. Finally, we define the metrics used to assess dense point 
clouds and camera poses. 
 
3.1 Dataset Configuration 
We use the UseGeo dataset (Nex et al. 2024), which includes images and LiDAR collected at the same 
time across diverse urban and peri-urban areas. The UseGeo dataset is intended for rigorous benchmarking 
in the context of photogrammetry applications. A total of 829 high‐resolution images were captured at an 
average altitude of 80 m during three flights that cover three distinct areas, which we refer to as Dataset-1, 
Dataset-2, and Dataset-3. Each dataset contains eight flight strips, with typical image overlap of 60–80%. 
LiDAR was acquired simultaneously at about 51 points per square meter, which corresponds to a Ground 
Sample Distance (GSD) of approximately 2 cm. Following image and LiDAR acquisition, the hybrid 
adjustment (Glira et al. 2019) method was employed to jointly refine the o