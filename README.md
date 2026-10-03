# Computer Vision in Python

**Matej Gaberc · Machine Perception / Umetno zaznavanje**

Six Python assignments covering image processing, feature extraction, geometric reconstruction and dimensionality reduction. Each section below gives a quick overview, working solution results, and an interactive showcase.

[Assignment 1](#assignment-1) · [Assignment 2](#assignment-2) · [Assignment 3](#assignment-3) · [Assignment 4](#assignment-4) · [Assignment 5](#assignment-5) · [Assignment 6](#assignment-6)

The original Python submissions and PDFs are in the six assignment folders. Result figures were regenerated from the coursework using the runnable adaptations in [showcase/](showcase/README.md); they are not historical screenshots. [Implementation notes and corrections](PROVENANCE.md).

## Assignment 1

### Image processing, histograms & morphology

**Quick summary.** Images become arrays: manipulate pixels, measure intensity distributions, and separate foreground regions.

- **Basic image processing:** Load RGB images, compute grayscale from channel means, inspect a green-channel crop, invert a rectangular patch and reduce the intensity range. Array slicing and broadcasting let the same operation act on thousands of pixels.
- **Thresholding & histograms:** Implement threshold masks in two ways and build a normalized histogram with configurable bins. Compare exposure distributions, then select a foreground threshold with Otsu’s between-class criterion.
- **Morphology & connected components:** Implement erosion and dilation experiments, clean a bird mask, apply it to RGB channels, invert an eagle mask and filter coin-image connected components by area. Morphology changes local shape; component analysis selects whole regions.

[Original Python](assignment1/assigment1.py) · [Assignment PDF](assignment1/instructions.pdf) · [Detailed walkthrough](assignment1/README.md)

### Working solution results

**1.1 Basic image processing**

![1.1 Basic image processing](showcase/site/assets/01-pixels.png)

**1.2 Thresholding & histograms**

![1.2 Thresholding & histograms](showcase/site/assets/01-histograms.png)

**1.3 Morphology & connected components**

![1.3 Morphology & connected components](showcase/site/assets/01-morphology.png)

**Bird segmentation & foreground masking**

![Bird segmentation & foreground masking](showcase/site/assets/01-processing.png)

### Interactive showcase

**[Open Assignment 1 →](https://gabercmatej.github.io/computer-vision-python/#study-1)**

Compare the full segmentation pipeline with its extracted foreground. The task gallery also lets you browse every result shown above.

---

## Assignment 2

### Convolution, filtering & image retrieval

**Quick summary.** Build local signal operators, compare noise-removal filters, and search an image collection with compact color descriptions.

- **1D convolution:** Write convolution and boundary handling, generate normalized Gaussian kernels for several scales, and explore how composing kernels changes a signal. A kernel slides over neighboring samples and forms a weighted sum.
- **Image filtering:** Apply separable Gaussian filtering to noisy images, sharpen with a custom kernel, compare 1D median windows and implement a 2D median filter. Median filtering rejects isolated impulses while Gaussian averaging smooths both noise and detail.
- **Global descriptors & retrieval:** Build an 8 × 8 × 8 RGB histogram, implement L2, chi-square, intersection and Hellinger distances, rank a 120-image collection, inspect sorted distances and weight frequent colors less strongly.

[Original Python](assignment2/assigment2.py) · [Assignment PDF](assignment2/assignment2_instructions.pdf) · [Detailed walkthrough](assignment2/README.md)

### Working solution results

**2.1 1D convolution**

![2.1 1D convolution](showcase/site/assets/02-convolution.png)

**2.2 Image filtering**

![2.2 Image filtering](showcase/site/assets/02-filtering.png)

**2.3 Global descriptors & retrieval**

![2.3 Global descriptors & retrieval](showcase/site/assets/02-retrieval.png)

**Sharpening comparison**

![Sharpening comparison](showcase/site/assets/02-sharpening.png)

**Four distance functions & frequency weighting**

![Four distance functions & frequency weighting](showcase/site/assets/02-distances.png)

### Interactive showcase

**[Open Assignment 2 →](https://gabercmatej.github.io/computer-vision-python/#study-2)**

Switch between noise removal and image retrieval. The task gallery also lets you browse every result shown above.

---

## Assignment 3

### Derivatives, edges & the Hough transform

**Quick summary.** Progress from local intensity changes to thin edges and geometric line or circle hypotheses.

- **Image derivatives:** Construct Gaussian derivative filters, inspect impulse responses, calculate first and second image derivatives, and derive gradient magnitude and orientation. Also implement an 8 × 8 spatial grid with 8 orientation bins: a 512-dimensional gradient descriptor.
- **Edge detection:** Threshold gradient magnitude, suppress responses across the gradient direction, then use connected-component hysteresis to retain weak edges linked to strong ones. Each step removes a different kind of ambiguity.
- **Hough voting:** Implement line-parameter accumulators, threshold peaks, suppress neighboring maxima and draw strong line hypotheses on images. Extend voting to circle centers when the radius is known.

[Original Python](assignment3/assigment3.py) · [Assignment PDF](assignment3/assignment3_instructions.pdf) · [Detailed walkthrough](assignment3/README.md)

### Working solution results

**3.1 Image derivatives**

![3.1 Image derivatives](showcase/site/assets/03-derivatives.png)

**3.2 Edge detection**

![3.2 Edge detection](showcase/site/assets/03-thinning.png)

**3.3 Hough voting**

![3.3 Hough voting](showcase/site/assets/03-edges.png)

**Known-radius circle detection**

![Known-radius circle detection](showcase/site/assets/03-circles.png)

### Interactive showcase

**[Open Assignment 3 →](https://gabercmatej.github.io/computer-vision-python/#study-3)**

Move the edge threshold slider across four computed settings. The task gallery also lets you browse every result shown above.

---

## Assignment 4

### Feature detection, matching & homography

**Quick summary.** Find distinctive points, match local descriptions across views, and estimate an image alignment despite outliers.

- **Hessian & Harris detectors:** Calculate second-derivative and structure-tensor responses, select strong locations and compare detector behavior at different scales. Hessian responses emphasize blob-like structure; Harris responses emphasize variation in two directions.
- **Local descriptor matching:** Compare supplied local descriptors using Hellinger distance, first with one-way nearest neighbors and then with a mutual-match check. This removes pairs that do not agree in both directions. The source also contains video stabilization experiments.
- **Homography & RANSAC:** Estimate a projective transform with DLT, align supplied point pairs, explore line-fitting RANSAC, reject feature-match outliers, measure reprojection error and implement inverse nearest-neighbor warping.

[Original Python](assignment4/assigment4.py) · [Assignment PDF](assignment4/assignment4_instructions.pdf) · [Detailed walkthrough](assignment4/README.md)

### Working solution results

**4.1 Hessian & Harris detectors**

![4.1 Hessian & Harris detectors](showcase/site/assets/04-detectors.png)

**4.2 Local descriptor matching**

![4.2 Local descriptor matching](showcase/site/assets/04-matches.png)

**4.3 Homography & RANSAC**

![4.3 Homography & RANSAC](showcase/site/assets/04-homography.png)

### Interactive showcase

**[Open Assignment 4 →](https://gabercmatej.github.io/computer-vision-python/#study-4)**

Drag the alignment comparison slider and inspect matched points. The task gallery also lets you browse every result shown above.

---

## Assignment 5

### Disparity, epipolar geometry & triangulation

**Quick summary.** Use the relationship between two views to estimate pixel displacement and recover sparse 3D structure.

- **Disparity from stereo images:** Explore the inverse relationship between depth and disparity, implement normalized cross-correlation patch matching, and experiment with consistency filtering, median smoothing and disparity-based warping.
- **Fundamental matrix:** Build the normalized eight-point system, solve it with SVD, enforce rank two, draw epipolar lines and calculate symmetric point-to-line distances. A point in one image restricts its match to a line in the other.
- **Linear triangulation:** Stack camera projection constraints and solve for each homogeneous 3D point with SVD. Visualize the recovered house and experiment with fundamental-matrix RANSAC before triangulation.

[Original Python](assignment5/assigment5.py) · [Assignment PDF](assignment5/assignment5_instructions.pdf) · [Detailed walkthrough](assignment5/README.md)

### Working solution results

**5.1 Disparity from stereo images**

![5.1 Disparity from stereo images](showcase/site/assets/05-disparity.png)

**5.2 Fundamental matrix**

![5.2 Fundamental matrix](showcase/site/assets/05-epipolar.png)

**5.3 Linear triangulation**

![5.3 Linear triangulation](showcase/site/assets/05-stereo.png)

### Interactive showcase

**[Open Assignment 5 →](https://gabercmatej.github.io/computer-vision-python/#study-5)**

Rotate the reconstructed 3D house and inspect the two input views. The task gallery also lets you browse every result shown above.

---

## Assignment 6

### PCA, dimensionality reduction & eigenfaces

**Quick summary.** Learn directions of variation, compress observations, and reconstruct faces from a small set of coefficients.

- **Direct PCA:** Compute the mean, covariance and eigenvectors; draw principal directions; project points onto one component; compare nearest neighbors before and after projection; select components retaining 80% of variance in 50D data.
- **Dual PCA:** Decompose the sample-space covariance matrix, map its eigenvectors back to feature space, compare nonzero eigenvalues with direct PCA and reconstruct the original points. This is useful when there are many more pixels than training images.
- **Face-space decomposition:** Load faces as columns, compute eigenfaces, reconstruct images, compare pixel edits with coefficient edits, sweep component counts, vary the first two coefficients and project an elephant image into face space. A webcam-recognition example is also preserved.

[Original Python](assignment6/exercise6.py) · [Assignment PDF](assignment6/assignment6_instructions.pdf) · [Detailed walkthrough](assignment6/README.md)

### Working solution results

**6.1 Direct PCA**

![6.1 Direct PCA](showcase/site/assets/06-point-pca.png)

**6.2 Dual PCA**

![6.2 Dual PCA](showcase/site/assets/06-dual-pca.png)

**6.3 Face-space decomposition**

![6.3 Face-space decomposition](showcase/site/assets/06-pca.png)

**Learned eigenfaces**

![Learned eigenfaces](showcase/site/assets/06-eigenfaces.png)

### Interactive showcase

**[Open Assignment 6 →](https://gabercmatej.github.io/computer-vision-python/#study-6)**

Change the number of PCA components and compare face reconstructions. The task gallery also lets you browse every result shown above.

---

## Run the code

Original scripts, PDFs and supplied data live in `assignment1/` through `assignment6/`. The interactive website, reusable adaptations, figure generators and tests are grouped under **[showcase/](showcase/README.md)**.

[Setup and reproduction](showcase/README.md) · [Provenance](PROVENANCE.md) · [Corrections](CHANGELOG.md) · [Course-material credits](ATTRIBUTION.md)
