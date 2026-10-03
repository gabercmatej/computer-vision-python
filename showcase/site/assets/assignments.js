window.ASSIGNMENT_WALKTHROUGHS = [
  {
    "id": 1,
    "title": "Image processing, histograms & morphology",
    "summary": "Images become arrays: manipulate pixels, measure intensity distributions, and separate foreground regions.",
    "code": "assigment1.py",
    "pdf": "instructions.pdf",
    "tasks": [
      {
        "title": "1.1 Basic image processing",
        "description": "Load RGB images, compute grayscale from channel means, inspect a green-channel crop, invert a rectangular patch and reduce the intensity range. Array slicing and broadcasting let the same operation act on thousands of pixels.",
        "image": "01-pixels.png",
        "caption": "The plate uses a fixed [0, 1] display range so the reduced-intensity image actually appears darker."
      },
      {
        "title": "1.2 Thresholding & histograms",
        "description": "Implement threshold masks in two ways and build a normalized histogram with configurable bins. Compare exposure distributions, then select a foreground threshold with Otsu\u2019s between-class criterion.",
        "image": "01-histograms.png",
        "caption": "The original bright/dark files were not supplied; these explicitly labeled variants are derived from the available image. The runnable Otsu version is deterministic; the original submission uses sampling."
      },
      {
        "title": "1.3 Morphology & connected components",
        "description": "Implement erosion and dilation experiments, clean a bird mask, apply it to RGB channels, invert an eagle mask and filter coin-image connected components by area. Morphology changes local shape; component analysis selects whole regions.",
        "image": "01-morphology.png",
        "caption": "This plate compares bird erosion/dilation and coin components. The coin visualization explicitly uses 0.9, correcting the original threshold-variable typo; it omits the exploratory large-kernel cleanup to make the area selection visible."
      },
      {
        "title": "Bird segmentation & foreground masking",
        "description": "",
        "image": "01-processing.png",
        "caption": "Regenerated result from the assignment experiment."
      }
    ]
  },
  {
    "id": 2,
    "title": "Convolution, filtering & image retrieval",
    "summary": "Build local signal operators, compare noise-removal filters, and search an image collection with compact color descriptions.",
    "code": "assigment2.py",
    "pdf": "assignment2_instructions.pdf",
    "tasks": [
      {
        "title": "2.1 1D convolution",
        "description": "Write convolution and boundary handling, generate normalized Gaussian kernels for several scales, and explore how composing kernels changes a signal. A kernel slides over neighboring samples and forms a weighted sum.",
        "image": "02-convolution.png",
        "caption": "The associativity illustration uses full convolution to avoid truncation at intermediate boundaries; the submission also explores padded filtering."
      },
      {
        "title": "2.2 Image filtering",
        "description": "Apply separable Gaussian filtering to noisy images, sharpen with a custom kernel, compare 1D median windows and implement a 2D median filter. Median filtering rejects isolated impulses while Gaussian averaging smooths both noise and detail.",
        "image": "02-filtering.png",
        "caption": "A fixed synthetic salt-and-pepper sample makes this comparison reproducible. The source also contains Gaussian-noise and Lena experiments."
      },
      {
        "title": "2.3 Global descriptors & retrieval",
        "description": "Build an 8 \u00d7 8 \u00d7 8 RGB histogram, implement L2, chi-square, intersection and Hellinger distances, rank a 120-image collection, inspect sorted distances and weight frequent colors less strongly.",
        "image": "02-retrieval.png",
        "caption": "The query is excluded from its own nearest-neighbor results. Histogram similarity describes color, not object identity; no dataset-wide accuracy is claimed."
      },
      {
        "title": "Sharpening comparison",
        "description": "",
        "image": "02-sharpening.png",
        "caption": "Regenerated result from the assignment experiment."
      },
      {
        "title": "Four distance functions & frequency weighting",
        "description": "",
        "image": "02-distances.png",
        "caption": "Regenerated result from the assignment experiment."
      }
    ]
  },
  {
    "id": 3,
    "title": "Derivatives, edges & the Hough transform",
    "summary": "Progress from local intensity changes to thin edges and geometric line or circle hypotheses.",
    "code": "assigment3.py",
    "pdf": "assignment3_instructions.pdf",
    "tasks": [
      {
        "title": "3.1 Image derivatives",
        "description": "Construct Gaussian derivative filters, inspect impulse responses, calculate first and second image derivatives, and derive gradient magnitude and orientation. Also implement an 8 \u00d7 8 spatial grid with 8 orientation bins: a 512-dimensional gradient descriptor.",
        "image": "03-derivatives.png",
        "caption": "The descriptor is implemented, but its requested integration into the retrieval system is not present."
      },
      {
        "title": "3.2 Edge detection",
        "description": "Threshold gradient magnitude, suppress responses across the gradient direction, then use connected-component hysteresis to retain weak edges linked to strong ones. Each step removes a different kind of ambiguity.",
        "image": "03-thinning.png",
        "caption": "The reusable implementation includes corrected suppression and hysteresis. Four threshold settings remain available in the interactive demo."
      },
      {
        "title": "3.3 Hough voting",
        "description": "Implement line-parameter accumulators, threshold peaks, suppress neighboring maxima and draw strong line hypotheses on images. Extend voting to circle centers when the radius is known.",
        "image": "03-edges.png",
        "caption": "The building example displays 12 line hypotheses. Optional gradient-directed voting (3f) and line-length normalization (3h) are blank in the original."
      },
      {
        "title": "Known-radius circle detection",
        "description": "",
        "image": "03-circles.png",
        "caption": "Regenerated result from the assignment experiment."
      }
    ]
  },
  {
    "id": 4,
    "title": "Feature detection, matching & homography",
    "summary": "Find distinctive points, match local descriptions across views, and estimate an image alignment despite outliers.",
    "code": "assigment4.py",
    "pdf": "assignment4_instructions.pdf",
    "tasks": [
      {
        "title": "4.1 Hessian & Harris detectors",
        "description": "Calculate second-derivative and structure-tensor responses, select strong locations and compare detector behavior at different scales. Hessian responses emphasize blob-like structure; Harris responses emphasize variation in two directions.",
        "image": "04-detectors.png",
        "caption": "The plate uses scales 3, 6 and 9 with a normalized 0.4 threshold. Spatial nonmaximum suppression is corrected in the reusable version."
      },
      {
        "title": "4.2 Local descriptor matching",
        "description": "Compare supplied local descriptors using Hellinger distance, first with one-way nearest neighbors and then with a mutual-match check. This removes pairs that do not agree in both directions. The source also contains video stabilization experiments.",
        "image": "04-matches.png",
        "caption": "Descriptors come from the course helper; the matching logic is implemented in the submission. The custom-descriptor section 2c is blank. No video input was supplied, so stabilization is code-only."
      },
      {
        "title": "4.3 Homography & RANSAC",
        "description": "Estimate a projective transform with DLT, align supplied point pairs, explore line-fitting RANSAC, reject feature-match outliers, measure reprojection error and implement inverse nearest-neighbor warping.",
        "image": "04-homography.png",
        "caption": "The displayed result uses normalized DLT and seeded RANSAC with a consensus refit, added during portfolio cleanup. OpenCV renders this warp; the original custom warp remains in the source. Section 3d is blank."
      }
    ]
  },
  {
    "id": 5,
    "title": "Disparity, epipolar geometry & triangulation",
    "summary": "Use the relationship between two views to estimate pixel displacement and recover sparse 3D structure.",
    "code": "assigment5.py",
    "pdf": "assignment5_instructions.pdf",
    "tasks": [
      {
        "title": "5.1 Disparity from stereo images",
        "description": "Explore the inverse relationship between depth and disparity, implement normalized cross-correlation patch matching, and experiment with consistency filtering, median smoothing and disparity-based warping.",
        "image": "05-disparity.png",
        "caption": "The displayed NCC result uses images resized to 120 px, 6 \u00d7 6 patches and a search limit of 24 px. It shows raw left-to-right disparity. The original reverse-search direction makes its consistency-filter attempt unverified."
      },
      {
        "title": "5.2 Fundamental matrix",
        "description": "Build the normalized eight-point system, solve it with SVD, enforce rank two, draw epipolar lines and calculate symmetric point-to-line distances. A point in one image restricts its match to a line in the other.",
        "image": "05-epipolar.png",
        "caption": "Uses supplied house correspondences. Fully automatic fundamental-matrix estimation (2d) and PROSAC (2e) are blank."
      },
      {
        "title": "5.3 Linear triangulation",
        "description": "Stack camera projection constraints and solve for each homogeneous 3D point with SVD. Visualize the recovered house and experiment with fundamental-matrix RANSAC before triangulation.",
        "image": "05-stereo.png",
        "caption": "Both views\u2019 correspondences and camera matrices are supplied. The RANSAC extension also uses supplied matches; this is not a fully automatic reconstruction pipeline."
      }
    ]
  },
  {
    "id": 6,
    "title": "PCA, dimensionality reduction & eigenfaces",
    "summary": "Learn directions of variation, compress observations, and reconstruct faces from a small set of coefficients.",
    "code": "exercise6.py",
    "pdf": "assignment6_instructions.pdf",
    "tasks": [
      {
        "title": "6.1 Direct PCA",
        "description": "Compute the mean, covariance and eigenvectors; draw principal directions; project points onto one component; compare nearest neighbors before and after projection; select components retaining 80% of variance in 50D data.",
        "image": "06-point-pca.png",
        "caption": "The figure shows principal directions, rank-one reconstruction and the 50D variance curve. Variance is normalized by the total, correcting the original 1d plot."
      },
      {
        "title": "6.2 Dual PCA",
        "description": "Decompose the sample-space covariance matrix, map its eigenvectors back to feature space, compare nonzero eigenvalues with direct PCA and reconstruct the original points. This is useful when there are many more pixels than training images.",
        "image": "06-dual-pca.png",
        "caption": "The reusable implementation removes null modes before dividing by singular values. The first source subsection is accidentally labeled 3a, but belongs to Exercise 2."
      },
      {
        "title": "6.3 Face-space decomposition",
        "description": "Load faces as columns, compute eigenfaces, reconstruct images, compare pixel edits with coefficient edits, sweep component counts, vary the first two coefficients and project an elephant image into face space. A webcam-recognition example is also preserved.",
        "image": "06-pca.png",
        "caption": "The displayed reconstruction uses one of the 64 training images, not a held-out recognition test. Personal webcam training images were not supplied. Several original plotting blocks are disabled."
      },
      {
        "title": "Learned eigenfaces",
        "description": "",
        "image": "06-eigenfaces.png",
        "caption": "Regenerated result from the assignment experiment."
      }
    ]
  }
];
