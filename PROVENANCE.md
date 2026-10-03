# Implementation & provenance

This portfolio starts with Matej Gaberc's six Python submissions for **Umetno zaznavanje (Machine Perception)**. The source snapshots in `assignment1/` … `assignment6/` are preserved byte-for-byte from the supplied files. Course PDFs, helper modules and example datasets are included as reference material, with their original authorship retained.

## Three layers

| Layer | Purpose | Authorship / role |
|---|---|---|
| `assignment1/` … `assignment6/` | Original scripts, assignment briefs, utilities and data | Matej's submissions alongside supplied course materials |
| `showcase/cvportfolio/` | Selected algorithms extracted into importable modules, corrected and checked | Adapted from the submissions; portfolio refactoring and fixes prepared with Codex assistance |
| `showcase/site/` | Visual explorer, real figures and measured outputs | Portfolio presentation added with Codex assistance |

The viewer does not run Python in the browser. It selects saved Python outputs, including four edge thresholds and six PCA component counts. The alignment slider compares two images in a common coordinate frame. The 3D animation is rendered from actual triangulated coordinates.

## What is implemented, and what is delegated

- **NumPy implementations:** normalized histograms, padded 1D convolution, median filtering, histogram distances, image derivatives using separable kernels, non-maximum suppression, Hough voting, feature matching, DLT homography estimation, RANSAC, normalized eight-point estimation, rank-two projection, linear triangulation and dual PCA.
- **OpenCV operations:** image decoding/color conversion, primitive filtering, connected components, morphology, local-max dilation and geometric image warping. Course helper functions also use OpenCV filtering.
- **Supplied helpers:** local descriptors and convolution helper from `a4_utils.py`; point normalization and correspondence reader from `a5_utils.py`. Copies are isolated under `showcase/cvportfolio/course_utils/` and retain their code.
- **Matplotlib/Pillow:** figures and the rotating 3D GIF.
- **HTML/CSS/JavaScript:** responsive results explorer; no service or database required.

The phrase “implemented” here does not imply inventing these established algorithms. Their construction was the subject of the course assignments.

## Scope of the reproducible showcase

The generator provides a result plate for each of the 18 main exercises, plus selected additional experiments. It does **not** execute every optional exercise or claim that every original script is production-ready. The unchanged scripts retain their original relative paths, commented plots, experimental alternatives and known limitations. Run the curated generator for a supported end-to-end experience.

The original video stabilization and webcam examples are preserved as coursework. They are not advertised as verified live demos: their video/personal-image inputs were not supplied. Missing brightness variants in Assignment 1 are replaced by explicitly labeled derived exposures in its histogram illustration.

## Reading the measurements

- Homography error is the median forward pixel reprojection error of final-model inliers on one New York image pair, at a 2 px threshold.
- Stereo reprojection error measures triangulated points projected into the two supplied cameras. Correspondences and projection matrices are given by the dataset.
- The reported “epipolar” measure is symmetric point-to-epipolar-line distance. The coursework calls its function `reprojection_errors`; it is not the same measure as 3D point reprojection.
- PCA reconstruction uses an image from the training set. Explained variance and reconstruction error do not measure held-out recognition accuracy.
- Denoising error uses intensities in [0, 1]; PCA error uses [0, 255]. These values are not directly comparable.
- The portfolio demonstrates classical computer vision and linear representations, not a trained deep neural network.

See [CHANGELOG.md](CHANGELOG.md) for the exact portfolio corrections and [ATTRIBUTION.md](ATTRIBUTION.md) for material credits.
