# Portfolio preparation

The original course files remain unchanged in `coursework/`. The curated `cvportfolio/` modules select reusable functions and remove top-level data loading, plotting and interactive side effects.

## Correctness improvements in curated modules

- Replaced random threshold sampling with deterministic, histogram-based Otsu selection over 256 intensity bins. Histogram weights use the total pixel count.
- Fixed edge hysteresis: label all candidate edges, then retain connected components containing at least one strong pixel. Labeling only weak pixels could never find a strong pixel inside those same labels.
- Corrected the diagonal neighbor directions in edge non-maximum suppression for image row coordinates.
- Added true 2D spatial suppression for Harris and Hessian points instead of using edge suppression with zero directions. Constant-response images yield no feature points.
- Added a shaped empty result for feature matching when either image has no feature points.
- Normalized point coordinates for homography DLT; reject insufficient, nonfinite or rank-deficient configurations.
- Made homography RANSAC reproducible with a local seed; skip degenerate samples, refit the consensus and report inliers of the final model.
- Removed zero/near-zero eigenvalues from dual PCA before back-projection. Fixed eigenvector signs for consistent visual exports.
- Calculated cumulative explained variance as cumulative eigenvalues divided by their sum.

## Presentation additions

- Six deterministic result-generation routines, extra retrieval/eigenface views, a real rotating triangulation plot and machine-readable metrics.
- Responsive static viewer with edge/PCA settings, an alignment comparison slider, source links and explicit experiment limitations.
- Numerical regression tests, asset validation and a GitHub Actions workflow that regenerates every experiment.

Original known limitations are deliberately retained in the archive, including the misspelled coin threshold variable, experimental disparity direction logic, disabled displays and absent optional personal inputs. Use `python -m cvportfolio.generate` for the checked portfolio path.
