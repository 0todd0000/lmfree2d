# lmfree2d: Landmark-free hypothesis tests regarding 2D contour shapes

<br>

### Overview

This repository contains data, Python and R code for conducting parametric or
nonparametric, landmark-free hypothesis testing regarding 2D contour shapes.
Data in this repository are redistributed from the [The 2D Shape Structure Dataset](http://2dshapesstructure.github.io)
under the terms of its [MIT license](https://opensource.org/licenses/MIT).

Landmark-free hypothesis testing results look like those in the figure below. These results
represent mass-multivariate analysis of registered 2D contour data, using [Statistical Parametric
Mapping](https://en.wikipedia.org/wiki/Statistical_parametric_mapping) and [Statistical non-Parametric
Mapping](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.1058).

This figure depicts mean shapes for each of two groups (A and B), an omnibus p value (representing
the probability that smooth, bivariate continuum data would randomly produce a mean
difference as large as the largest observed difference), and highlighted contour points (hot-colored circles)
representing locations where shape effects were largest.

<br>




<br>

<img src="https://github.com/0todd0000/lmfree2d/tree/master/Figures/results_spm.pdf" alt="results_spm" width="700"/>

<br>

### Getting started

The best place to start is the notebooks (in the Notebooks folder), which summarize
four different approaches to hypothesis testing for 2D contour shapes:

1. **landmarks_uv**:  &nbsp; &nbsp; univariate analysis of landmarks (i.e., Procrustes ANOVA)
1. **landmarks_massmv**:  &nbsp; &nbsp; mass-multivariate analysis of landmarks 
1. **contours_uv**: &nbsp; &nbsp; landmark-free, univariate analysis of registered contours (i.e., contour-level Procrustes ANOVA)
1. **contours_massmv**:  &nbsp; &nbsp; landmark-free, mass-multivariate analysis of registered contours using Statistical Parametric Mapping 

Note that the results in the figure above correspond to the final method: **contours_massmv**.

Please refer to the notebooks and the papers below for more details.

<br>


### Dependencies:

Python:

* [geomdl](https://pypi.org/project/geomdl/)
* [networkx](https://networkx.github.io)
* [pandas](https://pandas.pydata.org)
* [pycpd](https://pypi.org/project/pycpd/)
* [sklearn](https://scikit-learn.org/stable/)
* [spm1d](https://spm1d.org)

R:

* [geomorph](https://cran.r-project.org/web/packages/geomorph/index.html)


<br>

### Support

For support, please [submit an issue here](https://github.com/0todd0000/lmfree2d/issues).

Please do **not** email the authors directly. Email requests for support will be routed to this repository's [issues site on GitHub](https://github.com/0todd0000/lmfree2d/issues).

<br>

### Please cite:

Pataky TC, Yagi M, Ichihashi N, Cox PG (2020). Automated, landmark-free,
parametric hypothesis tests regarding two-dimensional contour shapes using
coherent point drift registration and statistical parametric mapping.
*PeerJ Comp Sci* (in review).

Carlier, A., Leonard, K., Hahmann, S., Morin, G., and Collins, M. (2016).
The 2D shape structure dataset: a user annotated open access database.
*Computers & Graphics* **58**: 23–30.





