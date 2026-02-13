In this project, we inverted radiative transfer models against hypothetical TRUTHS reflectance data at different global locations and at different sampling geometries 
to retrieve the albedo and biophysical traits (such as leaf area index) of the canopy scene. In phase 1 of the project, we produced synthetic observations of bidirectional 
reflectance factors (BRFs) given specific illumination, viewing geometries and known canopy characteristics (e.g. leaf area index and projected crown cover) using the GORT radiative transfer model. 
In this phase, we inverted the BRFs using the linear Ross-Thick Li-Sparse kernel model to assess the ability to retrieve reflectances and model black sky surface albedos. 

In phase 2 of the project, we generated synthetic BRFs using the Semi-Discrete model and inverted against the same nonlinear model using the 4DEnVar data assimilation technique. 
In each phase, we quantified the additional benefits of having TRUTHS' data, specifically with regard to characteristics such as instrument accuracy and additional geometries, 
over those offered by other missions alone (such as Sentinel-2). This study illustrates the capacity, although beyond the lifetime of this project, to assimilate TRUTHS data 
into full complex land surface models such as JULES.

The Python files used in both Phases of the project can be found in the TRUTH_python_files folder. The files used to create the StreamLit application for displaying all Phase 1 results 
(https://truthsdirtt.streamlit.app/) are in the main folder.

The scripts of code required to run the GORT model (Phase 1), the Ross-Thick Li-Sparse kernel model (Phase 1), the Semi-Discrete model (Phase 2) and to apply the 4DEnVar data assimilation technique (Phase 2) are all background IP. The remaining code was produced for the purposes of this project.

The GORT and the Semi-Discrete model codes are written in C and Fortran respectively and must first be compiled (using the make command). The rest of the code in this project was written in Python so the GORT and Semi-Discrete models both required Python wrappers for implementation.

Code for the GORT model can be accessed at: https://github.com/tquaife/gort

Code for the Semi-Discrete model can be accessed at: https://fapar.jrc.ec.europa.eu/_www/models.php
