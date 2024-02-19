# 3d-image-coords-volume-registration
An image processing pipeline to perform 3D Image volume (```*.tif```) and coordinates (```*.iv /*.swc```) registration for drosophila neuron traces data. The project was originally for Prof Chi-Hon Lee lab at ICOB, Academia Sinica.

To get started, check  ```explanation_example.ipyb```, where the detailed installation instruction was described.

In my opinion, the most useful function here is ```iv2swc.py```, which converts ```*.iv``` file (used by old NIH software to save coordinates) to ```*.swc``` file (more up-to-date format to save coordinates). 