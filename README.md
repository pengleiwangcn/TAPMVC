# TAPMVC

A click-and-run MATLAB demo for the paper **Topology-Aware Preference Learning for Multi-View Fuzzy Clustering**.

## Run the demo

1. Download the repository and set its root folder as the **Current Folder** in MATLAB.
2. Open [`demo.m`](demo.m).
3. Click **Run** or press **F5**.

The default example uses **MSRCv1**. The MATLAB Command Window displays ACC, NMI, Purity, F1, Precision, Recall, and ARI.

## Use another dataset

Change the dataset filename at the top of `demo.m` and set `rho` to the corresponding value listed in the script, then run it again. The included datasets are in [`data/`](data/).

