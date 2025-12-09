# Implementing-Dense-Match-Summarization-for-Faster-Two-View-Estimation

For the video demonstration of this repo, please find the video at [placeholder].
In addition, the project report is named "Project_Report" in the repo.

A brief overview of the pipeline folder:
-precompute_dkm_scannet.py (precompute the DKM dense matches and process camera poses and intrinsics)
-DM_Core.h (data and experiment configuration)
-DM_IO.h/.cpp (load precomputed matches, preprocessed camera poses and intrinsics, as well as original images)
-DM_ClusterSummary.h/.cpp (clustering and summarizing matches per image pair)
-DM_RansacEval.h/.cpp (run RANSAC algorithm and collect evaluational data)
-Dense_Match_Summarization.cpp (main driver to run the pipeline)

Steps to run the pipeline (windows environment)
Step 1: Download the pipeline folder.
Step 2: Download the dataset at https://drive.google.com/drive/folders/1UDUlP5yzXc9CDQmC40P1-P8rMI-faG92?usp=drive_link, and move the unzipped data folder into the project folder.
Step 3.1: (Python part)
1. make sure python is installed
2. open terminal and navigate to the project folder, run the following commands:
   -python -m venv venv_dkm
   -venv_dkm\Scripts\activate
   -python -m pip install --upgrade pip
   -pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 (noted that 130 indicates the cuda version in your environment, please double check ur version before proceeding with this command)
   -pip install git+https://github.com/Parskatt/DKM.git
   -python precompute_dkm_scannet.py
Step 3.2: (C++ part)
1. make sure opencv is installed
2. build and run using the main driver file

Notes:
  Currently, the pipeline only loads 6 images from 5 scenes and computes 2000 DKM matches per image pair. To modify this, please visit precompute_dkm_scannet.py and adjust MAX_SCENES, MAX_PAIRS_PER_SCENE, and MAX_MATCHES_PER_PAIR. Afterwards, adjust maxScenesToLoad, maxPairsPerScene, and numPairsToProcess in Dense_Match_Summarization.cpp to match the values you input earlier.

  
