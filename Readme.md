
<img width="864" height="275" alt="image" src="https://github.com/user-attachments/assets/05647aea-9750-4568-8d38-bc2c4a1df28f" />

# AT-TSVM: Active Transfer Transductive Support Machine
Although there are models that explore to leverage features extracted from protein 3D structures in order to produce a better representative contact model, such a approach only remains theoretical, assuming the structure features to be available -- whereas in reality they are only available in the training data, but not in the testing data, whose structure is what needs to be predicted. In this work, we propose a novel approach that can train a model with training examples that contain both sequence features and atomic features, and apply the model on the test data that contain only sequence features but not atomic features, and yet still improves contact prediction than using sequence features alone. Specifically, our method AT-TSVM employs Transductive Support Vector Machines and augments with transfer and active learning to enhance contact prediction accuracy.

## Installation
Require python>=3.7

```
git clone https://github.com/Bander-Almalki/AT-TSVM.git
cd AT_TSVM
conda env create -f environment.yml
conda activate AT_TSVM
```
## About
- The Data folder contains the dataset in CSV format. The features can be categorized into 2 main categories. Atomic based features and sequence based features.

### Sequence-Based Features:
- The sequence based features focuse on the use of Co-evolutionary features extracted using four different models: CCmpred , Evfold, plmDCA, Gaussian DCA.
- A feature window of size 4 is applied to collect more information from neighboring residues.
(i + x, j + y), (i + x, j − y), (i − x, j − y), and (i − x, j + y), where (x, y) ∈ (0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (3, 0), (3, 4), (4, 0), (4, 3), (4, 4).
- A moving window of size 3 is also applied (i + 1, i, i − 1), where i is the residue in the first helix
- the total number of sequence based features are 300 features per residue.
- PCA is used to reduce the feature space by ranking the PCAs based on their contact predictive power then choosing the Top 10 PCAs
- To extract the complete sequence-based features, please refer to [DeepHelicon Model](https://github.com/2003100127/deephelicon/tree/master)

### Structure-Based Features
- 5 Features per residue
  - Average distance between atoms
  - Relative residue angle,
  - Cα atoms distance,
  - Variation in distances between atoms, and
  -  inter-helical angle.
- 3∗3 window
- result in 40 features
- PCA is applied to reduce the feature space
- top 2 PCAs are used.
- To extract the complete structure-based features, please refer to [Structural_Features](https://link.springer.com/chapter/10.1007/978-3-031-34960-7_25)
## The main scripts:
