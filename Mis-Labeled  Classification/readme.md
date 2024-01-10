# Mis-Labeled Classification

Mis-Labeled Classification is specific classification mehod to deal with Data for supervised learning consists of incorrect labels for variety of reasons.
<br/>

## Background

* Labeling is either done manually by experts or at least checked by humans as, which costs both time and money.
* Subjectivity
* Data-entry error
* Inadequacy of the information

## Releated Works

* Learning with Label Noise
* Label Cleansing
* Label Noise Identification

## Methodology

### Main Idea

**Use Classifiers as filters!**

1. Traing instances to Filter
2. Correctly Labeled Training instances
3. Learning Algorithm to get fianl results

### Pipeline

1. Train a classifier g on a given dataset (x, y)
2. Use the same classifier g to obtain class probabilities for x.
3. Look for instances for which the class probability of the original label is minimal, e.g., instances for which the
classifier considers the original label extremely unlikely given
the feature distribution learned during training.
4. Re-evluated these returned instances.


### Classification Algorithm

* Clssifier needs be flexible to correctly learn from a variety of datasets.
* Generalize well on noisy datasets.
* Neural networks, especially with dropout layers are a natural choice in this setting.

### Types of Mis-Labeled Classification for Filtering
* **Single Algorithm Filter**
  * Filtering is done by one algorithm
  * Instance is marked as mislabeled if this algorithm tagged it as mislabeled
* **Majority Vote Filter**
  * Filtering is done by multiple algorithms
  * Instance is marked as mislabeled if more than half of the algorithms tagged it as mislabeled
* **Consensus Filter**
  * Filtering is done by multiple algorithms
  * Instance is marked as mislabeled if all of the algorithms tagged it as mislabeled

## Reference
1. [Identifying Mislabeled Instances in Classification Datasets](https://arxiv.org/pdf/1912.05283)
2. [Handling mislabeled training data for classification](https://longjp.github.io/statcomp/projects/mislabeled.pdf)
