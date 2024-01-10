# Mis-Labeled Classification

Foobar is a Python library for dealing with word pluralization.


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

### Reference
1. [Identifying Mislabeled Instances in Classification Datasets](https://arxiv.org/pdf/1912.05283)
