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

***

# Nested Cross-Validation

Nested cross-validation is a technique used in machine learning model evaluation to obtain a more robust and unbiased estimate of a model's performance. It involves performing two levels of cross-validation: an outer loop and an inner loop.
<br/>

## Background
* By using nested cross-validation, the risk of overfitting the hyperparameters to a specific dataset is reduced.
* It provides a more realistic evaluation of the model's performance on unseen data by simulating the process of training and testing on multiple independent datasets.
* This approach is particularly useful when dealing with *limited data* and helps to produce a more robust estimation of a model's performance.

## Outer Cross-Validation (Outer Loop)
* The dataset is split into multiple folds (e.g., k folds), where each fold serves as a testing set in one iteration.
* In each iteration of the outer loop, one fold is used as the test set, and the remaining folds are used for training.

## Inner Cross-Validation (Inner Loop)
* For each training set in the outer loop, a further split is performed into multiple folds.
* The inner loop is used for hyperparameter tuning or model selection.
* It involves training and evaluating models on different subsets of the training data with different hyperparameter configurations.
* The best hyperparameters or model selected during the inner loop are then used to train a model on the entire training set from the outer loop.

## Performance Aggregation
* The performance of the model trained with the selected hyperparameters is evaluated on the test set from the outer loop.
* This process is repeated for each fold in the outer loop, and the performance metrics are aggregated to obtain a more reliable estimate of the model's generalization performance.

## Advantages
1. Allows validation-testing for various cases, increasing the reliability of model performance and enabling the attainment of a more generalized model.
2. Effective when the data is limited or each dataset varies significantly, making a single validation/test insufficient for trustworthy results.

## Limitations
1. Requires a considerable amount of time due to the need for nesting cross-validation.
