# Decision Trees

## Algorithm Overview
Decision Trees are a type of supervised learning algorithm used for both classification and regression tasks. They work by splitting the data into subsets based on the value of input features, creating a tree-like model of decisions.

## Problem Type
- Classification
- Regression

## Mathematical Foundation
Decision Trees use a tree structure where each internal node represents a feature (or attribute), each branch represents a decision rule, and each leaf node represents an outcome. The algorithm recursively partitions the data based on feature values.

## Cost Function
The cost function for Decision Trees varies based on the task:
- For classification, it often uses Gini impurity or entropy.
- For regression, it typically uses mean squared error (MSE).

## Optimization Techniques
- Pruning: Reducing the size of the tree by removing sections that provide little power to classify instances.
- Setting a maximum depth to avoid overfitting.

## Hyperparameters
- Maximum depth of the tree
- Minimum samples split
- Minimum samples leaf
- Criterion (Gini impurity or entropy for classification, MSE for regression)

## Assumptions
- The features are independent of each other.
- The relationship between the features and the target variable is non-linear.

## Advantages
- Easy to understand and interpret.
- Requires little data preprocessing (no need for normalization).
- Can handle both numerical and categorical data.

## Workflow
1. Select the best feature to split the data.
2. Create branches for each possible value of the feature.
3. Repeat the process for each branch until a stopping criterion is met (e.g., maximum depth or minimum samples).

## Implementations
- Scikit-learn: `DecisionTreeClassifier` and `DecisionTreeRegressor`
- R: `rpart` package
- MATLAB: `fitctree` and `fitrtree`

## Hyperparameter Tuning
- Use techniques like Grid Search or Random Search to find the optimal hyperparameters.
- Cross-validation can help in assessing the performance of different hyperparameter settings.

## Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score for classification tasks.
- Mean Absolute Error (MAE), Mean Squared Error (MSE) for regression tasks.

## Bias-Variance Analysis
- Decision Trees can exhibit high variance, leading to overfitting.
- Pruning and setting constraints can help reduce variance while maintaining bias.

## Overfitting Handling
- Pruning the tree after it has been created.
- Setting a maximum depth or minimum samples per leaf.

## Comparisons
- Compared to linear models, Decision Trees can capture non-linear relationships.
- They are less robust to noise compared to ensemble methods like Random Forests.

## Real-World Applications
- Customer segmentation
- Credit scoring
- Medical diagnosis

## Practical Projects
- Building a customer churn prediction model.
- Creating a decision support system for loan approvals.

## Performance Optimization
- Use ensemble methods like Random Forests or Gradient Boosting to improve performance.
- Optimize hyperparameters using cross-validation.

## Common Interview Questions
1. What are the advantages and disadvantages of Decision Trees?
2. How do you prevent overfitting in Decision Trees?
3. Explain the difference between Gini impurity and entropy.
4. What is pruning, and why is it important?