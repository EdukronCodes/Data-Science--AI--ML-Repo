# Jupyter Projects — Gen AI & Agentic AI Notes

Twenty end-to-end notebooks using **TensorFlow/Keras**, with theory markdown and runnable code.
**Note:** Cells are not pre-executed. Run locally after `pip install` dependencies.

## Setup
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow requests yfinance joblib openpyxl statsmodels
```

## Notebooks

### ANN (Feedforward / MLP)
- [ANN/01-Retail-Customer-Churn-ANN.ipynb](ANN/01-Retail-Customer-Churn-ANN.ipynb) — Telco retail churn prediction with MLP
- [ANN/02-Banking-Loan-Default-ANN.ipynb](ANN/02-Banking-Loan-Default-ANN.ipynb) — German credit loan default risk (UCI)
- [ANN/03-Employee-Attrition-ANN.ipynb](ANN/03-Employee-Attrition-ANN.ipynb) — HR employee attrition classification
- [ANN/04-Credit-Card-Fraud-ANN.ipynb](ANN/04-Credit-Card-Fraud-ANN.ipynb) — Imbalanced fraud detection (OpenML)
- [ANN/05-Housing-Price-Regression-ANN.ipynb](ANN/05-Housing-Price-Regression-ANN.ipynb) — California housing price regression

### CNN
- [CNN/01-MNIST-Digit-Classification-CNN.ipynb](CNN/01-MNIST-Digit-Classification-CNN.ipynb) — MNIST digits with ConvNet
- [CNN/02-Fashion-MNIST-Apparel-CNN.ipynb](CNN/02-Fashion-MNIST-Apparel-CNN.ipynb) — Fashion-MNIST clothing CNN
- [CNN/03-CIFAR10-Object-Classification-CNN.ipynb](CNN/03-CIFAR10-Object-Classification-CNN.ipynb) — CIFAR-10 object CNN
- [CNN/04-Chest-XRay-Pneumonia-CNN.ipynb](CNN/04-Chest-XRay-Pneumonia-CNN.ipynb) — Chest X-ray pneumonia (transfer learning)
- [CNN/05-Plant-Flower-Classification-CNN.ipynb](CNN/05-Plant-Flower-Classification-CNN.ipynb) — Flower/leaf image CNN (TF flowers)

### RNN / LSTM
- [RNN/01-IMDB-Sentiment-LSTM.ipynb](RNN/01-IMDB-Sentiment-LSTM.ipynb) — IMDB review sentiment LSTM
- [RNN/02-Stock-Price-Forecasting-LSTM.ipynb](RNN/02-Stock-Price-Forecasting-LSTM.ipynb) — AAPL stock forecasting LSTM
- [RNN/03-Energy-Consumption-Forecasting-LSTM.ipynb](RNN/03-Energy-Consumption-Forecasting-LSTM.ipynb) — Household power LSTM (UCI)
- [RNN/04-Air-Passengers-Forecasting-RNN.ipynb](RNN/04-Air-Passengers-Forecasting-RNN.ipynb) — Air Passengers classic RNN
- [RNN/05-Shakespeare-Char-RNN.ipynb](RNN/05-Shakespeare-Char-RNN.ipynb) — Character-level Shakespeare RNN

### Retail & Banking
- [Retail-Banking/01-Online-Retail-Customer-Segmentation.ipynb](Retail-Banking/01-Online-Retail-Customer-Segmentation.ipynb) — RFM + ANN high-value segments
- [Retail-Banking/02-Store-Sales-Forecasting-Retail.ipynb](Retail-Banking/02-Store-Sales-Forecasting-Retail.ipynb) — Retail demand LSTM forecast
- [Retail-Banking/03-Banking-Credit-Risk-German-Credit.ipynb](Retail-Banking/03-Banking-Credit-Risk-German-Credit.ipynb) — Bank credit risk scoring
- [Retail-Banking/04-Credit-Card-Fraud-Detection-Banking.ipynb](Retail-Banking/04-Credit-Card-Fraud-Detection-Banking.ipynb) — Banking fraud ANN pipeline
- [Retail-Banking/05-Bank-Marketing-Term-Deposit.ipynb](Retail-Banking/05-Bank-Marketing-Term-Deposit.ipynb) — Term deposit campaign prediction

## Folder Layout
```
Jupyter-Projects/
  ANN/          (5 notebooks)
  CNN/          (5 notebooks)
  RNN/          (5 notebooks)
  Retail-Banking/ (5 notebooks)
```
