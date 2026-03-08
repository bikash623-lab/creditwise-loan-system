CreditWise Loan Approval Prediction
This was a project I worked on where I had to build a machine learning system for a fictional bank called SecureTrust Bank. The goal was simple — the bank was tired of manually reviewing loan applications and wanted an ML model to automatically predict whether a loan should be approved or rejected.

The Problem
SecureTrust Bank processes hundreds of loan applications every day. Their old process was fully manual — loan officers would go through each application one by one, which was slow and inconsistent. Sometimes good customers got rejected, sometimes risky ones got approved. Neither is good for the bank.
My job was to build something smarter using historical application data.

Dataset
The dataset has around 1000 applicants with information like their income, credit score, employment status, savings, loan amount, and more.
ColumnWhat it meansApplicant_IncomeMonthly incomeCredit_ScoreCredit bureau scoreDTI_RatioDebt-to-income ratioSavingsSavings balanceLoan_AmountHow much they're asking forEmployment_StatusSalaried / Self-Employed / BusinessEducation_LevelUndergraduate / Graduate / PostgraduateLoan_ApprovedTarget — 1 = Approved, 0 = Rejected
One thing I noticed early on — the dataset is imbalanced. About 70% of applications were rejected and only 30% approved. This meant I couldn't just use accuracy as my main metric.

What I Did
EDA first — before touching any model I explored the data to understand it. I plotted distributions, checked for outliers using boxplots, and looked at how each feature related to the target. Credit score and DTI ratio turned out to be the strongest signals.
Cleaned the data — handled missing values separately for numeric columns (filled with median) and categorical columns (filled with most frequent value). Also dropped the Applicant_ID column since it's just an identifier and adds no value to the model.
Encoding — used LabelEncoder for Education Level since it has a natural order (Undergraduate → Graduate → Postgraduate). Used OneHotEncoder for everything else like Gender, Employment Status, Marital Status etc. since those have no ranking between them.
Scaling — applied StandardScaler before training since some models like KNN are sensitive to feature ranges.

Models I Tried
I tested three models and compared them:
ModelAccuracyPrecisionRecallF1 ScoreLogistic Regression86.5%78.33%77.05%77.69%Naive Bayes86.5%80.36%73.77%76.92%KNN (k=9)76.0%65.85%44.26%52.94%
Logistic Regression came out on top. It tied with Naive Bayes on accuracy but had the best F1 score which matters more here because of the class imbalance.
KNN struggled quite a bit — I think this is because after OneHot encoding there were a lot of columns, and KNN doesn't handle high dimensions well.

Confusion Matrix — Logistic Regression
                  Predicted No   Predicted Yes
Actual No              126              13
Actual Yes              14              47
13 people were incorrectly flagged as approved (false positives) and 14 actual approvals were missed (false negatives). For a bank, the false positives are the more dangerous ones — those are risky customers who slip through.
