## Table 5.1 -- Experimental Coverage

| Configuration | Model | Completed datasets | Learning rate |
|---|---|---|---|
| No fine-tuning | TabPFN v3 | 32 | - |
| Attention-only | TabPFN v3 | 30 | 1e-05 |
| Layer-wise (layer 0) | TabPFN v3 | 30 | 1e-05 |
| Layer-wise (layer 11) | TabPFN v3 | 30 | 1e-05 |
| Layer-wise (layer 17) | TabPFN v3 | 30 | 1e-05 |
| Layer-wise (layer 23) | TabPFN v3 | 30 | 1e-05 |
| Layer-wise (layer 6) | TabPFN v3 | 30 | 1e-05 |
| MLP-only | TabPFN v3 | 30 | 1e-05 |
| Full fine-tuning | TabPFN v3 | 26 | 1e-05 |

## Table -- Attention-only vs Baseline

| Metric | Mean baseline | Mean variant | Mean difference | Wins | Ties | Losses |
|---|---|---|---|---|---|---|
| Accuracy | 0.8673 | 0.8674 | 0.00010 | 0 | 30 | 0 |
| Balanced Accuracy | 0.7169 | 0.7172 | 0.00026 | 1 | 28 | 1 |
| ROC-AUC | 0.8663 | 0.8667 | 0.00035 | 0 | 30 | 0 |
| Negative Log Loss | -0.3137 | -0.3131 | 0.00056 | 0 | 30 | 0 |

(compared on 30 datasets present in both baseline and this run)

## Table -- Layer-wise (layer 0) vs Baseline

| Metric | Mean baseline | Mean variant | Mean difference | Wins | Ties | Losses |
|---|---|---|---|---|---|---|
| Accuracy | 0.8673 | 0.8673 | -0.00002 | 0 | 30 | 0 |
| Balanced Accuracy | 0.7169 | 0.7170 | 0.00008 | 0 | 29 | 1 |
| ROC-AUC | 0.8663 | 0.8663 | -0.00002 | 0 | 30 | 0 |
| Negative Log Loss | -0.3137 | -0.3136 | 0.00006 | 0 | 30 | 0 |

(compared on 30 datasets present in both baseline and this run)

## Table -- MLP-only vs Baseline

| Metric | Mean baseline | Mean variant | Mean difference | Wins | Ties | Losses |
|---|---|---|---|---|---|---|
| Accuracy | 0.8673 | 0.8675 | 0.00019 | 0 | 30 | 0 |
| Balanced Accuracy | 0.7169 | 0.7173 | 0.00042 | 0 | 30 | 0 |
| ROC-AUC | 0.8663 | 0.8667 | 0.00038 | 0 | 30 | 0 |
| Negative Log Loss | -0.3137 | -0.3131 | 0.00060 | 1 | 29 | 0 |

(compared on 30 datasets present in both baseline and this run)

## Table -- Full fine-tuning vs Baseline

| Metric | Mean baseline | Mean variant | Mean difference | Wins | Ties | Losses |
|---|---|---|---|---|---|---|
| Accuracy | 0.8535 | 0.8535 | 0.00004 | 0 | 26 | 0 |
| Balanced Accuracy | 0.7161 | 0.7163 | 0.00024 | 1 | 25 | 0 |
| ROC-AUC | 0.8579 | 0.8581 | 0.00017 | 0 | 26 | 0 |
| Negative Log Loss | -0.3428 | -0.3423 | 0.00058 | 1 | 25 | 0 |

(compared on 26 datasets present in both baseline and this run)

## Table 5.5 -- Overall Comparison of Fine-Tuning Strategies (mean Δ [95% CI])

| Strategy | Datasets | Mean Δ Accuracy [95% CI] | Mean Δ Bal. Acc. [95% CI] | Mean Δ ROC-AUC [95% CI] | Mean Δ Negative Log Loss [95% CI] |
|---|---|---|---|---|---|
| Attention-only | 30 | 0.00010 [-0.00015, 0.00036] | 0.00026 [-0.00043, 0.00095] | 0.00035 [-0.00004, 0.00075] | 0.00056 [0.00019, 0.00093] |
| Layer-wise (layer 0) | 30 | -0.00002 [-0.00014, 0.00011] | 0.00008 [-0.00040, 0.00055] | -0.00002 [-0.00018, 0.00014] | 0.00006 [-0.00012, 0.00023] |
| MLP-only | 30 | 0.00019 [-0.00011, 0.00049] | 0.00042 [-0.00026, 0.00109] | 0.00038 [-0.00005, 0.00081] | 0.00060 [0.00017, 0.00103] |
| Full fine-tuning | 26 | 0.00004 [-0.00015, 0.00024] | 0.00024 [-0.00064, 0.00112] | 0.00017 [-0.00011, 0.00046] | 0.00058 [-0.00001, 0.00117] |

*95% CI: two-sided t-distribution (df = n_datasets - 1), computed directly over the dataset-level deltas shown above -- not averaged from per-fold CIs.*

## Table 5.5b -- Overall Comparison of Fine-Tuning Strategies (median)

| Strategy | Median Δ Accuracy | Median Δ Bal. Acc. | Median Δ ROC-AUC | Median Δ Negative Log Loss |
|---|---|---|---|---|
| Attention-only | -0.00000 | 0.00001 | 0.00011 | 0.00022 |
| Layer-wise (layer 0) | -0.00003 | 0.00002 | 0.00001 | 0.00004 |
| MLP-only | 0.00000 | 0.00011 | 0.00010 | 0.00015 |
| Full fine-tuning | 0.00002 | 0.00007 | 0.00002 | 0.00021 |

## Table 5.6 -- Common-Subset Comparison (only datasets present in baseline + all 4 strategies, n=26)

| Strategy | Mean Δ Accuracy [95% CI] | Mean Δ Bal. Acc. [95% CI] | Mean Δ ROC-AUC [95% CI] | Mean Δ Negative Log Loss [95% CI] |
|---|---|---|---|---|
| Attention-only | 0.00005 [-0.00023, 0.00034] | 0.00017 [-0.00056, 0.00090] | 0.00023 [-0.00010, 0.00056] | 0.00050 [0.00012, 0.00089] |
| Layer-wise (layer 0) | -0.00004 [-0.00018, 0.00010] | -0.00011 [-0.00060, 0.00039] | 0.00005 [-0.00006, 0.00016] | 0.00007 [-0.00010, 0.00023] |
| MLP-only | 0.00016 [-0.00017, 0.00049] | 0.00028 [-0.00038, 0.00093] | 0.00026 [-0.00012, 0.00065] | 0.00056 [0.00009, 0.00102] |
| Full fine-tuning | 0.00004 [-0.00015, 0.00024] | 0.00024 [-0.00064, 0.00112] | 0.00017 [-0.00011, 0.00046] | 0.00058 [-0.00001, 0.00117] |

*95% CI: two-sided t-distribution (df = n_datasets - 1) over the common-subset dataset-level deltas.*

| Strategy | Wins | Ties | Losses | (Accuracy, on common subset) |
|---|---|---|---|---|
| Attention-only | 0 | 26 | 0 | n=26 |
| Layer-wise (layer 0) | 0 | 26 | 0 | n=26 |
| MLP-only | 0 | 26 | 0 | n=26 |
| Full fine-tuning | 0 | 26 | 0 | n=26 |

Datasets included in the common subset: Amazon_employee_access, Bank_Customer_Churn, E-CommereShippingData, Fitness_Club, HR_Analytics_Job_Change_of_Data_Scientists, Is-this-a-good-customer, MIC, Marketing_Campaign, anneal, bank-marketing, blood-transfusion-service-center, churn, credit-g, credit_card_clients_default, diabetes, hazelnut-spread-contaminant-detection, heloc, in_vehicle_coupon_recommendation, jm1, maternal_health_risk, online_shoppers_intention, qsar-biodeg, seismic-bumps, splice, students_dropout_and_academic_success, website_phishing

## Table 5.3 -- Layer-wise Fine-Tuning: Mean Δ per Layer [95% CI]

| Layer | Datasets | Mean Δ Accuracy [95% CI] | Mean Δ Bal. Acc. [95% CI] | Mean Δ ROC-AUC [95% CI] | Mean Δ Negative Log Loss [95% CI] |
|---|---|---|---|---|---|
| 0 | 30 | -0.00002 [-0.00014, 0.00011] | 0.00008 [-0.00040, 0.00055] | -0.00002 [-0.00018, 0.00014] | 0.00006 [-0.00012, 0.00023] |
| 6 | 30 | -0.00003 [-0.00015, 0.00009] | 0.00006 [-0.00041, 0.00053] | -0.00004 [-0.00019, 0.00011] | 0.00003 [-0.00014, 0.00020] |
| 11 | 30 | 0.00006 [-0.00011, 0.00024] | 0.00014 [-0.00034, 0.00062] | 0.00010 [-0.00012, 0.00031] | 0.00018 [-0.00009, 0.00045] |
| 17 | 30 | 0.00001 [-0.00012, 0.00013] | 0.00007 [-0.00040, 0.00054] | -0.00002 [-0.00017, 0.00013] | 0.00006 [-0.00011, 0.00022] |
| 23 | 30 | -0.00004 [-0.00018, 0.00010] | 0.00003 [-0.00046, 0.00051] | -0.00007 [-0.00024, 0.00009] | -0.00001 [-0.00020, 0.00019] |

*95% CI: two-sided t-distribution (df = n_datasets - 1) over the dataset-level deltas for that layer.*
