# Results

Every number is computed from the saved predictions in `results/predictions.parquet` by `python -m senti.report`.

## Positive or negative: accuracy (%) on the positive and negative test items

Sorted by the mean of the two fresh sets (Amazon 2022 to 2023, original and translated), the only text no model can have trained on.

| Model | IMDb movie reviews (en) | Amazon reviews 2022 to 2023 (en) | Same reviews, machine translated (9 languages) | Tweets (8 languages) | Amazon reviews (6 languages) | Fresh mean |
|---|---|---|---|---|---|---|
| nlptown | 87.2 (84.3 to 89.8) | 94.3 (92.3 to 96.1) | 88.1 (87.2 to 88.9) | 71.4 (69.9 to 72.9) | 89.4 (88.5 to 90.4) | 91.2 |
| clapai | 97.3 (96.0 to 98.5) | 95.2 (93.3 to 96.7) | 85.0 (84.0 to 85.9) | 90.0 (88.9 to 90.9) | 96.3 (95.7 to 97.0) | 90.1 |
| cardiff_xlmt | 62.0 (58.0 to 65.8) | 85.3 (82.3 to 88.1) | 83.6 (82.6 to 84.5) | 85.6 (84.5 to 86.7) | 88.1 (87.1 to 89.2) | 84.4 |
| tabularisai | 82.0 (79.2 to 85.2) | 85.2 (82.2 to 88.0) | 83.6 (82.6 to 84.6) | 68.7 (67.1 to 70.1) | 84.5 (83.3 to 85.6) | 84.4 |
| lxyuan | 74.2 (70.7 to 77.5) | 84.3 (81.4 to 87.2) | 78.9 (77.7 to 80.1) | 74.1 (72.7 to 75.5) | 81.3 (80.0 to 82.5) | 81.6 |
| siebert | 94.0 (92.0 to 95.8) | 93.7 (91.7 to 95.5) | 66.7 (65.5 to 67.9) | 66.3 (64.7 to 67.9) | 72.4 (70.9 to 73.9) | 80.2 |
| cardiff_en | 78.8 (75.5 to 82.2) | 90.2 (87.6 to 92.4) | 58.7 (57.4 to 60.0) | 66.6 (65.0 to 67.9) | 61.8 (60.3 to 63.4) | 74.4 |
| distilbert_v1 | 91.0 (88.7 to 93.2) | 88.3 (85.7 to 91.0) | 54.0 (52.7 to 55.3) | 55.5 (53.9 to 57.0) | 59.5 (58.1 to 61.1) | 71.2 |
| distilbert_sst2 | 90.0 (87.7 to 92.2) | 83.5 (80.6 to 86.1) | 55.6 (54.3 to 56.9) | 55.9 (54.3 to 57.6) | 60.4 (58.8 to 61.9) | 69.5 |
| distilbert_v2 | 91.3 (89.0 to 93.5) | 87.8 (85.1 to 90.5) | 50.3 (48.9 to 51.6) | 56.7 (55.0 to 58.3) | 56.9 (55.2 to 58.3) | 69.1 |
| tfidf_lr | 89.3 (87.2 to 91.5) | 78.8 (75.5 to 81.8) | 53.3 (52.0 to 54.6) | 53.9 (52.1 to 55.6) | 55.9 (54.2 to 57.6) | 66.1 |

## With an unsure band: 3 way macro F1 (%) against negative, neutral, positive labels

The band [lo, hi] is chosen per model on validation rows only. IMDb has no neutral items, so it is left out here.

| Model | Band | Amazon reviews 2022 to 2023 (en) | Same reviews, machine translated (9 languages) | Tweets (8 languages) | Amazon reviews (6 languages) | Fresh mean | Neutral items marked unsure | Polar items it commits on | Accuracy when it commits |
|---|---|---|---|---|---|---|---|---|---|
| nlptown | 0.24 to 0.74 | 72.2 (69.2 to 74.9) | 64.6 (63.6 to 65.5) | 47.8 (46.4 to 49.3) | 67.7 (66.5 to 68.9) | 68.4 | 53.1 | 71.4 | 93.0 |
| clapai | 0.23 to 0.76 | 74.4 (71.6 to 77.2) | 61.9 (60.8 to 62.8) | 74.2 (73.1 to 75.4) | 79.2 (78.1 to 80.3) | 68.2 | 67.9 | 80.7 | 97.3 |
| cardiff_xlmt | 0.22 to 0.67 | 61.3 (58.2 to 64.3) | 58.2 (57.2 to 59.2) | 65.9 (64.6 to 67.2) | 60.7 (59.5 to 62.0) | 59.7 | 52.8 | 60.3 | 92.7 |
| lxyuan | 0.42 to 0.68 | 57.1 (54.0 to 60.2) | 53.0 (51.9 to 54.0) | 50.1 (48.7 to 51.5) | 54.5 (53.3 to 55.8) | 55.0 | 37.9 | 72.8 | 84.3 |
| tabularisai | 0.19 to 0.59 | 55.4 (52.1 to 58.4) | 54.4 (53.4 to 55.4) | 45.7 (44.3 to 47.1) | 55.7 (54.6 to 57.0) | 54.9 | 35.6 | 75.4 | 86.2 |
| siebert | 0.01 to 0.99 | 54.9 (52.5 to 57.4) | 44.8 (43.8 to 45.8) | 44.1 (42.7 to 45.5) | 45.7 (44.3 to 46.9) | 49.8 | 21.3 | 86.9 | 81.8 |
| cardiff_en | 0.48 to 0.55 | 52.4 (49.9 to 54.9) | 40.1 (39.0 to 41.0) | 50.1 (48.7 to 51.4) | 42.2 (40.9 to 43.6) | 46.2 | 40.9 | 71.9 | 77.5 |
| distilbert_sst2 | 0.03 to 0.51 | 52.2 (49.5 to 54.9) | 35.4 (34.3 to 36.4) | 38.7 (37.3 to 39.9) | 41.1 (39.8 to 42.4) | 43.8 | 40.5 | 68.8 | 75.6 |
| distilbert_v1 | 0.50 to 0.76 | 50.3 (47.7 to 52.8) | 36.3 (35.2 to 37.4) | 37.4 (36.1 to 38.8) | 41.2 (39.8 to 42.5) | 43.3 | 36.3 | 72.8 | 73.8 |
| distilbert_v2 | 0.50 to 0.71 | 54.5 (51.7 to 57.5) | 29.2 (28.4 to 29.9) | 37.0 (35.7 to 38.3) | 38.0 (36.7 to 39.4) | 41.9 | 43.6 | 67.8 | 72.7 |
| tfidf_lr | 0.44 to 0.52 | 48.9 (45.9 to 52.0) | 32.7 (31.8 to 33.7) | 34.1 (32.8 to 35.4) | 36.6 (35.2 to 38.0) | 40.8 | 43.2 | 63.6 | 70.0 |

## Accuracy (%) by language, positive and negative items

| Model | amazon23/en | amazon23_mt/ar | amazon23_mt/de | amazon23_mt/es | amazon23_mt/fr | amazon23_mt/hi | amazon23_mt/it | amazon23_mt/ja | amazon23_mt/pt | amazon23_mt/zh | amazon_multi/de | amazon_multi/en | amazon_multi/es | amazon_multi/fr | amazon_multi/ja | amazon_multi/zh | imdb/en | tweets/ar | tweets/de | tweets/en | tweets/es | tweets/fr | tweets/hi | tweets/it | tweets/pt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nlptown | 94.3 | 83.7 | 91.7 | 89.8 | 91.7 | 83.8 | 91.0 | 83.0 | 90.0 | 87.8 | 95.7 | 95.7 | 95.2 | 96.5 | 76.7 | 77.0 | 87.2 | 70.0 | 73.2 | 81.6 | 76.1 | 74.1 | 60.5 | 72.3 | 63.2 |
| clapai | 95.2 | 79.3 | 91.0 | 92.0 | 92.2 | 59.2 | 85.7 | 87.7 | 90.5 | 87.3 | 97.5 | 96.5 | 98.2 | 98.2 | 95.7 | 92.0 | 97.3 | 79.1 | 92.3 | 98.2 | 92.7 | 93.0 | 82.3 | 91.1 | 91.1 |
| cardiff_xlmt | 85.3 | 80.2 | 81.8 | 84.5 | 82.5 | 83.8 | 84.2 | 86.2 | 84.2 | 84.7 | 88.2 | 89.3 | 89.5 | 91.0 | 87.5 | 83.3 | 62.0 | 84.8 | 89.3 | 92.0 | 87.3 | 87.3 | 73.2 | 80.0 | 91.1 |
| tabularisai | 85.2 | 81.2 | 84.7 | 84.7 | 84.8 | 83.2 | 83.8 | 81.5 | 84.2 | 84.5 | 85.2 | 84.5 | 89.7 | 87.2 | 80.2 | 80.2 | 82.0 | 65.9 | 68.6 | 78.4 | 74.8 | 69.1 | 58.4 | 66.8 | 67.3 |
| lxyuan | 84.3 | 72.3 | 80.7 | 84.7 | 81.2 | 65.5 | 81.3 | 78.3 | 81.0 | 84.7 | 80.8 | 82.2 | 83.5 | 83.0 | 73.7 | 84.5 | 74.2 | 78.2 | 75.2 | 83.9 | 75.9 | 80.9 | 65.2 | 66.6 | 67.0 |
| siebert | 93.7 | 50.2 | 80.3 | 85.2 | 76.3 | 50.2 | 77.0 | 49.8 | 80.5 | 50.8 | 80.3 | 96.2 | 85.3 | 72.5 | 50.0 | 50.2 | 94.0 | 50.0 | 71.8 | 91.1 | 68.0 | 64.5 | 65.9 | 60.7 | 58.4 |
| cardiff_en | 90.2 | 51.5 | 50.3 | 70.8 | 67.3 | 49.8 | 65.7 | 55.5 | 67.0 | 50.5 | 51.2 | 90.8 | 64.7 | 63.0 | 52.3 | 49.0 | 78.8 | 65.5 | 62.5 | 95.5 | 61.1 | 62.7 | 68.9 | 57.0 | 59.3 |
| distilbert_v1 | 88.3 | 49.8 | 50.7 | 57.5 | 53.3 | 54.8 | 57.3 | 47.7 | 58.8 | 56.2 | 51.0 | 89.3 | 55.5 | 56.3 | 51.5 | 53.5 | 91.0 | 39.3 | 53.2 | 80.7 | 51.1 | 56.4 | 62.5 | 58.0 | 42.5 |
| distilbert_sst2 | 83.5 | 53.5 | 51.3 | 61.8 | 60.0 | 51.8 | 59.7 | 51.8 | 57.8 | 52.2 | 51.3 | 87.3 | 63.0 | 60.7 | 50.0 | 49.8 | 90.0 | 48.4 | 51.6 | 75.9 | 53.0 | 54.8 | 62.7 | 52.5 | 48.6 |
| distilbert_v2 | 87.8 | 50.0 | 49.8 | 50.8 | 50.2 | 50.0 | 51.0 | 49.8 | 51.3 | 50.0 | 49.8 | 89.5 | 50.2 | 51.8 | 50.0 | 50.0 | 91.3 | 48.6 | 54.3 | 80.5 | 53.4 | 57.5 | 58.6 | 56.1 | 44.5 |
| tfidf_lr | 78.8 | 50.8 | 55.2 | 58.7 | 56.0 | 51.5 | 55.7 | 51.0 | 50.8 | 50.0 | 50.2 | 79.8 | 54.5 | 52.0 | 49.7 | 49.3 | 89.3 | 58.6 | 54.8 | 66.8 | 53.2 | 50.7 | 54.8 | 43.2 | 48.9 |

## Speed on one 4 core CPU runner (same 320 test texts for every model, same machine)

| Model | Texts per second | Parameters | Licence |
|---|---|---|---|
| nlptown | 5.8 | 167M | mit |
| clapai | 0.5 | 150M | apache-2.0 |
| cardiff_xlmt | 6.7 | 278M | none listed |
| tabularisai | 11.2 | 135M | cc-by-nc-4.0 |
| lxyuan | 11.3 | 135M | apache-2.0 |
| siebert | 1.0 | 355M | none listed |
| cardiff_en | 3.4 | 125M | cc-by-4.0 |
| distilbert_v1 | 12.9 | 67M | none listed |
| distilbert_sst2 | 9.1 | 67M | apache-2.0 |
| distilbert_v2 | 12.8 | 67M | ours |
| tfidf_lr | 11381.7 |  | ours |

## Accuracy (%) with and without test texts the model may have trained on

clapai: any match in its training collection. lxyuan: matches from tyqiangz/multilingual-sentiments, its training data.

| Model | Set | Test rows | Possibly seen | All rows | Unseen rows only |
|---|---|---|---|---|---|
| clapai | imdb | 600 | 251 | 97.3 | 97.1 |
| clapai | amazon23 | 900 | 1 | 95.2 | 95.2 |
| clapai | amazon23_mt | 8100 | 1 | 85.0 | 85.0 |
| clapai | tweets | 5280 | 4843 | 90.0 | 85.6 |
| clapai | amazon_multi | 5400 | 4923 | 96.3 | 91.7 |
| lxyuan | tweets | 5280 | 4424 | 74.1 | 75.9 |
| lxyuan | amazon_multi | 5400 | 1193 | 81.3 | 80.4 |

## Test texts found word for word in the clapAI training collection

| Set | Language | Seen | Of |
|---|---|---|---|
| amazon23 | en | 1 | 1200 |
| amazon23_mt | ar | 1 | 1200 |
| amazon_multi | de | 1183 | 1200 |
| amazon_multi | en | 811 | 1200 |
| amazon_multi | es | 1198 | 1200 |
| amazon_multi | fr | 1191 | 1200 |
| amazon_multi | ja | 1185 | 1200 |
| amazon_multi | zh | 1003 | 1200 |
| imdb | en | 334 | 800 |
| tweets | ar | 708 | 870 |
| tweets | de | 807 | 870 |
| tweets | en | 802 | 870 |
| tweets | es | 853 | 870 |
| tweets | fr | 846 | 870 |
| tweets | hi | 794 | 870 |
| tweets | it | 821 | 870 |
| tweets | pt | 735 | 870 |

Every other set and language had no matches.

| Source in the collection | Test texts matched |
|---|---|
| https://huggingface.co/datasets/tyqiangz/multilingual-sentiments | 7413 |
| https://huggingface.co/datasets/mteb/amazon_reviews_multi | 6375 |
| https://huggingface.co/datasets/iberbench/tass2020_sentiment_analysis | 538 |
| https://huggingface.co/datasets/scikit-learn/imdb | 334 |
| https://www.kaggle.com/datasets/bwandowando/grab-app-reviews-from-google-store | 1 |
| https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated | 1 |
| https://www.kaggle.com/datasets/fahdseddik/arabic-company-reviews | 1 |

## Full IMDb test set (24,801 reviews once exact duplicates are removed), cut at 0.5

| Model | Accuracy (%) |
|---|---|
| tfidf_lr | 89.5 (89.1 to 89.8) |
| distilbert_v1 | 91.2 (90.8 to 91.5) |
| distilbert_v2 | 91.4 (91.0 to 91.7) |

## distilbert_v2 training

| Epoch | Validation accuracy (%) | Minutes |
|---|---|---|
| 1 | 89.84 | 165.9 |
| 2 | 90.80 | 28.7 |
| 3 | 90.52 | 165.1 |

Best epoch on validation: 2.
