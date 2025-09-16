# Introduction
This is the code for the paper:

Next POI Recommendation for Long-tail Users via Spatial-Social Aware Contrastive Learning


# Implementation
## Requirements

Python >= 3.7  
Pytorch >= 1.2.0  
tqdm == 4.26.0
hnswlib==0.7.0
numba==0.53.1
numpy==1.25.2
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.6.2

## Datasets

Four datasets are included in `dataset` folder. 

| datasets               | Users  | POIs   | Check-ins| Friendships | Ave.length | Long-tail users | Time span | Sparsity |
| ---------------------- | ------ | -------- |------ | -------- |------ | -------- |------ | -------- |
| Gowalla-New York        | 7479 | 27881 | 168619 | 167002 |22.55 | 5168 |Apr.2008-Oct.2010 | 91.9\% |
| Gowalla-Los Angeles      | 7574 | 43559 | 232603 | 160435 |30.71 | 5241 |Apr.2008-Oct.2010 | 92.9\% |
| Brightkite-New York | 4294| 24022 |142582 | 64679 | 33.20 | 3511 | Feb.2009-Oct.2010 | 86.2\% |
| Brightkite-Los Angeles      | 4448 | 34803 |224871 | 67905 |50.56 | 3624 |Feb.2009-Oct.2010 | 85.5\% |



## Train Model

To train SSCL on four datasets, change to the `src` folder and run following command: 

```
python main.py -- data_name Gowalla-New York
python main.py -- data_name Gowalla-Los Angeles
python main.py -- data_name Brightkite-New York
python main.py -- data_name Brightkite-Los Angeles

```


## Evaluate Model (Gowalla-New York as example)

You can directly evaluate a trained model on test set by running:

```
python main.py --data_name Gowalla-New York  --do_eval
```

If you want the test set to only include long-tail users, then run the following command:

```
python main.py --data_name Gowalla-New York  --do_eval --tail_test
```
