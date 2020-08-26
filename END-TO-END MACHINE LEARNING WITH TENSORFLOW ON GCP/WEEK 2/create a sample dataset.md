<h1> 2. Creating a sampled dataset </h1>

This notebook illustrates:
<ol>
<li> Sampling a BigQuery dataset to create datasets for ML
<li> Preprocessing with Pandas
</ol>


```python
!sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst
```


```python
# Ensure the right version of Tensorflow is installed.
!pip freeze | grep tensorflow==2.1
```


```python
# change these to try this notebook out
BUCKET = 'qwiklabs-gcp-02-090e06281b65'
PROJECT = 'qwiklabs-gcp-02-090e06281b65'
REGION = 'us-central1-c'
```


```python
import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
```


```bash
%%bash
if ! gsutil ls | grep -q gs://${BUCKET}/; then
  gsutil mb -l ${REGION} gs://${BUCKET}
fi
```

<h2> Create ML dataset by sampling using BigQuery </h2>
<p>
Let's sample the BigQuery data to create smaller datasets.
</p>


```python
# Create SQL query using natality data after the year 2000
from google.cloud import bigquery
query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""
```

There are only a limited number of years and months in the dataset. Let's see what the hashmonths are.


```python
# Call BigQuery but GROUP BY the hashmonth and see number of records for each group to enable us to get the correct train and evaluation percentages
df = bigquery.Client().query("SELECT hashmonth, COUNT(weight_pounds) AS num_babies FROM (" + query + ") GROUP BY hashmonth").to_dataframe()
print("There are {} unique hashmonths.".format(len(df)))
df.head()
```

    There are 96 unique hashmonths.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hashmonth</th>
      <th>num_babies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1403073183891835564</td>
      <td>351299</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3545707052733304728</td>
      <td>327823</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1451354159195218418</td>
      <td>334485</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-7773938200482214258</td>
      <td>343795</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-328012383083104805</td>
      <td>359891</td>
    </tr>
  </tbody>
</table>
</div>



Here's a way to get a well distributed portion of the data in such a way that the test and train sets do not overlap:


```python
# Added the RAND() so that we can now subsample from each of the hashmonths to get approximately the record counts we want
trainQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) < 3 AND RAND() < 0.0005"
evalQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) = 3 AND RAND() < 0.0005"
traindf = bigquery.Client().query(trainQuery).to_dataframe()
evaldf = bigquery.Client().query(evalQuery).to_dataframe()
print("There are {} examples in the train dataset and {} in the eval dataset".format(len(traindf), len(evaldf)))
```

    There are 13531 examples in the train dataset and 3157 in the eval dataset


<h2> Preprocess data using Pandas </h2>
<p>
Let's add extra rows to simulate the lack of ultrasound. In the process, we'll also change the plurality column to be a string.


```python
traindf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight_pounds</th>
      <th>is_male</th>
      <th>mother_age</th>
      <th>plurality</th>
      <th>gestation_weeks</th>
      <th>hashmonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.109009</td>
      <td>False</td>
      <td>20</td>
      <td>1</td>
      <td>36.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.000983</td>
      <td>True</td>
      <td>18</td>
      <td>2</td>
      <td>37.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.194990</td>
      <td>True</td>
      <td>21</td>
      <td>1</td>
      <td>38.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.938355</td>
      <td>True</td>
      <td>18</td>
      <td>1</td>
      <td>33.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.187070</td>
      <td>True</td>
      <td>23</td>
      <td>1</td>
      <td>40.0</td>
      <td>3095933535584005890</td>
    </tr>
  </tbody>
</table>
</div>



Also notice that there are some very important numeric fields that are missing in some rows (the count in Pandas doesn't count missing data)


```python
# Let's look at a small sample of the training data
traindf.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight_pounds</th>
      <th>mother_age</th>
      <th>plurality</th>
      <th>gestation_weeks</th>
      <th>hashmonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13523.000000</td>
      <td>13531.000000</td>
      <td>13531.000000</td>
      <td>13435.000000</td>
      <td>1.353100e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.235601</td>
      <td>27.302638</td>
      <td>1.035400</td>
      <td>38.600893</td>
      <td>2.762742e+17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.307310</td>
      <td>6.157826</td>
      <td>0.194538</td>
      <td>2.521605</td>
      <td>5.197212e+18</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.500449</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>-9.183606e+18</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.563162</td>
      <td>22.000000</td>
      <td>1.000000</td>
      <td>38.000000</td>
      <td>-3.340563e+18</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.312733</td>
      <td>27.000000</td>
      <td>1.000000</td>
      <td>39.000000</td>
      <td>-3.280124e+17</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.062305</td>
      <td>32.000000</td>
      <td>1.000000</td>
      <td>40.000000</td>
      <td>4.331750e+18</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13.000660</td>
      <td>50.000000</td>
      <td>4.000000</td>
      <td>47.000000</td>
      <td>8.599690e+18</td>
    </tr>
  </tbody>
</table>
</div>




```python
# It is always crucial to clean raw data before using in ML, so we have a preprocessing step
import pandas as pd
def preprocess(df):
  # clean up data we don't want to train on
  # in other words, users will have to tell us the mother's age
  # otherwise, our ML service won't work.
  # these were chosen because they are such good predictors
  # and because these are easy enough to collect
  df = df[df.weight_pounds > 0]
  df = df[df.mother_age > 0]
  df = df[df.gestation_weeks > 0]
  df = df[df.plurality > 0]
  
  # modify plurality field to be a string
  twins_etc = dict(zip([1,2,3,4,5],
                   ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)']))
  df['plurality'].replace(twins_etc, inplace=True)
  
  # now create extra rows to simulate lack of ultrasound
  nous = df.copy(deep=True)
  nous.loc[nous['plurality'] != 'Single(1)', 'plurality'] = 'Multiple(2+)'
  nous['is_male'] = 'Unknown'
  
  return pd.concat([df, nous])
```


```python
traindf.head()# Let's see a small sample of the training data now after our preprocessing
traindf = preprocess(traindf)
evaldf = preprocess(evaldf)
traindf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight_pounds</th>
      <th>is_male</th>
      <th>mother_age</th>
      <th>plurality</th>
      <th>gestation_weeks</th>
      <th>hashmonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.109009</td>
      <td>False</td>
      <td>20</td>
      <td>Single(1)</td>
      <td>36.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.000983</td>
      <td>True</td>
      <td>18</td>
      <td>Twins(2)</td>
      <td>37.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.194990</td>
      <td>True</td>
      <td>21</td>
      <td>Single(1)</td>
      <td>38.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.938355</td>
      <td>True</td>
      <td>18</td>
      <td>Single(1)</td>
      <td>33.0</td>
      <td>3095933535584005890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.187070</td>
      <td>True</td>
      <td>23</td>
      <td>Single(1)</td>
      <td>40.0</td>
      <td>3095933535584005890</td>
    </tr>
  </tbody>
</table>
</div>




```python
traindf.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight_pounds</th>
      <th>is_male</th>
      <th>mother_age</th>
      <th>plurality</th>
      <th>gestation_weeks</th>
      <th>hashmonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13526</th>
      <td>7.687519</td>
      <td>Unknown</td>
      <td>37</td>
      <td>Single(1)</td>
      <td>39.0</td>
      <td>-774501970389208065</td>
    </tr>
    <tr>
      <th>13527</th>
      <td>6.757168</td>
      <td>Unknown</td>
      <td>31</td>
      <td>Single(1)</td>
      <td>38.0</td>
      <td>-774501970389208065</td>
    </tr>
    <tr>
      <th>13528</th>
      <td>4.625298</td>
      <td>Unknown</td>
      <td>35</td>
      <td>Single(1)</td>
      <td>38.0</td>
      <td>-774501970389208065</td>
    </tr>
    <tr>
      <th>13529</th>
      <td>8.126239</td>
      <td>Unknown</td>
      <td>29</td>
      <td>Single(1)</td>
      <td>39.0</td>
      <td>-774501970389208065</td>
    </tr>
    <tr>
      <th>13530</th>
      <td>7.500126</td>
      <td>Unknown</td>
      <td>28</td>
      <td>Single(1)</td>
      <td>39.0</td>
      <td>-774501970389208065</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Describe only does numeric columns, so you won't see plurality
traindf.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight_pounds</th>
      <th>mother_age</th>
      <th>gestation_weeks</th>
      <th>hashmonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>26854.000000</td>
      <td>26854.000000</td>
      <td>26854.000000</td>
      <td>2.685400e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.236157</td>
      <td>27.300514</td>
      <td>38.604752</td>
      <td>2.761032e+17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.306863</td>
      <td>6.155911</td>
      <td>2.508681</td>
      <td>5.195149e+18</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.500449</td>
      <td>12.000000</td>
      <td>19.000000</td>
      <td>-9.183606e+18</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.563162</td>
      <td>22.000000</td>
      <td>38.000000</td>
      <td>-3.340563e+18</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.312733</td>
      <td>27.000000</td>
      <td>39.000000</td>
      <td>-3.280124e+17</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.062305</td>
      <td>32.000000</td>
      <td>40.000000</td>
      <td>4.331750e+18</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13.000660</td>
      <td>50.000000</td>
      <td>47.000000</td>
      <td>8.599690e+18</td>
    </tr>
  </tbody>
</table>
</div>



<h2> Write out </h2>
<p>
In the final versions, we want to read from files, not Pandas dataframes. So, write the Pandas dataframes out as CSV files. 
Using CSV files gives us the advantage of shuffling during read. This is important for distributed training because some workers might be slower than others, and shuffling the data helps prevent the same data from being assigned to the slow workers.



```python
traindf.to_csv('train.csv', index=False, header=False)
evaldf.to_csv('eval.csv', index=False, header=False)
```


```bash
%%bash
wc -l *.csv
head *.csv
tail *.csv
```

       6256 eval.csv
      26854 train.csv
      33110 total
    ==> eval.csv <==
    8.87581066812,True,37,Single(1),37.0,1088037545023002395
    6.37576861704,False,24,Single(1),39.0,1088037545023002395
    6.6248909731,True,17,Single(1),39.0,6392072535155213407
    7.50012615324,True,25,Single(1),39.0,-6782146986770280327
    7.68751907594,True,21,Single(1),40.0,-6244544205302024223
    7.68751907594,True,22,Single(1),37.0,-4740473290291881219
    4.9273315556999995,False,28,Twins(2),36.0,6392072535155213407
    8.75014717878,False,37,Single(1),41.0,270792696282171059
    7.3303702115,False,32,Single(1),38.0,-6782146986770280327
    7.87491199864,True,33,Single(1),36.0,-1891060869255459203
    
    ==> train.csv <==
    6.1090092800199995,False,20,Single(1),36.0,3095933535584005890
    6.0009827716399995,True,18,Twins(2),37.0,3095933535584005890
    6.1949895622,True,21,Single(1),38.0,3095933535584005890
    4.9383546688,True,18,Single(1),33.0,3095933535584005890
    7.1870697412,True,23,Single(1),40.0,3095933535584005890
    5.37486994756,True,18,Single(1),37.0,3095933535584005890
    8.75014717878,True,28,Single(1),40.0,3095933535584005890
    3.56267015392,True,26,Twins(2),31.0,3095933535584005890
    7.2201390805,False,39,Single(1),39.0,3095933535584005890
    7.31273323054,True,31,Single(1),41.0,3095933535584005890
    ==> eval.csv <==
    7.8484565272,Unknown,36,Single(1),43.0,270792696282171059
    8.68841774542,Unknown,34,Single(1),40.0,1088037545023002395
    6.0009827716399995,Unknown,25,Single(1),44.0,6365946696709051755
    6.686620406459999,Unknown,24,Single(1),38.0,-1639186255933990135
    7.99837086536,Unknown,28,Single(1),40.0,-6141045177192779423
    6.4595442766,Unknown,27,Single(1),39.0,2246942437170405963
    7.25100379718,Unknown,23,Single(1),38.0,8904940584331855459
    6.97322134706,Unknown,34,Single(1),41.0,6365946696709051755
    8.56275425608,Unknown,28,Single(1),40.0,-6141045177192779423
    7.2421853067,Unknown,19,Single(1),38.0,-4740473290291881219
    
    ==> train.csv <==
    8.50102482272,Unknown,29,Single(1),41.0,-774501970389208065
    6.4374980503999994,Unknown,23,Multiple(2+),38.0,-774501970389208065
    7.5287862473,Unknown,19,Single(1),45.0,-774501970389208065
    7.5618555866,Unknown,22,Single(1),43.0,-774501970389208065
    8.000575487979999,Unknown,36,Single(1),34.0,-774501970389208065
    7.68751907594,Unknown,37,Single(1),39.0,-774501970389208065
    6.7571683303,Unknown,31,Single(1),38.0,-774501970389208065
    4.62529825676,Unknown,35,Single(1),38.0,-774501970389208065
    8.12623897732,Unknown,29,Single(1),39.0,-774501970389208065
    7.50012615324,Unknown,28,Single(1),39.0,-774501970389208065


Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
