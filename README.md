# Car-Info

A project for extracting and analyzing users' comments on cars.

This repo includes the following five functions: 
- [Web crawler](web_crawler).
- [Text classification](#text_classification)
- Key-phrase extraction
- Sentiment analysis
- Web application integrating above three functions
- (plus) Word vector remapping to sentiment aware embedding

The following several sections will introduce these functions respectively, each of the section includes its own requirements, usage and notes.

Note that each functions mentioned corresponds to a directory in the repo, except that 'text classification' resides in 'text_clf/' and 'sent-conv-torch/'

## Web Crawler

Web crawler of different car-comment websites. All in `cralwer/crawler.py`, for detailed comments, see the code within.

### Requirement

requests, bs4, mysqldb

### Usage

Simply run:
```python
python crawler.py
```

It will first get all url corresponding to car-series, and store them info `url.json`, and request them one by one. The crawled data will be stored into mysql.

### Note

Car strucutrue: brand (宝马) -> series (车系) -> spec (品牌).
Considering the different structure in websites, some of websites do not distinguish btw specs. So only series is considered currently. 

I do not use scrapy or other well-structured api, so it's currently NOT sustainable, i.e., visited url are not recorded and it may suffers from unexpected problems. 

## Text Classification

### Introduction

The text classification of sentences is done in different ways: traditional machine learning techniques (in `text_clf/`), CNN (in `sent-conv-torch/`). 
It's a 12-class classification problem. Ten classes are meaningful ones (like *space*, *appearance*). one class named *neutral* means it does not conform to any actual aspect of car-comment, the other named *other* means the sentence says something about the car, but not conformed with our classification taximony (like noise info). 

Currently, the F1 of 12-class classification reaches 87% using CNN.

Also, we try another task: given a whole passage, try to *split* it into sentences with different classes. It's more difficult because some sentences are not indicative enough and require context to determine its class.


### Requirement

traditional machine learning (using python):

xlrd, xlwt, cPickle, h5py, MySQLdb, jieba, numpy, scipy, sklearn, nltk, matplotlib

CNN (using lua):

torch7, hdf5

### Architecture

- text_clf/car_preprocessing_tool.py: It offers useful tools to manipulate data crawled. Also the basic training models and testing tools.
- text_clf/car_train.py: It offers automatic training of multiple experiments. Using `matplotlib` to plot the results.
- text_clf/evalute.py: It offers ways to evalute the model, also predict passage function is included.
- text_clf/test_passage(2).py: These two files are actually test set for passage prediction. Data collected from xcar.
- sent-conv-torch/: CNN code via lua. Actually copied from repo [here](https://github.com/harvardnlp/sent-conv-torch/). Difference: our dataset is in `data/custom.*`

### Note


## Keyphrase Extraction

### Introduction

It collects comments of cars and extract key-phrase from sentences (like 空间大, 加速给力). We use rules of syntactic patterns (e.g., if we find the POS tag with pattern "n + adj", we extract it as a keyphrase). 

It reaches only 65% in F1.

We also tried several other ways (totally 4) to improve the rules, see `Main.py` for detail.

### Requirement

xlrd, xlwt, jieba 

### Notes

## Sentiment Analysise

## Web App

### Requirement

nodejs

### Usage

To run in foreground:
```bash
node carinfo.js # start
```


To run it in background, we need a nodejs package: `forever`:
```bash
forever start carinfo.js # start
forever stop carinfo.js # stop
```
