# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP

# EXECUTIVE SUMMARY

# Problem Statement

Constellation Brands, a publicly traded company under the ticker `$STZ`, is an international producer and marketer of beverage alcohol. Their annual revenue is roughly `$8 Billion dollars` and the company spends almost `$800 million dollars` per year on advertising. With an advertising spend that has been growing steadily every year, Constellation spent nearly 300 percent more on advertising in 2020 than they did in 2014 (See Chart Below). 

**In this project the use of natural language processing (NLP) tools will be considered for the purpose of adding efficiency and effect to the marketing and advertising efforts of Constellation Brands. It is believed that NLP will compliment advertising spend growth. More explicitly, it is hypothesized that the analysis and predictive modeling capabilities of NLP regarding social media interactions will help determine marketing cues and insights and inform better marketing and advertising decisions. In light of this focus they have requested a preliminary report detailing insights and actions regarding a few new brands that are being considered for launch. Both brands will be geared toward do-it-yourself (DIY) home spirits makers who have an affinity for high-quality and handcrafted spirits.** 

Constellation Brands Company Overview
https://www.crunchbase.com/organization/constellation-brands
Constellation Brands LinkedIn
https://www.linkedin.com/company/constellation-brands/
**Constellation Brands advertising costs worldwide in the fiscal years 2014 to 2020**
![Screen%20Shot%202021-01-07%20at%2010.45.15%20AM.png](attachment:Screen%20Shot%202021-01-07%20at%2010.45.15%20AM.png)
* Source: https://www.statista.com/statistics/621085/constellation-brands-ad-spend/#:~:text=According%20to%20annual%20SEC%20reporting,million%20U.S.%20dollars%20into%20advertising.
* Constellation Brands company description and summary financial information 
* https://www.reuters.com/companies/STZ


To conduct this preliminary analysis I will used Reddit data scraped from the r/Homebrewing and r/Winemaking Subreddits. I cleaned the data for use in natural language preprocessing, exploratory data analysis, key insight creation, visualization creation and then finally conducted predictive modeling to determine whether my insights are gleaned from truly distinguisable characteristics.

The production model is the VotingClassifier() model fit with Support Vector Classifier, Ada Boost Classifier, Random Forest Classifier and Logistic Regression Classifier as the estimator hyperparameters. The production model was vectorized with a TFIDF Vectorizer with hyperparameters of Max_df = .08, Max_features = 300, ngram_range = (1,2) and stop_words = 'english'. 

# BASELINE VS PRODUCTION MODEL

THe production model scored 91.8% on balanced accuracy score whereas the baseline null model scored a 27.5%. The production appeared to perform significantly better than the baseline in that metric. 

Like F1 score it attempts to harmonize precision and recall which accounts for positive label identification. However, it adds a layer of accountability on top of F1 score and also takes into consideration how many negatives were labeled correctly. The balanced accuracy metric gives half its weight to correctly labeled positives and the other half of its weight to correctly labeled negatives. 

This analysis was presented to the executive marketing and data science teams of Constellation Brands who will be using it to decide whether the data will be helpful for marketing and advertising decisions. An action plan outlining next steps will be developed and executed provided that this project is successful in determining efficacy. Ultimately, the plan will extend programs and guidlines to executive and intermediary marketing managers for their use.


# Data Dictionary

https://pushshift.io/api-parameters/


# About API_beer_pull Notebook

This notebook takes in data from https://www.reddit.com/r/Homebrewing/  via the Push Shift API found at www.pushshift.io

This notebook consists of five steps of initial data retrieval and preprocessing for a Natural Language Processing Project. 

The project at large will intake data from two different subreddits (from www.Reddit.com), clean and process the data, explore the data for relationships, create and manipulate features and then model to produce a predictive classification model. 

This notebook will begin the process described above through the following steps: 

Step 1: Data Retrieval - Is the process of pulling the data from Reddit using the Pushshift API.

* Here code was written that identifies the url to call the respective subreddit.
* It sets parameters for the Pushshift API
* Loops through to create a new request using updated parameters so that each time new information is being pulled from the subreddit.
* Creates a dataframe to store each information pull. 
* Concatenates the dataframe. 


Step 2: Initial Analysis for Cleaning - Which will analyze unique values, indexes and summary info. 



Step 3: Cleaning - Will reduce the amount of features being processed, handle duplicate values, and eliminate null values. 

Step 4: Preliminary EDA - Will conduct an initial exploration of the data to analyze unique values in greater detail, initially explore the possibility of outliers in the data and look into features that may be interesting overall but that may not make it into the advanced EDA and modeling process. 

Step 5: Preparing DF for Text Analysis - This process will reduce feature columns to just those that will be used in advanced EDA and modeling. It will then save this file to CSV for the purpose of importing into another notebook focused on advanced EDA and feature creation. 



# About API Wine Notebook

This notebook takes in data from https://www.reddit.com/r/winemaking/ via the Push Shift API found at www.pushshift.io

This notebook consists of five steps of initial data retrieval and preprocessing for a Natural Language Processing Project. 

The project at large will intake data from two different subreddits (from www.Reddit.com), clean and process the data, explore the data for relationships, create and manipulate features and then model to produce a predictive classification model. 

This notebook will begin the process described above through the following steps: 

Step 1: Data Retrieval - Is the process of pulling the data from Reddit using the Pushshift API.

Step 2: Initial Analysis for Cleaning - Which will analyze unique values, indexes and summary info. 

Step 3: Preprocessing / Cleaning - Will reduce the amount of features being processed, handle duplicate values, and eliminate null values. 

Step 4: Preliminary EDA - Will conduct an initial exploration of the data to analyze unique values in greater detail, initially explore the possibility of outliers in the data and look into features that may be interesting overall but that may not make it into the advanced EDA and modeling process. 

Step 5: Preparing DF for Text Analysis - This process will reduce feature columns to just those that will be used in advanced EDA and modeling. It will then save this file to CSV for the purpose of importing into another notebook focused on advanced EDA and feature creation. 


# Preprocessing and EDA Notebook

This notebook takes in cleaned data that was pulled from r/Homebrewing and r/Winemaking subreddits, preprocesses it, executes and analyses exploratory data analysis and finishes off with a Spacy Scattertext plot. 


## Preprocessing

Cleaning Symbols/Numbers/Capital Letters

The code for this section has been adapted from this article in Towards Data Science and from class lectures. Articles contributions to this came in bits and pieces.
https://towardsdatascience.com/preprocessing-text-data-using-python-576206753c28

I streamlined the code into one function and added some other modifications.
This function takes in a df label and a series/column label, eliminates numbers, strips white space, lowercases characters, tokenizes the data, removes punctuations, sets changes to original series/column and keeps tokenized column.

## Stop Word Creation
inspired by these links:
* https://stackoverflow.com/questions/24386489/adding-words-to-scikit-learns-countvectorizers-stop-list
* https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe

* I combined this all into 1 function
* I decided to use both Count Vectorizer and NLTK stop words in addition to my own custom stop words found in EDA for the purpose of adding robustness. 

## Instantiating and Fitting CountVectorizers

Instantiating 2 seperate Count Vectorizers. One for Homebrewing DF and one for winemaking DF. I am doing this separately in order to analyze each df's word trends seperately.


## Frequency Vectorization

#### Top 25 Words in Homebrew and Winemaking Subreddits
* After stop words are removed
* After basic preprocessing with CountVectorizer

The code to achieve this was inspired by this site:
https://stackoverflow.com/questions/16078015/list-the-words-in-a-vocabulary-according-to-occurrence-in-a-text-corpus-with-sc

* I adapted the code to work in my custom function. 
* I can now easily do any amount of top words by adjusting n_words parameter. 



## Top 25 Word Charts
![Top 25 Word Homebrewing]('./data/01_beer.csv')

![Top 25 Word Winemaking]('./data/01_wine.csv')

### Analysis of Top 25 Word Charts

Not surprisingly the top word in Winemaking is 'wine' and the top word in Homebrewing is 'beer.' Also not surprising are other shared words in the top 25 such as fermentation, water and yeast. 

I think there are some distincive words in each of these lists that will make a prediciting model very successful. In the homebrewing list words like beer, brew, keg, hops and grain will add distinguishing capability to the model. In the winemaking list words like wine, grapes, primary and secondary will add distinguishing capability to the model.

### Outliers and Overly Frequent Words

After reviewing these charts it will likely be necessary to remove the terms 'wine', 'beer', 'brew' and 'brewing' from the corpus as these counts may make the model overpredictive to many of these features. 

In the case of words like 'yeast', 'fermentation', 'batch', 'amp', 'bottle', 'water', 'gallon', 'thanks', 'question', 'help' and 'recipe' it may be necessary to remove these words because they appear frequently in both models. 

[X] These words (All the mentioned words in this cell) are indeed removed later in the notebook. 


## TF-IDF Encoding

## IDF Values
The lower the IDF value of a word, the less unique it is to any particular document

Used this website for guidance on this process:
https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.X_DmeulKhTY

I used Tfidf Transformer instead of Tfidf Vectorizer.
Tf = term frequency
Tf-idf = term frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval, that has also found good use in document classification. 

The goal of using tf-idf instead of the raw frequencies of occurence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus. 

## Analysis TF-IDF 

The results here are not very surprising as the lowest weighted values are very similar to the top 25 most frequent words. I didn't include TF-IFD in any further steps because admittedly I didn't understand them very well when initially producing these results. In hindsight I think there is a possibility that TF-IDF may have some advantages as processed data. It is easy to calculate and would have saved me an extra step when I added more stop words because words appeared to frequently in both corpuses or had too much predictive capability. There are downsides to TF_IDF. It does not recognize the difference between plural and singular words and may incorrectly downplay the weightings of some words if they are represented by a good deal of both plural and singular nouns. 

"Although many new algorithms have come up in the recent past, the simple TF-IDF algorithm is still the benchmark for Query retrieval methods. But, the simple TF-IDF has its own limitations as well, like it fails to distinguish between a singular word and plural words. So, if suppose you search for ‘drug’, TF-IDF will not be able to equate the word ‘drug’ with ‘drugs’ categorizing each instead as separate words and slightly decreasing the word’s ‘Wd’ value."

* https://www.bluepiit.com/blog/using-tf-idf-algorithm-to-find-relevance-score-in-document-queries/#:~:text=%23%23%20Advantages%20of%20TF%2DIDF,relevant%20to%20a%20particular%20query

![Top 25 Word Homebrewing]('./data/01_beer.csv')

## Parts of Speech Chart Analysis

#### Nouns
* **NN means Noun, singular or mass.**
* **NNS is another form of Noun, plural.** 

Not surprising that it is the most common part of speech in both categories as nouns are a prominent part of all speech and especially when discussing tools used to produce things like wine and beer. 

#### Adjective
* **JJ mean Adjective**

another crucial part of every day language is found to be the 2nd most abundant part of speech used in both subreddits. 

#### Verbs
* **VBG means Verb gerund or present participle**
* **VBD means Verb, past tense**
* **VBP means Verb, 3rd person singular present.**

Verbs are also not suprisingly found amoung the most common words in both subredits. 

#### Final Thoughts of POS

I am disappointed that there are no glaring differences in POS of speech. However, what I do find interesting is the nearly identical numbers for both subreddits. The type of POS and the percentage of the respective POS in relation to total word count is almost identical in every category. This could be inspiration for further dives into POS research in groups that may not be so closely in common. It would be interesting to see if there can be some distinguishing characteristics. 

#### A Caveat

This POS analysis was done after stop words had been removed. This could have had an impact on the end results of the analysis. I think further research may also be necessary to compare POS before and after stop words are removed. If there is a difference it would be interesting to note if either subreddit used the stop words more or less than the other. This could have implications regarding command of language and perhaps culture of the different subreddits. 

## Spacy Scattertext

Spacy Scattertext in the case is a count frequency plot ditributed by both frequency and a weigting score that determines in which class the term should be weighted toward. 'Spacy Scattertext uses scaled f-score, which takes into account the category-specific precision and term frequency. While a term may appear frequently in both categories (Highand Lowrating), the scaled f-score determines whether the term is more characteristic of a category than others (High or Low rating).
For example, while the term park is frequent in both High and Low rating, the scaled f-score concludes park is more associated with High 0.90 than Low 0.10 rating. Thus, when a review includes the term park it is more characteristic of a High rating category.'

![Scattertext Plot]('./data/02_scattertext.html')

* https://towardsdatascience.com/analyzing-yelp-dataset-with-scattertext-spacy-82ea8bb7a60e

* Code derived from Spacy Documentation and Jason Kessler's (Creator of Scattertext) personal Github. 

https://github.com/JasonKessler/scattertext/blob/master/demo_category_frequencies.py

A medium article from Jason Kessler regarding Scattertext. 
https://medium.com/analytics-vidhya/visualizing-phrase-prominence-and-category-association-with-scattertext-and-pytextrank-f7a5f036d4d2

video by Jason Kessler explaining Scattertext
https://www.youtube.com/watch?v=H7X9CA2pWKo

Slide share presentation from Jason Kessler
https://www.slideshare.net/JasonKessler/natural-language-visualization-with-scattertext

*Scattertext documentation that only has half the code needed to deploy the model. :(
https://spacy.io/universe/project/scattertext


# Directory Structure

```
project_3
|__ code (folder)
    |__ 03_modeling.ipynb 
    |__ 02_eda_proprocessing.ipynb
    |__ 01_beer_api_pull_and_preprocessing.ipynb
    |__ 01_wine_api_pull_and_clean.ipynb
|__ data (folder)
    |__ 03_preprocessed_df.csv
    |__ 02_scattertext.html
    |__ 01_beer.csv
    |__ 01_wine.csv
|__ presentation_folder (folder)
|__README.md
    

