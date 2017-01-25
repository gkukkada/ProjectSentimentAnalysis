from __future__ import division
import os  # operating system commands
import nltk
import re  # regular expressions
import pandas as pd  # DataFrame structure and operations
import numpy as np  # arrays and numerical processing
import matplotlib.pyplot as plt  # 2D plotting
import statsmodels.api as sm  # logistic regression
import statsmodels.formula.api as smf  # R-like model specification
from patsy import dmatrices  # translate model specification into design matrices
from sklearn import svm  # support vector machines
from sklearn.ensemble import RandomForestClassifier  # random forest
from langdetect import detect
from nltk.corpus import PlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer as DV
from nltk.collocations import *
import collections
from nltk.util import ngrams
import pdb
import pygal
from collections import Counter
from prettytable import PrettyTable
from prettytable import from_csv
from BeautifulSoup import BeautifulSoup as bs
from nltk.tokenize import word_tokenize
from nltk import *
from nltk.stem.snowball import SnowballStemmer
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsIC
from sklearn.grid_search import GridSearchCV
from sklearn_pandas import DataFrameMapper, cross_val_score
from multiprocessing import Pool
import multiprocessing # try to incorporate multiprocessing for slow lookups
from sklearn import pipeline
from sklearn import cross_validation
from datetime import datetime
import time
import HTMLParser
# class for debugging errors
class MyObj(object):
    def __init__(self, num_loops):
        self.count = num_loops

    def go(self):
        for i in range(self.count):
            pdb.set_trace()
            print i
        return
# This function will remove unwanted spaces, characters and format lines that will closely match our lexicon(s)
def clean_tweet(tweet):

    more_stop_words = ['rt', 'cant','didnt','doesnt','dont','goes','isnt','hes','shes','thats','theres',\
					  'theyre','wont','youll','youre','youve', 'br', 've', 're', 'vs', 'goes','isnt',\
					  'hes', 'shes','thats','theres','theyre','wont','youll','youre','youve', 'br',\
                      've', 're', 'vs', 'this', 'i', 'get','cant','didnt','doesnt','dont','goes','isnt','hes',\
					  'shes','thats','theres','theyre','wont','youll','youre','youve', 'br', 've', 're', 'vs']
    
    # start with the initial list and add the additional words to it.
    stoplist = nltk.corpus.stopwords.words('english') + more_stop_words

    # define list of codes to be dropped from document
    # carriage-returns, line-feeds, tabs
    codelist = ['\r', '\n', '\t']

    # insert a space at the beginning and end of the tweet
    # tweet = ' ' + tweet + ' '

    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    tweet = re.sub('http[^\\s]+',' ', tweet)
    tweet = re.sub(r"\[", '', tweet)
    tweet = re.sub(r"\]", '', tweet)
    tweet = re.sub(r"'rt", '', tweet)
    tweet = re.sub(r'\'', '', tweet)
    tweet = re.sub(r'\'\,', '', tweet)
    tweet = re.sub(r'\,\'', '', tweet)
    tweet = re.sub('rt[^\\s]+', '', tweet)
    tweet = re.sub(r"' ,", '', tweet)
    tweet = re.sub(r"\' ,", '', tweet)
    tweet = re.sub(r", ',',", '', tweet)
    tweet = re.sub(r"\,", '', tweet)
    tweet = re.sub(r"\, \"\'\"\,", '', tweet)
    tweet = re.sub(r"\, \"\' \,\"\,", '', tweet)
    tweet = re.sub(r"\, \"\'\ \,\"\,", '', tweet)
    tweet = re.sub(r"\,\ \"\'\"\,", '', tweet)
    tweet = re.sub(r"\,", '', tweet)
    tweet = re.sub(r"\"", '', tweet)
    tweet = re.sub(r"\'", '', tweet)
    tweet = re.sub(r"\'\,", '', tweet)
    tweet = re.sub(r'"', '', tweet)
    tweet = re.sub(",", '', tweet)
    
    temp_tweet = re.sub('[^a-zA-Z]', ' ', tweet)     # replace non-alphanumeric with space                              
    html_parser = HTMLParser.HTMLParser()
    tweet = html_parser.unescape(tweet)
    # temp_tweet = re.sub('\d', '  ', temp_tweet)

    for i in range(len(codelist)):
        stopstring = ' ' + codelist[i] + '  '
        temp_tweet1 = re.sub(stopstring, '  ', temp_tweet)
       
    # convert uppercase to lowercase
    temp_tweet = temp_tweet1.lower()   

    # replace single-character words with space
    temp_tweet = re.sub('\s.\s', ' ', temp_tweet)

    # replace selected character strings/stop-words with space
    for i in range(len(stoplist)):
        stopstring = ' ' + str(stoplist[i]) + ' '
        temp_tweet = re.sub(stopstring, ' ', temp_tweet)

    # replace multiple blank characters with one blank character
    temp_tweet = re.sub('\s+', ' ', temp_tweet) 

    return(temp_tweet)

# This, and the next function are a generic function which can create a frequency histogram of terms/words in the corpus(es)
def word_freq_dist(tweet_words):
    word_freq = dict()

    for words in tweet_words:
        if (word_freq.has_key(words)):
            # This word already exists in the frequency dictionary, bump the count
            word_freq[words] += 1
        else:
            # insert the word into the frequency dictionary
            word_freq[words] = 1

    return word_freq

def plotMostFrequentWords(words, plot_file_name, plot_title):

    # compute a frequency distribution dictionary.
    word_freq_dict = word_freq_dist(words)

    # convert the dictionary into a sorted list.
    # lambda signifies an anonymous function. In this case, this function 
    # takes the single argument x and returns x[1] (i.e. the item at index 1 in x).
    # The values in the dictionary are in column [1]. lamda x: x[1] will sort the 
    # dictionary by the values of each entry within the dictionary; reverse=True
    # tells sorted to sort from largest to smallest instead of the default which is
    # smallest to largest.
    # see: http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    
    freq_sorted_list = list()
    freq_sorted_list = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    #freq_sorted_list[0][0] gives most frequent word
    #freq_sorted_list[0][1] gives count for that word

    # print the top 15 words and their counts
    print('Top 15 words in terms of frequency: ')

    max_num = 15
    if (len(freq_sorted_list) < max_num):
        max_num = len(freq_sorted_list)

    for i in range(max_num):
        print('index: ', i, ' words: ', freq_sorted_list[i][0],
              ' count: ', freq_sorted_list[i][1])

    #print('\n')
    
    # convert the sorted list into a data frame so that we can plot
    freq_sorted_df =  pd.DataFrame(freq_sorted_list, columns=['Word', 'Count'])

    #print freq_sorted_df.head()

    freq_sorted_word_chart = freq_sorted_df[:15].plot(kind='bar', x='Word', y='Count',
                                                      title = plot_title)
    
    freq_sorted_word_chart.set_ylabel('Word Count')
    freq_sorted_word_chart.set_xlabel('')
    freq_sorted_word_chart.legend().set_visible(False)
    plt.savefig((plot_file_name), bbox_inches = 'tight', edgecolor='b', orientation='landscape', papertype=None, format=None, 
    transparent=True)  # plot to file

    # clear the figure
    plt.clf()
    
    return freq_sorted_list
# Time the script; probably need to add Multiprocessing Module to speed up
startTime = time.time()

#Define directory and file with all tweets to be used, read it in from source
dir=('./')
twitter_df=pd.read_csv(dir + 'tweets.csv')  #This is the Twitter Feeds data pulled from the API !!!!
# This is a method for finding key terms (qualitatively defined) in the tweets; it will later be used in a regression to predict Retweet Count

twitter_df['pricing'] = twitter_df.status_text.str.contains("pricing|price|cost")
twitter_df['free'] = twitter_df.status_text.str.contains("free")
twitter_df['promo'] = twitter_df.status_text.str.contains("promo|promotion|offer")
twitter_df['service'] = twitter_df.status_text.str.contains("service")
twitter_df['fast'] = twitter_df.status_text.str.contains("fast")
twitter_df['slow'] = twitter_df.status_text.str.contains("slow")
twitter_df['movie_game'] = twitter_df.status_text.str.contains("movie|game|played|playing")
twitter_df['texting'] = twitter_df.status_text.str.contains("texting|messaging")


twitter_df['pricing'] = twitter_df['pricing']*1
twitter_df['free'] = twitter_df['free']*1
twitter_df['promo'] = twitter_df['promo']*1
twitter_df['service'] = twitter_df['service']*1
twitter_df['fast'] = twitter_df['fast']*1
twitter_df['slow'] = twitter_df['slow'] *1
twitter_df['movie_game'] = twitter_df['movie_game']*1
twitter_df['texting'] = twitter_df['texting']*1
#apply the tweet cleaning function from above

print 'dataframe: ', twitter_df.head()
#clean up all tweets
review_tweets = twitter_df.status_text  
cleaned_tweets = []

for line in review_tweets:
    cleaned_tweet = clean_tweet(line)    
    cleaned_tweets.append(cleaned_tweet)
print 'cleaned_tweets created'
location = str(twitter_df.location)
locations = re.sub(r'[^\x00-\x7F]+',"", location)

#apply tokenization, lemmatization, bigrams, and stemmer to look at different sequences of terms; this will determine the best features
tokens = [word for sent in nltk.sent_tokenize(str(cleaned_tweets)) for word in nltk.word_tokenize(sent)]
#for token in sorted(set(tokens))[:30]:
   # print 'tokens are: ' + token + ' [' + str(tokens.count(token)) + ']'

lemmatizer = nltk.WordNetLemmatizer()
lemm_tokens = [lemmatizer.lemmatize(t) for t in tokens]
#for token in sorted(set(lemm_tokens))[:30]:
#    print 'lemm are: ' + token + ', [' + str(lemm_tokens.count(token)) + ']'


bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
bigramslist = re.sub(',', '', str(bigrams))
#print 'bigrams: ', bigramslist[:10]

stemmer = SnowballStemmer("english")
stemmed_tokens = [stemmer.stem(t) for t in tokens]
#for token in sorted(set(stemmed_tokens))[:30]:
#    print 'stems are: ' + token + ' [' + str(stemmed_tokens.count(token)) + ']'

# n = 3
# trigrams = ngrams(str(tokens).split(), n)
# for grams in sorted(set(trigrams))[:20]:
#     print 'tri grams are:', grams

trigrams = [" ".join(pair) for pair in nltk.trigrams(tokens)]
trigramslist = re.sub(',', '', str(trigrams))
#print 'trigrams: ', trigramslist[:10]

# Use Python collection for counting frequency OF USERS
twitter_df.screen_name = twitter_df.screen_name.str.replace(r'[^\x00-\x7F]+', '').astype('str') 
user_count = Counter()
retweet_count = Counter()
for index, row in twitter_df.iterrows():
    user_count[row['screen_name'] ] += 1
    retweet_count[row['retweet_count'] ] += 1

# Prepare the svg Plot

barplot = pygal.HorizontalBar(style=pygal.style.SolidColorStyle )

topnum = 10
for i in range(topnum):
    barplot.add( user_count.most_common(topnum)[i][0], \
              [ { 'value': user_count.most_common(topnum)[i][1], \
                  'label': user_count.most_common(topnum)[i][0]} ] )
barplot.config.title = barplot.config.title= "Top " + str(topnum) + " Most Prolific Tweeters"
barplot.config.legend_at_bottom=True
barplot.render_to_file("Top_Tweeters.svg")


src=Counter(twitter_df.source)

# Convert the "Counter" container to Pandas dataframe for easy manipulation
frame = []
for i,j in src.iteritems():
    match=re.match(r"^.*\">(.*)\<.*$", str(i))
    if match:    
        frame.append( [j, match.group(1)])
    else:
        frame.append([j, ''])
sourcedf = pd.DataFrame(frame, columns=["COUNT", "SOURCE"])

# A lookup table to normalize the data in the containers we want
#   - all iOS Platforms (iPad, iPhone et. al. goes into iOS etc.)
sourcelookup = { "web": "Web",                              "Twitter for iPhone": "iOS",
                "Twitter for Android": "Android",           "TweetDeck": "TweetDeck",
                "Tweetbot for iOS": "iOS",                  "Twitter for iPad": "iOS",
                "Twitter for Mac": "Mac",                   "Tweetbot for Mac": "Mac",
                "Twitter for Android Tablets": "Android",   "Twitterrific": "iOS",
                "iOS": "iOS",                               u"Plume\xa0for\xa0Android": "Android",
                "YoruFukurou": "Mac",                       "TweetCaster for Android": "Android",
                "Guidebook on iOS": "iOS",                  "Twitter for Android": "Android",
                "UberSocial for iPhone": "iOS",             "Twitterrific for Mac": "Mac"
                }


# A helper function for looking up the table defined above
def translate(txt):
    try:
        return sourcelookup[txt]
    except KeyError:
        return "Other"

# Create a new column with normalized field
sourcedf['NSOURCE']=sourcedf.SOURCE.apply(lambda x: translate(x))
twitter_df['source']=sourcedf['NSOURCE']
# Groupby the normalized field "NSOURCE"
grouped = sourcedf.groupby(by=["NSOURCE"])

# Create the chart (PieChart) of device source used by Twitter Users
chart = pygal.Pie( style=pygal.style.SolidColorStyle )

for i in grouped.groups.iteritems():
    chart.add( i[0], grouped.get_group(i[0]).COUNT.tolist() )

chart.config.title="Twitter Source for PyData-SV Users"
chart.render_to_file('pie_chart_twitter_usersource.svg')


############# This will read in the unigrams list of Positive and Negative lexicons
positive_list = PlaintextCorpusReader(dir, 'unigrams-pos.txt')   

negative_list = PlaintextCorpusReader(dir, 'unigrams-neg.txt')   

positive_words = positive_list.words()
negative_words = negative_list.words()


# define bag-of-words dictionaries 
def bag_of_words(words, value):
    return dict([(word, value) for word in words])
positive_scoring = bag_of_words(positive_words, 1)
negative_scoring = bag_of_words(negative_words, -1)
scoring_dictionary = dict(positive_scoring.items() + negative_scoring.items())
"""
for k, v in scoring_dictionary.items():
     print k, v 
scoring_dictionary=set(scoring_dictionary)
"""
# scores are -1 if in negative word list, +1 if in positive word list
# and zero otherwise. We use a dictionary for scoring.

score = [0] * len(tokens)

for word in range(len(tokens)):
    if tokens[word] in scoring_dictionary:
        score[word] = scoring_dictionary[tokens[word]]

#define a corpus for later use
corp=nltk.Text(tokens)
print "----------------------------------------"
print('Average Sentiment Probability`:')  
sentscore = sum(score) / len(tokens)
if sentscore < 0 :
	print "Probability is Negative:",sentscore
	print "-----------------------------------------"
else:
	print "Probability is Positive:",sentscore
	print "--------------------------------------------"

#print 'sum score', sum(score)
#print 'len tokens', len(tokens)
#-0.141606706372 is from 1 run of Twitter feeds
# sum score -32906
# len tokens 232376

# identify the most frequent positive words (features to be used later for modeling)

positive_words_in = nltk.FreqDist(w for w in positive_words)
word_features_p = positive_words_in.keys()
negative_words_in = nltk.FreqDist(w for w in negative_words)
word_features_n = negative_words_in.keys()

def count_positive(token):    
    positive_w_in = []
    positive_w_in = [w for w in token if w in word_features_p]
    return positive_w_in

def count_negative(token):  
    negative_w_in = [] 
    negative_w_in = [w for w in token if w in word_features_n]
    return negative_w_in

positive_w_in = count_positive(tokens)
negative_w_in = count_negative(tokens)

print 'negative_w_in'
print type(negative_w_in)
print negative_w_in[:15]

count = Counter([w for w in positive_w_in])
count2 = Counter([w for w in negative_w_in])

pos = []
for i,j in count.iteritems():
    if j > 1:
        pos.append([j, i])

# Prepare the positive terms found histogram
posf= pd.DataFrame(pos, index=None, columns=['Count', 'Word'])
posf['Word'] = posf['Word'].str.replace('[^\w\s]','')
posf['Word'] = posf['Word'].str.replace('http', '')
posf.sort(columns="Count", inplace=True, ascending=False)
pos_chart = posf[:15].plot(kind='bar', x='Word', y='Count', title = 'Top Positive Words')
pos_chart.set_ylabel('Word Count')
pos_chart.set_xlabel('')
pos_chart.legend().set_visible(False)
plt.savefig(dir + 'pos.png', bbox_inches = 'tight', edgecolor='b', orientation='landscape', papertype=None, format=None, transparent=True)

neg = []
for i,j in count2.iteritems():
    if j > 1:
        neg.append([j, i])

# Prepare the negative terms found histogram

negf= pd.DataFrame(neg, index=None, columns=['Count', 'Word'])
negf['Word'] = negf['Word'].str.replace('[^\w\s]','')
negf['Word'] = negf['Word'].str.replace('http', '')
negf.sort(columns="Count", inplace=True, ascending=False)
neg_chart = negf[:15].plot(kind='bar', x='Word', y='Count', title = 'Top Negative Words')
neg_chart.set_ylabel('Word Count')
neg_chart.set_xlabel('')
neg_chart.legend().set_visible(False)
plt.savefig(dir + 'neg.png', bbox_inches = 'tight', edgecolor='b', orientation='landscape', papertype=None, format=None, transparent=True)

# Plot the freq dist for the full corpus, when adjusted for lemmatization 
# plot a bar chart for top words in terms of counts

print('Get the top 15 words: ')
full_plot_file_name = dir + 'full_review_word_count.png'
plot_title = 'full_review_word_count'
full_sort = plotMostFrequentWords(lemm_tokens, full_plot_file_name, plot_title)


# Plot the freq dist for the bigrams   
# plot a bar chart for top words in terms of counts

print('Get the top 15 bigrams: ')
neg_plot_file_name = dir + 'bigrams_count.png'
plot_title = 'bigrams_count'
negative_sort = plotMostFrequentWords(bigrams, neg_plot_file_name, plot_title)

twitter_df.to_csv(dir + 'twitter_df.csv', index=False)



# The next section is using a manually coded (0=not positive, 1=positive) sample to train the greater dataset of 
# tweets. The best classifier based on precision/recall/confusion matrix for probaility. 


#Finally do the modeling using classification models and predict sentiment
# sample=pd.read_csv(dir + 'sample_tweets_coded.csv')
# end_df=pd.merge(new_twitter_df, sample, how='left', right_index=True,left_index=True, on=None)

# # vectorize tweets for machine learning and remove stopwords
# vectorizer = CountVectorizer(min_df=1, stop_words='english')
# vector_data = vectorizer.fit_transform(end_df['cleaned_tweets'])
# # select only hand scored tweets for model training/evaluation
# scored_data = vector_data[end_df[end_df['Score'].isnull() == False].index]
# # create testing/training sets
# x_train, x_test, y_train, y_test = cross_validation.train_test_split(scored_data,
# end_df[end_df['Score'].isnull() == False]['Score'],
# test_size = 0.2, random_state = 0)
# print end_df.summary()

# # logistic regression classifier
# lr_clf = LogisticRegression()
# lr_clf = lr_clf.fit(x_train, y_train)
# lr_predicted = lr_clf.predict(x_test)
# # print classification report
# target_names = ['not postive','positive']
# print 'Logistic Regression Classification Report:'
# print (classification_report(y_test, lr_predicted, target_names = target_names))

# # support vector machine classifier
# from sklearn.linear_model import SGDClassifier
# svm = SGDClassifier()
# svm = svm.fit(x_train, y_train)
# svm_predicted = svm.predict(x_test)
# print 'Support Vector Machine Classification Report:'
# print (classification_report(y_test, svm_predicted, target_names = target_names))
"""
# # naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
nb_clf = MultinomialNB()
nb_clf = nb_clf.fit(x_train, y_train)
nb_predicted = nb_clf.predict(x_test)
print 'Naive Bayes Classification Report:'
print (classification_report(y_test, nb_predicted, target_names = target_names))

# # decided to use the output from the logistic regression
# # append results to data frame and save
end_df['predicted_sentiment'] = lr_clf.predict(vector_data)


# Now looking at the predicted sentiment probabilities
end_df['positive_probability'] = lr_clf.predict_proba(vector_data)[:,1]

end_df['negative_probability'] = lr_clf.predict_proba(vector_data)[:,0]

end_df.to_csv(dir+'final_data120214.csv', index=False)   

print "time took to output : ",time.time() - startTime
"""
