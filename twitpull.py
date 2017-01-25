from twitter import *
import pandas as pd
from datetime import datetime as dt
import time
import glob
import os
import sys
import json
import pdb
import twitter

def oauth_login():
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://dev.twitter.com/docs/auth/oauth for more information 
    # on Twitter's OAuth implementation.
    
    CONSUMER_KEY = 'jjyuA4XFZoVQSLqp8lBm9UIyE'
    CONSUMER_SECRET = 'AWHlNkSl07i8rCmNvGVMDpm6oQZQ7Xo3vH6yIdRMtAZU8oOxsa'
    OAUTH_TOKEN = '70013325-5TffKjTZcmwrwdUrDflTb63YrI5LHtNzRrwweonXi'
    OAUTH_TOKEN_SECRET = 'HZ0Ih5NvrZPfBYwidpwGDBil8Ovc2zEneSgxMF0XqyyC3'
    
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
    
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api

# Twitter search function to get a number of tweets using a search query
def TwitterSearch(twitterApi, query, approxCount = 3000, **kw):
    searchResults = twitterApi.search.tweets(q= query, count=100, **kw)
    statuses = searchResults['statuses']
    while len(statuses) < approxCount:
        try:
            nextResults = searchResults['search_metadata']['next_results']
        except KeyError, e:
            break
        print str(len(statuses)) + ' results have been downloaded from approximately ' \
            + str(approxCount)
        kwargs = dict([ kv.split('=') for kv in nextResults[1:].split("&") ])
        nextResults = twitter_api.search.tweets(**kwargs)
        statuses += nextResults['statuses'] # cool append notation
        print 'A total of ' + str(len(statuses)) + ' have been downloaded'
    return statuses

dir=('./')

twitter_api = oauth_login()

q = sys.argv[1]

statuses = TwitterSearch(twitter_api, q, language = 'en', approxCount = 200)
#status2 = TwitterSearch(twitter_api, q, language = 'en', approxCount = 2000)

status_id = [ status['id']
              for status in statuses ]
name = [ status['user']['name']
         for status in statuses ]
screen_name = [ status['user']['screen_name']
                for status in statuses ]
status_text = [ status['text']
                for status in statuses ]
location = [ status['user']['location']
             for status in statuses ]
geo = [ status['geo']
        for status in statuses ]
time_zone = [ status['user']['time_zone']
              for status in statuses ]
friend_count = [ status['user']['friends_count']
                 for status in statuses ]
follower_count = [ status['user']['followers_count']
				   for status in statuses ]
tmstamp = [ status['created_at']                 
            for status in statuses ]
retweet_ct = [ status['retweet_count']
              for status in statuses ]
source = [status['source']
            for status in statuses ]
place = [status['place']
            for status in statuses ]

data = {'status_id' : status_id,
        'name' : name,
        'screen_name' : screen_name,
        'status_text' : status_text,
        'tmstamp' : tmstamp,
        'time_zone' : time_zone,
        'location' : location,
        'geo' : geo,
        'friend_count' : friend_count,
        'follower_count' : follower_count,
        'retweet_count' : retweet_ct,
        'source' : source,
        'place' : place}
df = pd.DataFrame(data)
df.to_csv(dir + 'tweets' + '.csv', encoding = 'utf-8')



