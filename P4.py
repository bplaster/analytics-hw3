import math
import random
import numpy as np
from parse_movies_example import load_all_movies 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import P2
import re

if __name__ == '__main__':

	# Get set of all movies
	all_movies = list(load_all_movies("plot.list.gz"))
	random.shuffle(all_movies)
	years, plots, titles = [], [], []

	for movie in all_movies:
		years.append(movie['year'])
		plots.append(movie['summary'])
		titles.append(movie['title'])

	min_year, max_year, bin_num = P2.year_stats(years)

	# Get uniform subset of movies
	years_train, plots_train, titles_train = [], [], []
	years_test, plots_test, titles_test = [], [], []
	year_count_train = [0]*bin_num
	year_count_test = [0]*bin_num
	train_sample_size = 6000
	test_sample_size = 10

	# Create uniformly distributed training and test sets
	for i, year in enumerate(years):
		bin = int((year - min_year)/10)
		if year_count_train[bin] < train_sample_size: 
			year_count_train[bin] += 1
			years_train.append(year)
			plots_train.append(plots[i])
			titles_train.append(titles[i])
		elif year_count_test[bin] < test_sample_size: 
			year_count_test[bin] += 1
			years_test.append(year)
			plots_test.append(plots[i])
			titles_test.append(titles[i])

	# Extract features
	vec = CountVectorizer(tokenizer = P2.norm_words, encoding = 'latin-1')
	data = vec.fit_transform(plots_train)

	# Train classifier
	clf = MultinomialNB()
	clf.fit(data, years_train)

	#Test classifier
	correct_count = 0.
	years_test = np.array(years_test)
	predicted_decade = clf.predict(vec.transform(plots_test))
	for i,decade in enumerate(predicted_decade):
		if decade == years_test[i]:
			correct_count += 1.

	#print predicted_decade
	#print years_test		
	#print "length of predicted_decade: ", len(predicted_decade)		
	#print "length of years_test: ", len(years_test)
	print "Accuracy on test: ", correct_count/len(plots_test)
	

