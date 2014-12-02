import math
import random
import numpy
from parse_movies_example import load_all_movies 
from nltk.corpus import stopwords
import P2

def get_top_words(wc, top_num = 10):
	top_words = {}
	for word in wc:
		if not word in stopwords.words():
			for decade in wc[word]:
				p = P2.p_x_given_y(wc, word, decade)
				p_min = 1

				for n_decade in wc[word]:
					if not n_decade == decade:
						p_min = min(P2.p_x_given_y(wc, word, n_decade), p_min)

				ratio = p/p_min
				# ratio = wc[word][decade]/min(wc[word].values())

				if decade in top_words:
					w = top_words[decade]
					if len(w) < top_num:
						top_words[decade][word] = ratio
					else:
						mw = min(w, key=w.get)
						if ratio > w[mw]:
							del top_words[decade][mw]
							top_words[decade][word] = ratio
				else:
					top_words[decade] = {}
					top_words[decade][word] = wc[word][decade]
	return top_words


# Main Function
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
	test_sample_size = 1000

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
	wc = P2.all_x_all_y(years_train, plots_train)

	# 3a. Determine Top 10 words of each decade
	top_words = get_top_words(wc, 10)
	for decade in top_words:
		print decade, ':', top_words[decade].keys()

	# 3b. Test classifer without top 100 words
	top_words = get_top_words(wc, 100)
	correct_count_wo = [0.]*bin_num
	correct_count_w = [0.]*bin_num
	for i, plot in enumerate(plots_test):
		predicted_decade_wo, decade_probs_wo = P2.predict_decade(wc, years_train, plot=plot, skip_words=top_words)
		predicted_decade_w, decade_probs_w = P2.predict_decade(wc, years_train, plot=plot)

		actual_year = years_test[i]
		for i, decade in enumerate(decade_probs_wo):
			correct_count_wo[i] += 1. if decade[0] == actual_year else 0.
		for i, decade in enumerate(decade_probs_w):
			correct_count_w[i] += 1. if decade[0] == actual_year else 0.

	print "Accuracy on test (with words): ", correct_count_w[0]/len(plots_test)
	print "Accuracy on test (without words): ", correct_count_wo[0]/len(plots_test)











