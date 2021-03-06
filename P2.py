from parse_movies_example import load_all_movies 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords
import numpy as np
import operator
import math
import random
import re
 
# Returns statistics about years
def year_stats(years):
	min_y = min(years)
	max_y = max(years)
	bin_n = int((max_y - min_y) / 10) + 1
	return min_y, max_y, bin_n

# Returns list of unique words from string
def norm_words(words, exclude_stopwords=False):
	pattern = re.compile(r'\W+')
	word_list = []
	add_word = False
	for w in pattern.split(words):
		w = norm_word(w)
		if not w == '' and not w in word_list:
			add_word = True
			if exclude_stopwords:
				if w in stopwords.words('english'):
					add_word = False
			if add_word:
				word_list.append(w)
	return word_list

# Returns lower case word without special characters
def norm_word(word):
	word = ''.join(c.lower() for c in word if c.isalnum())
	return word

# Plots histogram over years
def hist_plot(years, title, fname):
	min_year, max_year, bin_num = year_stats(years)
	hist = plt.hist(years, bins=bin_num, range=[min_year-5, max_year+5], normed=True)
	plt.title(title)
	plt.xlabel('Year')
	plt.ylabel('P(Y)')
	plt.xticks(np.arange(min_year, max_year + 10, 10))
	plt.savefig(fname)
	#plt.show()
	plt.close('all')

def hist_bar_plot(probs, title, fname):
	min_year, max_year, bin_num = year_stats(years)
	ind = []
	ht = []
	width = 10
	total = 0
	for prob in probs:
		ind.append(prob[0])
		ht.append(prob[1])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	hist = ax.bar(ind, ht, width)

	plt.title(title)
	plt.xlabel('Year')
	plt.ylabel('log(P(Y))')
	plt.xticks(np.arange(min_year+5, max_year + 15, 10), np.arange(min_year, max_year + 10, 10))
	plt.savefig(fname)
	#plt.show()
	plt.close('all')

# Returns list of all years with given word in plot
def y_given_x(years, plots, word):
	y_g_x = []
	word = norm_word(word)
	for i, plot in enumerate(plots):
		if word in norm_words(plot, exclude_stopwords=False):
			y_g_x.append(years[i])
	return y_g_x

# Returns dictionary of all words and count in all years
def all_x_all_y(years, plots):
	wc = {}
	dw = {}
	for i, year in enumerate(years):
		for word in norm_words(plots[i], exclude_stopwords=False):
			if word in wc:
				if year in wc[word]: 
					wc[word][year] += 1.
				else:
					wc[word][year] = 1.
			else:
				wc[word] = {}
				wc[word][year] = 1.

			if year in dw:
				if word in dw[year]: 
					dw[year][word] += 1.
				else:
					dw[year][word] = 1.
			else:
				dw[year] = {}
				dw[year][word] = 1.

	return dw, wc

# Returns probability of word in given year
def p_x_given_y(wc, word, year, y_count):
	p = 0.00001
	word = norm_word(word)
	if word in wc:
		if year in wc[word]:
			p = wc[word][year]/y_count
	#print word, p
	return p	

# Returns sorted tuple of movie prediction for each year for given plot
def movie_decade_probs(wc, dw, years, train_sample_size, plot, skip_words=None):
	min_year, max_year, bin_num = year_stats(years)
	decade_probs = {}
	for year in range(min_year, max_year + 10, 10):
		p = 0
		# movie_words = norm_words(plot, exclude_stopwords=True)
		# decade_words = dw[year]
		# for word in decade_words:
		# 	if skip_words and year in skip_words:
		# 		if word in skip_words[year]:
		# 			continue
		# 	if word in movie_words:
		# 		p += math.log(p_x_given_y(wc, word, year, train_sample_size),10)
		# 	else:
		# 		p += math.log(1 - p_x_given_y(wc, word, year, train_sample_size),10)

		for word in norm_words(plot, exclude_stopwords=True):
			if skip_words and year in skip_words:
				if word in skip_words[year]:
					continue
			p += math.log(p_x_given_y(wc, word, year, train_sample_size),10)
		decade_probs[year] = p
	sorted_probs = sorted(decade_probs.items(), key=operator.itemgetter(1), reverse=True)
	return sorted_probs

# Returns year and plot of movie
def get_movie(all_movies, movie_title):
	year = 0
	plot = ''
	for movie in all_movies:
		if movie['title'] == movie_title:
			year = movie['year']
			plot = movie['summary']
			break
	return year, plot

# Returns Predicted movie year
def predict_decade(wc, dw, years_train, train_sample_size, plot='', title='', all_movies=None, prints=False, skip_words=None):
	decade = 0
	if not title == '' and not all_movies == None:
		decade, plot = get_movie(all_movies, title)
	
	decade_probs = movie_decade_probs(wc, dw, years_train, train_sample_size, plot, skip_words=skip_words)
	predicted_decade =  decade_probs[0][0]

	if prints:
		print title
		print "actual decade: ", decade
		print "predicted decade: ", predicted_decade
		hist_bar_plot(decade_probs, title, title+'.png')

	return predicted_decade, decade_probs

# Main Function
if __name__ == '__main__':

	# Get set of all movies
	all_movies = list(load_all_movies("plot.list.gz"))
	print "Total movies: ", len(all_movies)
	random.shuffle(all_movies)
	years, plots, titles = [], [], []

	for movie in all_movies:
		years.append(movie['year'])
		plots.append(movie['summary'])
		titles.append(movie['title'])

	min_year, max_year, bin_num = year_stats(years)

	# Get uniform subset of movies
	years_train, plots_train, titles_train = [], [], []
	years_test, plots_test, titles_test = [], [], []
	year_count_train = [0]*bin_num
	year_count_test = [0]*bin_num
	train_sample_size = 5000
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
	dw, wc = all_x_all_y(years_train, plots_train)

	# Test classifier
	correct_count = [0.]*bin_num
	confusion_matrix = np.zeros((bin_num,bin_num))
	for i, plot in enumerate(plots_test):
		predicted_decade, decade_probs = predict_decade(wc, dw, years_train, train_sample_size, plot=plot)
		actual_year = years_test[i]
		ay_bin = int((actual_year - min_year)/10)
		py_bin = int((predicted_decade - min_year)/10)
		confusion_matrix[ay_bin,py_bin] += 1
		for i, decade in enumerate(decade_probs):
			correct_count[i] += 1. if decade[0] == actual_year else 0.

	# 2a. Plot P(Y)
	hist_plot(years, 'PMF of P(Y)', 'P2a.png')

	# 2b. Plot P(Y|X "radio" > 0)
	hist_plot(y_given_x(years, plots, 'radio'), "PMF of P(Y|X'radio'>0)", 'P2b.png')

	# 2c. Plot P(Y|X "beaver" > 0)
	hist_plot(y_given_x(years, plots, 'beaver'), "PMF of P(Y|X'beaver'>0)", 'P2c.png')

	# 2d. Plot P(Y|X "the" > 0)
	hist_plot(y_given_x(years, plots, 'the'), "PMF of P(Y|X'the'>0)", 'P2d.png')

	# 2e. Plot P(Y)
	hist_plot(years_train, 'PMF of P(Y)', 'P2e.png')

	# 2f. Plot P(Y|X "radio" > 0)
	hist_plot(y_given_x(years_train, plots_train, 'radio'), "PMF of P(Y|X'radio'>0)", 'P2f.png')

	# 2g. Plot P(Y|X "beaver" > 0)
	hist_plot(y_given_x(years_train, plots_train, 'beaver'), "PMF of P(Y|X'beaver'>0)", 'P2g.png')

	# 2h. Plot P(Y|X "the" > 0)
	hist_plot(y_given_x(years_train, plots_train, 'the'), "PMF of P(Y|X'the'>0)", 'P2h.png')

	# 2j. Predicts for certain movies
	predict_decade(wc, dw, years_train, train_sample_size, title='Finding Nemo', all_movies = all_movies, prints = True)
	predict_decade(wc, dw, years_train, train_sample_size, title='The Matrix', all_movies = all_movies, prints = True)
	predict_decade(wc, dw, years_train, train_sample_size, title='Gone with the Wind', all_movies = all_movies, prints = True)
	predict_decade(wc, dw, years_train, train_sample_size, title='Harry Potter and the Goblet of Fire', all_movies = all_movies, prints = True)
	predict_decade(wc, dw, years_train, train_sample_size, title='Avatar', all_movies = all_movies, prints = True)
	
	# 2k. Accuracy of the classifier
	print "Accuracy on test: ", correct_count[0]/len(plots_test)

	# 2l. Cumulative Match Curve
	cum_correct_count = [0.]*bin_num
	for i, c in enumerate(correct_count):
		cum_correct_count[i] = sum(correct_count[:i+1])/sum(correct_count)
	plt.plot(np.arange(1,10,1),cum_correct_count)
	plt.title('Cumulative Match Curve')
	plt.xlabel('k (guesses)')
	plt.ylabel('Accuracy within k guesses')
	plt.savefig('P2l.png')
	plt.close('all')


	# 2m. Plot confusion matrix
	print "Confusion Matrix:"
	print confusion_matrix
	fig, ax = plt.subplots()
	plt.title('Confusion Matrix')
	plt.xlabel('Actual Years')
	plt.ylabel('Predicted Years')
	ax.set_xticklabels(np.arange(min_year, max_year + 10, 10))
	ax.set_yticklabels(np.arange(min_year, max_year + 10, 10))
	ax.set_xticks(np.arange(confusion_matrix.shape[1]) + 0.5, minor=False)
	ax.set_yticks(np.arange(confusion_matrix.shape[0]) + 0.5, minor=False)
	heatmap = ax.pcolor(confusion_matrix, cmap="spectral")
	cb = fig.colorbar(heatmap, ax=ax)
	cb.set_label('Magnitude')
	plt.savefig('P2m.png')
	plt.close('all')














