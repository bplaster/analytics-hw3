from parse_movies_example import load_all_movies 
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
import random

# Returns statistics about years
def year_stats(years):
	min_y = min(years)
	max_y = max(years)
	bin_n = int((max_y - min_y) / 10) + 1
	return min_y, max_y, bin_n

# Returns lower case word without some special characters
def norm_word(word):
	word = word.strip().lower()
	tbr = ["'s", '.', '?', '!', ',', ':', ';','(',')',"'"]
	for c in tbr:
		if c in word:
			word = word.replace(c,'')
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
	plt.show()

# Returns list of all years with given word in plot
def y_given_x(years, plots, word):
	y_g_x = []
	word = norm_word(word)
	for i, plot in enumerate(plots):
		if word in plot.lower:
			y_g_x.append(years[i])
	return y_g_x

# Returns dictionary of all words and count in all years
def all_x_all_y(years, plots):
	wc = {}
	for i, year in enumerate(years):
		plot = plots[i].split()
		for word in plot:
			word = norm_word(word)
			if word in wc:
				if year in wc[word]: 
					wc[word][year] += 1.
				else:
					wc[word][year] = 1.
			else:
				wc[word] = {}
				wc[word][year] = 1.
	return wc

# Returns probability of word in given year
def p_x_given_y(wc, word, year):
	p = 0.00001
	word = norm_word(word)
	if word in wc:
		if year in wc[word]:
			p = wc[word][year]/sum(wc[word].values())
	#print word, p
	return p	

# Returns dictionary of movie prediction for each year
def movie_decade_predict(years, plots, movie_plot):
	min_year, max_year, bin_num = year_stats(years)
	decades = {}
	wc = all_x_all_y(years, plots)
	for year in range(min_year, max_year + 10, 10):
		p = 0
		for word in movie_plot.split():
			p += math.log(p_x_given_y(wc, word, year),10)
		decades[year] = p
	#print decades
	return decades

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

# Prints Predicted and Actual movie year
def predict_movie(years_train, plots_train, plot = '', title = '', all_movies = None, prints = False):
	decade = 0
	if not title == '' and not all_movies == None:
		decade, plot = get_movie(all_movies, title)
	
	decades = movie_decade_predict(years_train, plots_train, plot)
	predicted_decade =  max(decades.iteritems(), key=operator.itemgetter(1))[0]

	if prints:
		print title
		print "actual decade: ", decade
		print "predicted decade: ", predicted_decade

	return predicted_decade

# Main Function
if __name__ == '__main__':

	# Get set of all movies
	all_movies = list(load_all_movies("plot.list.gz"))
	years, plots, titles = [], [], []

	for movie in all_movies:
		years.append(movie['year'])
		plots.append(movie['summary'])
		titles.append(movie['title'])

	min_year, max_year, bin_num = year_stats(years)

	# # 2a. Plot P(Y)
	# hist_plot(years, 'PMF of P(Y)', 'P2a.png')

	# # 2b. Plot P(Y|X "radio" > 0)
	# hist_plot(y_given_x(years, plots, 'radio'), "PMF of P(Y|X'radio'>0)", 'P2b.png')

	# # 2c. Plot P(Y|X "beaver" > 0)
	# hist_plot(y_given_x(years, plots, 'beaver'), "PMF of P(Y|X'beaver'>0)", 'P2c.png')

	# # 2d. Plot P(Y|X "the" > 0)
	# hist_plot(y_given_x(years, plots, 'the'), "PMF of P(Y|X'the'>0)", 'P2d.png')

	# Get uniform subset of movies
	years_train, plots_train, titles_train = [], [], []
	years_test, plots_test, titles_test = [], [], []
	year_count_train = [0]*bin_num
	year_count_test = [0]*bin_num
	train_sample_size = 6000
	test_sample_size = 100

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


	# # 2e. Plot P(Y)
	# hist_plot(years_train, 'PMF of P(Y)', 'P2e.png')

	# # 2f. Plot P(Y|X "radio" > 0)
	# hist_plot(y_given_x(years_train, plots_train, 'radio'), "PMF of P(Y|X'radio'>0)", 'P2f.png')

	# # 2g. Plot P(Y|X "beaver" > 0)
	# hist_plot(y_given_x(years_train, plots_train, 'beaver'), "PMF of P(Y|X'beaver'>0)", 'P2g.png')

	# # 2h. Plot P(Y|X "the" > 0)
	# hist_plot(y_given_x(years_train, plots_train, 'the'), "PMF of P(Y|X'the'>0)", 'P2h.png')

	# 2j
	predict_movie(years_train, plots_train, title='Finding Nemo', all_movies = all_movies, prints = True)
	predict_movie(years_train, plots_train, title='The Matrix', all_movies = all_movies, prints = True)
	predict_movie(years_train, plots_train, title='Gone with the Wind', all_movies = all_movies, prints = True)
	predict_movie(years_train, plots_train, title='Harry Potter and the Goblet of Fire', all_movies = all_movies, prints = True)
	predict_movie(years_train, plots_train, title='Avatar', all_movies = all_movies, prints = True)

	correct_count = 0
	for i, plot in enumerate(plots_test):
		decade = predict_movie(years_train, plots_train, plot=plot)
		correct_count += 1 if decade == years_test[i] else 0
		#print titles_test[i]

	print "Accuracy on test: ", correct_count/len(plots_test)







