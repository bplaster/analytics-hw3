from sklearn import random_projection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import math
import random
import numpy as np
import P2
from parse_movies_example import load_all_movies 
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn.lda import LDA
import matplotlib.pyplot as plt

dimensions = [500, 2000, 10000, 30000]

def vary_dimensions (dimensions,features,years_train,years_test,plots_test):
	accuracies = []
	clf = LDA()
	for dimension in dimensions:
		# Dimensionality reduction
		transformer = random_projection.SparseRandomProjection(dense_output = True, n_components = dimension)
		reduced_features = transformer.fit_transform(features)
		# Scaling
		scaler = preprocessing.StandardScaler().fit(reduced_features)
		scaled_reduced_features = scaler.transform(reduced_features)
		# Training
		clf.fit(scaled_reduced_features,years_train)
		# Transform test plots
		transformed_test_plots = scaler.transform(transformer.transform(vec.transform(plots_test)))
		accuracies.append(get_accuracy (clf,years_test,transformed_test_plots))
	return accuracies

def get_accuracy (classifier, years_test, transformed_test_plots):
	correct_count = 0.
	years_test = np.array(years_test)
	#new_test_plots = scaler.transform(transformer.transform(vec.transform(plots_test)))
	predicted_decade = classifier.predict(transformed_test_plots)
	for i,decade in enumerate(predicted_decade):
		if decade == years_test[i]:
			correct_count += 1.
	return correct_count/len(plots_test)

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

	# Extract features
	vec = CountVectorizer(tokenizer = P2.norm_words, encoding = 'latin-1')
	features = vec.fit_transform(plots_train)
	print "Shape of original feature matrix: ", features.shape

	# Dimensionality reduction
	# transformer = random_projection.SparseRandomProjection(dense_output = True, n_components = 500)
	# reduced_features = transformer.fit_transform(features)
	# print "Shape of reduced feature matrix: ", reduced_features.shape

	# Scaling
	# scaler = preprocessing.StandardScaler().fit(reduced_features)
	# scaled_reduced_features = scaler.transform(reduced_features)

	# Train classifier
	# clf = linear_model.SGDClassifier()
	# clf = svm.LinearSVC()
	# clf = svm.SVC(kernel = 'rbf')
	# clf = linear_model.Perceptron(penalty='l1')
	# clf = linear_model.Perceptron(penalty='l2',n_iter = 25)
	# clf = neighbors.KNeighborsClassifier()
	# clf = LDA()
	# clf.fit(scaled_reduced_features, years_train)
	# print "Completed training"

	# Transform test plots
	# transformed_test_plots = scaler.transform(transformer.transform(vec.transform(plots_test)))

	# Vary k for LDA (favorite classifier)
	accuracies = vary_dimensions (dimensions,features,years_train,years_test,plots_test)
	print "Accuracy for respective dimensions: ", accuracies

	plt.plot(dimensions,accuracies)
	plt.title('Variation of accuracy with number of features')
	plt.xlabel('No. of features')
	plt.ylabel('Accuracy')
	plt.savefig('dimensionality_accuracy.png')
	plt.close('all')

	#Test classifier
	#print "Accuracy on test: ", get_accuracy(clf,years_test,transformed_test_plots)

