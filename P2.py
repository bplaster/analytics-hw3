from parse_movies_example import load_all_movies 

min_year = 1900
max_year = 2014
division = 10

all_movies = list(load_all_movies("plot.list.gz"))
year_counts = {}

for movie in all_movies:
	y = movie['year']
	if y in year_counts:
		year_counts[y] += 1
	else:
		year_counts[y] = 1

print year_counts