"this part is corresponding to Question 1 - 6"

import csv
path = "../ml-latest-small/ratings.csv"
total_possible_rating = 0
total_available_rating = 0

with open(path, 'r') as myFile:
    lines = csv.reader(myFile)
    user = set()
    movie = set()
    first_row = True
    for line in lines:
        if first_row:
            first_row = False
            continue
        total_available_rating += 1
        user.add(int(line[0]))
        movie.add(int(line[1]))
    total_possible_rating = len(user) * len(movie)

print total_available_rating
print total_possible_rating
print float(total_available_rating) / total_possible_rating

