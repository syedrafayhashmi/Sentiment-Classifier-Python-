import csv

import numpy as np

filename = "F:\FYP\hadi scraping\labelledData_200.csv"

fields = []
rows = []

with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    # extracting field names through first row

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
names = fields
print('fields = >',names)
print(rows)

