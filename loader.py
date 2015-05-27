#!/usr/bin/env python
import sys
import re
# import scikit one-hot

stopwords = ['the', 'of', 'a', 'co']

def gate(s):
	if len(s) < 2 or s in stopwords:
		return False
	else:
		return True

def standardize_title(s):
	s = s.lower()
	words = filter(lambda x: x != None, sorted(re.split('\/| |-|(|)|,|:', s)))
	good_words = filter(gate, words)
	return ' '.join(w for w in good_words)


# we should make buckets by looking at the data, not arbitrarily
def standardize_num_employees(n):
	try:
		n = int(n)
		if 0 < n <= 10:
			return '1-10'
		elif 10 < n <= 25:
			return '11-25'
		elif 25 < n <= 50:
			return '26-50'
		elif 50 < n <= 100:
			return '51-100'
		elif 100 < n <= 200:
			return '101-200'
		elif 200 < n <= 500:
			return '201-500'
		elif 500 < n <= 1000:
			return '501-1000'
		elif 1000 < n <= 5000:
			return '1001-5000'
		elif 5000 < n <= 10000:
			return '5001-10000'
		else:
			return '>10k'
	except:
		return 'n/a'

'''
We have a basic title standardizing function
We need a function to standardize no. of employees
We need to figure out how to apply functions to individual columns of a data frame
Use df.apply(fcn, axis)
'''