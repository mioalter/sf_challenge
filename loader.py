#!/usr/bin/env python
import sys
import re
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import LabelEncoder as le

'''
We have a basic title standardizing function
We have a function to standardize no. of employees
We have a function to make the dependent variable
To do
* decide how to pivot each column
* assemble results into one array
* NNs
* RNNs

Should we make a dictionary to store these so they are not all floating around?
Oh, no, we can just transform/standardize the columns directly.
Access a column by name with df['col_name']
Columns
* lead_source, initial_lead_source
* offer_type, initial_offer_type
* title
* job function
* job_level
* num_employees
* signup_edition
* promotion_code

WOW: one-hot expects INTEGERS even though it's meant to encode categorical variables.
So you have to map each string value to an integer


YES! Scracth that! Use scikit label-encoder! That's exactly what we want.

Can use df.fillna('') to replace nans with strings!
Also annoying, and could just change the query

SO: LabelEncoder takes no parameters, you have to encode all values.
OneHotEncoder then pivots all encoded values. Thus, if you want to pivot only the M most frequently-occuring
values, you have to figure that one out yourself. To solve this we want to 
* count the number of times each value occurs in a dict {value:count}
* make a mapping from original values to new values {old_value: new_value} which maps frequently-occuring values
to themselves and rare vales to 'other'
* replace old values with new values
* apply the existing LabelEncoder
'''

#############
## Loading ##
#############

def load_data(datafile):
	df = pd.read_csv(datafile, sep='\t', dtype = str)
	return df.fillna('')

###################
## Standardizing ##
###################

stopwords = set(['the', 'of', 'a', 'co', 'to', 'and'])

def gate(s):
	if len(s) < 2 or s in stopwords:
		return False
	else:
		return True

def standardize_title(s):
	s = s.lower()
	words = sorted(re.split('\/| |-|\(|\)|\,|\:|\&', s))
	words = [w.replace('.','') for w in words] #quick hack so 'c.e.o.' maps to 'ceo'
	good_words = filter(gate, words)
	return ' '.join(w for w in good_words)

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

def transform_dep(s):
	label = 0
	if len(s)> 0:
		label = 1
	return label

def make_dep(df, dep_index):
	return np.array([s for s in df.ix[:,dep_index].apply(transform_dep)])

def one2two(a):
	'''
	a = 0 or 1
	'''
	vect = [1,0]
	if a == 1:
		vect = [0,1]
	return vect

def make_dep_array(y):
	return np.array([one2two(a) for a in y])


def standardize_df(df):
	df['title'] = df['title'].apply(standardize_title)
	df['num_employees'] = df['num_employees'].apply(standardize_num_employees)
	return df

###################
## Preprocessing ##
###################

# def add_bias(X):
# 	n = X.shape[0]
# 	bias = np.ones(n, dtype=float)
# 	sparse_bias = ss.csr_matrix(bias.reshape(n,1))
# 	return ss.hstack((X,sparse_bias))


def label_mapper(V, n_values):
	'''
	maps frequently-occuring values to themselves, rare values to 'other'
	returns np.array of new values and set of new values
	'''
	
	vals_count = {}
	W = []
	
	for v in V:
		if v in vals_count:
			vals_count[v] += 1
		else:
			vals_count[v] = 1
	
	min_cardinality = 0
	if n_values != None:
		min_cardinality = sorted(vals_count.values(), reverse=True)[n_values]
	
	for v in V:
		if vals_count[v] > min_cardinality:
			W.append(v)
		else:
			W.append('other')
	return np.array(W), set(W)


def encode(V, n_values):
	new_V, new_values = label_mapper(V, n_values)
	encoder = le()
	encoded_V = encoder.fit_transform(new_V)
	return encoded_V, new_values

def make_array(L):
    cols = tuple([l.reshape(len(l),1) for l in L])
    return np.concatenate(cols, axis = 1)

def pivot_data(df, cols_and_vals):
	'''
	df - a pandas dataframe
	cols_and_vals - a list (str, int) of column names and number of values to pivot
	returns pivoted array and dict {column_name:pivoted_values}
	'''
	
	encoded_cols = []
	pivoted_vals = {}
	
	for p in cols_and_vals:
		col, n_vals = p[0], p[1]
		encoded_col, new_vals = encode(df[col], n_vals)
		encoded_cols.append(encoded_col)
		pivoted_vals[col] = new_vals
	
	encoded_array = make_array(encoded_cols)
	
	oneHot = ohe(categorical_features = 'all', n_values = 'auto')
	pivoted_array = oneHot.fit_transform(encoded_array)
	
	return pivoted_array, pivoted_vals

################
## Evaluation ##
################

def group_sum(group):
	return sum([g[1] for g in group])

def count_summer(accum, pair):
	total, data = accum[0], accum[1]
	bucket, count = pair[0], pair[1]
	new_total = total + count
	data.append((bucket, new_total))
	return new_total, data

def cummulative_opens(counts):
	return reduce(count_summer, counts, (0,[]))

def cummulative_stats(counts):
	total, sums = cummulative_opens(counts)
	percents = [(bucket, float(s) / total) for (bucket,s) in sums]
	return sums, percents

def score2bucket(L, p):
	'''
	L - list of bucket boundaries (floats)
	p - (score, label)
	return (bucket_number, label)
	'''
	score, label = p[0], p[1]
	bucket = 0
	for i in xrange(len(L) - 1):
		start, end = L[i], L[i+1]
		if start <= score < end:
			bucket = i
	return bucket+1, label

def make_equal_intervals(scores, bins):
	L = sorted(scores)
	interval_size = len(L) / bins
	endpoints = [L[i*interval_size] for i in xrange(bins)]
	endpoints.append(L[-1])
	return endpoints

def make_percents(pairs, bins, method):
	scores = [p[0] for p in pairs]
	if method=='equal':
		intervals = make_equal_intervals(scores, bins)
	else:
		intervals = np.histogram(scores, bins=bins)[1]
	bucket_pairs = sorted([score2bucket(intervals, p) for p in pairs], key=lambda x: x[0], reverse=True)
	groups = it.groupby(bucket_pairs, key=lambda x: x[0])
	counts = [(k, group_sum(g)) for k,g in groups]
	sums, percents = cummulative_stats(counts)
	return percents

def print_percents(percents):
	for p in percents:
		bucket, percent = p[0], p[1]
		print 'bucket: %s, percent opens: %s' %(bucket, percent)

def evaluate(model, X_val, y_val, title, bins=10):
	predictions = model.predict(X_val.todense())
	scores = [p[1] for p in predictions]
	pairs = zip(scores, y_val)
	equal_percents = make_percents(pairs, bins, 'equal')
	percentile_percents = make_percents(pairs, bins, 'percentile')
	plot_roc(y_val, scores, title)
	print 'equal interval percent opens:'
	print_percents(equal_percents)
	print 'percentile interval percent opens:'
	print_percents(percentile_percents)


##########
## Main ##
#########

df = load_data(datafile)
df = standardize_df(df)
y = make_dep(df, -2)

cols_and_vals = [ 
  ('lead_source', 50)
  , ('offer_type', 50)
  , ('title',75) 
  , ('job_function', None)
  , ('num_employees', None)
  , ('signup_edition', None)
  , ('promotion_code', 100)
]
X, values = pivot_data(df, cols_and_vals)

