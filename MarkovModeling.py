import numpy as np
import pandas as pd
import os
import math

#The model stores transition probabilities for each language, from each state to each next state
#The states will be ngrams and transitions from ngram A to ngram B is when AB occurs
model = dict()
#A list of every possible character that occurs in the data training or testing
character_vocabulary = list()
#Every ngram in the dataset
all_ngrams = list()
#The size of the ngrams
order = 3

#For printing a progress percentage
#Num is the counter, end is the value counter must reach to finish
def print_progress(num, end):
    print(round(num / end * 100), "%  \r", sep="", end="")

#Convert a sentence into a list of consecutive ngrams or length order
#IMPORTANT NOTE: does NOT include the last ngram, as it assumes we only want ngrams that have a character after.
#If you want all the ngrams of a sentence,  change "(len(sentence) - (order - 1) - 1)" to: "(len(sentence) - (order - 1))"
def ngramify(sentence):
	ngram_list = list()
	for i in range(len(sentence) - (order - 1) - 1):
		ngram_list.append(sentence[i:i+order])

	return ngram_list

#Train the markov model for a language by recording state transition probabilities with additive smoothing
#Language is a string identifying which language
#Data is a list of the training sentences
#Smoothing is the additive smoothing value added to each transition
def train(language, data, smoothing):
	global model
	#Initialize the model with just the smoothing value
	print("Initializing")
	model[language] = dict()
	for ngram in all_ngrams[order-1]:
		model[language][ngram] = dict()
		for c in character_vocabulary:
			model[language][ngram][c] = smoothing


	#Add up the transition counts
	print("Counting")
	counter = 0
	for sentence in data:
		if counter % 10000 == 0:
			print_progress(counter, len(data))
		counter += 1
		sentence_ngrams = ngramify(sentence, order)
		for (i, ngram) in enumerate(sentence_ngrams):
			model[language][ngram][sentence[i+order]] += 1

	#Normalize to convert counts into probabilities
	print("Normalizing")
	counter = 0
	for ngram in model[language]:
		if counter % 10000 == 0:
			print_progress(counter, len(model[language]))
		counter += 1
		count_sum = 0
		for c in model[language][ngram]:
			count_sum += model[language][ngram][c]
		for c in model[language][ngram]:
			model[language][ngram][c] /= count_sum

#Language is a string identifying the language the model will use to calculate the probability of the sentence
#Sentence is a string to be evaluated
def sentence_language_probability(language, sentence):
	log_prob = 0
	for (i, ngram) in enumerate(ngramify(sentence, order)):
		log_prob += math.log(model[language][ngram][sentence[i+order]])
	return math.exp(log_prob)

#Generate all ngrams from length 1 to length order and store in all_ngrams
def generate_all_ngrams(data_list):
	global all_ngrams
	all_ngrams.append(list(character_vocabulary))
	for i in range(1, order):
		tmp_list = list()
		for ngram in all_ngrams[i-1]:
			for c in character_vocabulary:
				tmp_list.append(ngram + c)
		all_ngrams.append(list(tmp_list))

def find_vocabulary(data_list):
	global character_vocabulary, all_ngrams
	chars = set()
	ngram_set = set()
	full_data_size = sum([len(data) for data in data_list])
	counter = 0
	for data in data_list:
		for sentence in data:
			if counter % 10000 == 0:
				print_progress(counter, full_data_size)
			counter += 1
			for i in range(len(sentence) - (order - 1) - 1):
				ngram_set.add(sentence[i:i+order])
				char.add(sentence[i])
			for i in range(len(sentence) - (order - 1) - 1, len(sentence)):
				chars.add(sentence[i])
	all_ngrams = list(ngram_set)
	character_vocabulary = list(chars)

CD = 'dataset/'
SL = 'EN' #this is a constant and should not be changed, i.e. Source Language is always English
TL = 'NL' #depending on the desired Target Language, this could be set, available abbr. choices are written in the introduction paragraph
print("Reading datasets")
data_EN_NL = pd.read_csv(CD+SL+'-'+TL+'/'+SL+'-'+TL+'.txt', sep='\t', header = None)[[0,1]].rename(columns = {0:SL, 1:TL})
data_EN = [x for x in data_EN_NL[SL] if type(x) is str]
data_NL = [x for x in data_EN_NL[TL] if type(x) is str]

print("Finding all characters")
find_all_characters([data_EN, data_NL])
print("Generating all ngrams")
generate_all_ngrams(ngram_order)
# print("Training EN")
# train("EN", data_EN, ngram_order, 1)
# print("Training NL")
# train("NL", data_NL, ngram_order, 1)

#10 fold cross validation
split = int(len(data_EN) / 10)
av_accuracy = 0
for i in range(10):
	print("Split", i+1)
	testing_EN = data_EN[i*split : (i+1)*split]
	testing_NL = data_NL[i*split : (i+1)*split]
	training_EN = data_EN[:i*split] + data_EN[(i+1)*split:]
	training_NL = data_NL[:i*split] + data_NL[(i+1)*split:]
	print("Training EN")
	train("EN", training_EN, ngram_order, 1)
	print("Training NL")
	train("NL", training_NL, ngram_order, 1)

	print("Testing EN")
	accuracy = 0
	counter = 0
	for sentence in testing_EN:
		if counter % 10000 == 0:
			print_progress(counter, len(testing_EN))
		counter += 1
		if sentence_language_probability("EN", sentence, ngram_order) > sentence_language_probability("NL", sentence, ngram_order):
			accuracy += 1

	print("Testing NL")
	counter = 0
	for sentence in testing_NL:
		if counter % 10000 == 0:
			print_progress(counter, len(testing_NL))
		counter += 1
		if sentence_language_probability("NL", sentence, ngram_order) > sentence_language_probability("EN", sentence, ngram_order):
			accuracy += 1

	print("Accuracy:", accuracy / (len(testing_EN) + len(testing_NL)))
	av_accuracy += accuracy / (len(testing_EN) + len(testing_NL))
print("Average accuracy:", av_accuracy / 10)
