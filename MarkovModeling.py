import numpy as np
import pandas as pd
import os
import math
import sys

#For printing a progress percentage
#Num is the counter, end is the value counter must reach to finish
def print_progress(num, end):
    print(round(num / end * 100), "%  \r", sep="", end="")

#Convert a sentence into a list of consecutive ngrams or length order
#IMPORTANT NOTE: does NOT include the last ngram, as it assumes we only want ngrams that have a character after.
#If you want all the ngrams of a sentence,  change "(len(sentence) - (order - 1) - 1)" to: "(len(sentence) - (order - 1))"
def ngramify(sentence, order):
	ngram_list = list()
	for i in range(len(sentence) - (order - 1)):
		ngram_list.append(sentence[i:i+order])

	return ngram_list

#Train the markov model for a language by recording state transition probabilities with additive smoothing
#Language is a string identifying which language
#Data is a list of the training sentences
#Smoothing is the additive smoothing value added to each transition
def train(lang_list, data_list, smoothing):
	global model, all_ngrams
	#Add up the transition counts
	print("Counting")
	counter = 0
	full_len = sum([len(x) for x in data_list])
	for i in range(len(lang_list)):
		lang = lang_list[i]
		data = data_list[i]
		model[lang] = dict()
		init_model[lang] = dict()
		for sentence in data:
			if counter % 1000 == 0:
				print_progress(counter, full_len)
			counter += 1
			init = sentence[:ngram_order]
			if init not in init_model[lang]:
				init_model[lang][init] = 0
			init_model[lang][init] += 1
			sentence_ngrams = ngramify(sentence, ngram_order+1)
			for (i, ngram) in enumerate(sentence_ngrams):
				init = ngram[:ngram_order]
				c = ngram[ngram_order]
				if init not in model[lang]:
					model[lang][init] = dict()
				if c not in model[lang][init]:
					model[lang][init][c] = 0
				model[lang][init][c] += 1
			init = sentence[len(sentence)-ngram_order:]
			if init not in model[lang]:
				model[lang][init] = dict()
			if "end" not in model[lang][init]:
				model[lang][init]["end"] = 0
			model[lang][init]["end"] += 1

	#Normalize to convert counts into probabilities
	print("Normalizing")
	counter = 0
	full_len = sum([len(model[l]) + len(init_model[l]) for l in model])
	for i in range(len(lang_list)):
		lang = lang_list[i]
		data = data_list[i]
		for ngram in model[lang]:
			if counter % 10000 == 0:
				print_progress(counter, full_len)
			counter += 1
			count_sum = 0
			for c in model[lang][ngram]:
				count_sum += model[lang][ngram][c]
				model[lang][ngram][c] += smoothing
			for c in model[lang][ngram]:
				model[lang][ngram][c] = math.log(model[lang][ngram][c] / (count_sum + (len(character_vocabulary) + 1) * smoothing))
			model[lang][ngram]["unseen"] = math.log(smoothing / (count_sum + (len(character_vocabulary) + 1) * smoothing))
		count_sum = 0
		for ngram in init_model[lang]:
			if counter % 10000 == 0:
				print_progress(counter, full_len)
			counter += 1
			count_sum += init_model[lang][ngram]
			init_model[lang][ngram] += smoothing
		for ngram in init_model[lang]:
			init_model[lang][ngram] = math.log(init_model[lang][ngram] / (count_sum + (len(character_vocabulary) ** ngram_order) * smoothing))
		init_model[lang]["unseen"] = math.log(smoothing / (count_sum + (len(character_vocabulary) ** ngram_order) * smoothing))

#Language is a string identifying the language the model will use to calculate the probability of the sentence
#Sentence is a string to be evaluated
def sentence_language_log_probability(language, sentence):
	ngram = sentence[:ngram_order]
	if ngram in init_model[language]:
		log_prob = init_model[language][ngram]
	else:
		log_prob = init_model[language]["unseen"]
	for (i, ngram) in enumerate(ngramify(sentence, ngram_order+1)):
		c = ngram[ngram_order]
		ngram = ngram[:ngram_order]
		if ngram in model[language]:
			if c in model[language][ngram]:
				log_prob += model[language][ngram][c]
			else:
				log_prob += model[language][ngram]["unseen"]
		else:
			log_prob += unseen_value
	ngram = sentence[len(sentence)-ngram_order:]
	if ngram in model[language]:
		if "end" in model[language][ngram]:
			log_prob += model[language][ngram]["end"]
		else:
			log_prob += model[language][ngram]["unseen"]
	else:
		log_prob += unseen_value
	# if math.exp(log_prob) == 0:
	# 	print("AAA", log_prob, math.exp(log_prob))
	return log_prob

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

def find_all_characters(data_dict):
	global character_vocabulary
	chars = set()
	full_data_size = sum([len(data_dict[lang]) for lang in data_dict])
	counter = 0
	for lang in data_dict:
		data = data_dict[lang]
		for sentence in data:
			if counter % 1000 == 0:
				print_progress(counter, full_data_size)
			counter += 1
			for i in range(len(sentence)):
				chars.add(sentence[i])
	character_vocabulary = list(chars)

file_list = os.listdir("data")
lang_list = [x[:2] for x in file_list]

# LIMIT = 1000000
LIMIT = None

# print([x[:-3] for x in open("data/IT.tsv", 'r').read().split("\n")[:10] if len(x) > 0][1:])
data_dict = {}
print("Reading from file:")
for lang in lang_list:
	print(lang)
	if LIMIT is not None:
		data_dict[lang] = [x[:-3] for x in open("data/" + lang + ".tsv", 'r').read(LIMIT).split("\n") if len(x) > 0][1:]
	else:
		data_dict[lang] = [x[:-3] for x in open("data/" + lang + ".tsv", 'r').read().split("\n") if len(x) > 0][1:]

print("Finding all characters")
find_all_characters(data_dict)
smoothing_value = 1
unseen_value = math.log(smoothing_value / len(character_vocabulary))

ngram_order = 4

print("NGRAM ORDER:", ngram_order)
#The model stores transition probabilities for each language, from each state to each next state
#The states will be ngrams and transitions from ngram A to ngram B is when AB occurs
model = dict()
init_model = dict()
#A list of every possible character that occurs in the data training or testing
character_vocabulary = list()
#Every ngram in the dataset
all_ngrams = list()

#10 fold cross validation
av_accuracy = 0
confusion_matrix = {}
for lang in data_dict:
	confusion_matrix[lang] = {}
	for lang1 in data_dict:
		confusion_matrix[lang][lang1] = 0

overall_accuracy = 0
balanced_accuracy = {}
precision = {}
recall = {}
fscore = {}
for lang in data_dict:
	balanced_accuracy[lang] = 0
	precision[lang] = 0
	recall[lang] = 0
	fscore[lang] = 0

for i in range(10):
	print("Split", i+1)
	split = int(len(data_dict[lang]) / 10)

	testing_sets = {}
	training_sets = {}
	for lang in data_dict:
		testing_sets[lang] = data_dict[lang][i*split : (i+1)*split]
		training_sets[lang] = data_dict[lang][:i*split] + data_dict[lang][(i+1)*split:]

	print("Training")
	train([lang for lang in data_dict], [training_sets[lang] for lang in data_dict], smoothing_value)
	print("Testing")
	lc = 0
	for test_lang in data_dict:
		print_progress(lc, len(data_dict))
		lc += 1
		# print("Testing", test_lang)
		for sentence in testing_sets[test_lang]:
			max_prob = None
			max_lang = None
			for lang1 in data_dict:
				p = sentence_language_log_probability(lang1, sentence)
				# print(p)
				if max_prob is None or p > max_prob:
					max_prob = p
					max_lang = lang1
			confusion_matrix[test_lang][max_lang] += 1

	for lang in confusion_matrix:
		TP = 0
		FP = 0
		FN = 0
		TN = 0
		for lang1 in confusion_matrix[lang]:
			if lang == lang1:
				TP += confusion_matrix[lang][lang1]
			else:
				FN += confusion_matrix[lang][lang1]
		for lang1 in confusion_matrix:
			if lang != lang1:
				FP += confusion_matrix[lang1][lang]
		for lang1 in confusion_matrix:
			for lang2 in confusion_matrix:
				if lang1 != lang and lang2 != lang:
					TN += confusion_matrix[lang1][lang2]

		balanced_accuracy[lang] += (TP / (TP + FN) + TN / (TN + FP)) / 2
		precision[lang] += TP / (TP + FP)
		recall[lang] += TP / (TP + FN)
		fscore[lang] += TP / (TP + (FP + FN) / 2)
	T = 0
	F = 0
	for lang in confusion_matrix:
		for lang1 in confusion_matrix:
			if lang == lang1:
				T += confusion_matrix[lang][lang1]
			else:
				F += confusion_matrix[lang][lang1]
	print("Overall accuracy:", T / (T+F))
	overall_accuracy += T / (T+F)

s = "  "
print(s, end='')
for lang in data_dict:
	print(lang, end=s)
print()

for lang in data_dict:
	print(lang, end=s)
	for lang1 in data_dict:
		print(confusion_matrix[lang][lang1]/10, end=s)
	print()

print()
print()

print("    ", end='')
for lang in data_dict:
	print(lang, end=s + "    ")
print()

for lang in data_dict:
	print(lang, end=s)
	m = 0
	for lang1 in data_dict:
		m += confusion_matrix[lang][lang1]
	for lang1 in data_dict:
		print("{:.4f}".format(confusion_matrix[lang][lang1]/m), end=s)
	print()

print()
print()
rsum = 0
psum = 0
fsum = 0

for lang in confusion_matrix:
	print(lang)
	print("Balanced accuracy:", balanced_accuracy[lang] / 10)
	print("Precision:", precision[lang] / 10)
	psum += precision[lang] / 10
	print("Recall:", recall[lang] / 10)
	rsum += recall[lang] / 10
	print("F-score:", fscore[lang] / 10)
	fsum += fscore[lang] / 10

print()
print()
print("Overall accuracy:", overall_accuracy / 10)
print("Average Precision:", psum / len(confusion_matrix))
print("Average Recall:", rsum / len(confusion_matrix))
print("Average F-score:", fsum / len(confusion_matrix))
