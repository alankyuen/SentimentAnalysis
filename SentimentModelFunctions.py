import os,sys
import csv, json
from nltk.corpus import wordnet as wn
import numpy as np 
import math, random


def getValence(word,valence_dict):
	''' takes word and returns valence '''

	#removes punctuation from word
	punctuations = '''!"#$%&()*+,./:;<=>?@[\]^_`{|}~\n'''
	word = word.translate(str.maketrans("", "", punctuations))	
	word = word.lstrip("N_")

	if(word in valence_dict):
		return valence_dict[word]
	else:
		return -999

def isNegator(word):
	
		negators = ["n't", "cannot", "never","no","nothing","none","nobody","neither","nowhere","not"]
		if(word in negators):
				return True
		else:
				return False


#*------------------------------------------------------------------------------------------*#
#*                                Review Data Structure Methods                             *#
#*------------------------------------------------------------------------------------------*#

def map_token_tree(tree):
	''' takes a NLTK_tree and maps it to token_tree '''
	token_tree = []
	for subtree in tree:
		if(subtree.height() > 2):
			token_tree.append(map_token_tree(subtree))
		else:
			token_tree.append(subtree[0])
	return token_tree

def map_valence_tree(tree, valence_dict):

	''' takes a token_tree and maps it to valence_tree '''

	valence_tree = []
	for subtree in tree:
		if(type(subtree) is list):
			valence_tree.append(map_valence_tree(subtree, valence_dict))
		else:
			valence_tree.append(getValence(subtree, valence_dict))
	return valence_tree

def find_and_convert_bigrams(sequence, valence_dict):

	''' (DEPRECATED) takes a 1D sequence and looks for bigrams in left to right order '''

	new_sequence = []
	#print(sequence)
	i = 0
	if(len(sequence) < 1):
			return None

	while i < len(sequence) - 1:
			if("N_" in sequence[i] and "N_" in sequence[i+1]):

					bigram_candidate = sequence[i][2:] + " " + sequence[i+1][2:]
					if bigram_candidate in valence_dict.keys():
							bigram_candidate = "N_" + bigram_candidate
							new_sequence.append(bigram_candidate)
							i += 1
					else:
							new_sequence.append(sequence[i])

			elif("N_" not in sequence[i] and "N_" not in sequence[i+1]):
					bigram_candidate = sequence[i] + " " + sequence[i+1]
					if bigram_candidate in valence_dict.keys():
							new_sequence.append(bigram_candidate)
							i += 1
					else:
							new_sequence.append(sequence[i])
			elif(i == len(sequence)-2):
					new_sequence.append(sequence[i])
					new_sequence.append(sequence[i+1])
					i += 1
			else:
					new_sequence.append(sequence[i])

			i+= 1

	if(i != len(sequence)):
			new_sequence.append(sequence[i])
			i+=1

	return new_sequence

def find_and_convert_bigrams2(sequence, valence_dict):

	''' takes a 1D sequence and pairs tokens into bigrams based on a definable measure 
		AKY 7/24 (TODO) define a measure
	'''

	#find possible bigram pairs
	possible_bigrams = []
	for i in range(len(sequence)-1):
		bigram_match_check = ("N_" in sequence[i] and "N_" in sequence[i+1])
		bigram_match_check = bigram_match_check or ("N_" not in sequence[i] and "N_" not in sequence[i+1])
		valid_bigram_check = sequence[i]+" "+sequence[i+1] in valence_dict.keys()
		if(bigram_match_check and valid_bigram_check):
			possible_bigrams.append([i,i+1])
	
	#measure each bigram possibility
	measured_possible_bigrams = []
	for id1,id2 in possible_bigrams:
		measure = random.random()
		measured_possible_bigrams.append([[id1,id2],measure])

	#rank bigram possibilities
	measured_possible_bigrams = sorted(measured_possible_bigrams, key = lambda e: e[1])

	#pair bigrams by rank
	while(len(measured_possible_bigrams) > 0):
		id1,id2 = measured_possible_bigrams[0][0]
		outed = sequence.pop(id2)
		sequence[id1] += " "+outed.lstrip("N_")
		measured_possible_bigrams.pop(0)

		#find all duplicates of id1 or id2 and pop them
		i_offset = 0
		for i in range(len(measured_possible_bigrams)):
			entry = measured_possible_bigrams[i-i_offset]
			if(id1 in entry[0] or id2 in entry[0]):
				measured_possible_bigrams.pop(i-i_offset)
				i_offset+=1

		#find all ids > id2 and subtract by 1
		for i in range(len(measured_possible_bigrams)):
			entry = measured_possible_bigrams[i]
			if(entry[0][0] > id2):
				entry[0][0] -= 1
				entry[0][1] -= 1

	return sequence

def find_and_convert_bigrams_recursive(sub_sequence, valence_dict):

	''' takes a token_tree and pairs tokens into bigrams based on a definable measure '''

	new_sequence = []
	sequence_buffer = []
	
	for branch in sub_sequence:
			#print("branch: {}".format(branch))
			if(type(branch) is list):
					if(len(sequence_buffer) > 0):
							#solve the sequence_buffer first
							solved_sequence = find_and_convert_bigrams2(sequence_buffer, valence_dict)

							new_sequence+=solved_sequence
							sequence_buffer = []

					solved_sequence = find_and_convert_bigrams_recursive(branch, valence_dict)
					new_sequence.append(solved_sequence)
			else:
					sequence_buffer.append(branch)

	if(len(sequence_buffer) > 0):
			#solve the last sequence_buffer
			solved_sequence = find_and_convert_bigrams2(sequence_buffer, valence_dict)
			new_sequence+=solved_sequence
			sequence_buffer = []

	return new_sequence


def tree_bigram_pairer(subtree, dictionary):

	''' takes NTLK_tree and pairs bigrams based on a measure 
		[AKY 7/23 (TODO) -- pick out a measure for bigram priorities]
	'''

	possible_bigrams = []
	for i in range(len(subtree)-1):
		#collapse ADVPs if it is only 1 word
			if(subtree[i].label() == "ADVP" and len(subtree[i]) == 1):
				subtree[i] = subtree[i][0]
			if(subtree[i+1].label() == "ADVP" and len(subtree[i]) == 1):
				subtree[i+1] = subtree[i+1][0]
			#collect indices of possible bigrams
			if(subtree[i].height() == 2 and subtree[i+1].height() == 2):
				if(subtree[i][0]+" "+subtree[i+1][0] in dictionary.keys()):
					possible_bigrams.append([i,i+1])

	#enumerate list to keep track of orders
	possible_bigrams = list(enumerate(possible_bigrams))

	#give each bigram a measure
	measured_possible_bigrams = []
	for bigram_entry in possible_bigrams:
		measure = random.random()
		entry = list(bigram_entry)+[measure]
		measured_possible_bigrams.append(entry)

	#resolve bigram in order based on measure
	sorted_bigram_entries = sorted(measured_possible_bigrams, key = lambda e:e[2])

	while(len(sorted_bigram_entries) > 0):
		#consume first entry
		id1,id2 = sorted_bigram_entries[0][1]
		outed = subtree.pop(id2)
		subtree[id1][0] += " "+outed[0]
		sorted_bigram_entries.pop(0)

		#find all duplicates of id1 or id2 and pop them
		i_offset = 0
		for i in range(len(sorted_bigram_entries)):
			entry = sorted_bigram_entries[i-i_offset]
			if(id1 in entry[1] or id2 in entry[1]):
				sorted_bigram_entries.pop(i-i_offset)
				i_offset+=1

		#find all ids > id2 and subtract by 1
		for i in range(len(sorted_bigram_entries)):
			entry = sorted_bigram_entries[i]
			if(entry[1][0] > id2):
				entry[1][0] -= 1
				entry[1][1] -= 1

	#recursively do this for all nodes that are non-leaves
	for i in range(len(subtree)):
		if(subtree[i].height() > 2):
			subtree[i] = tree_bigram_pairer(subtree[i], dictionary)

	return subtree


def getTreeIndices(subtree, parent_index = []):

	''' takes in token_tree and returns a 1D sequence of tree indices for every token '''
	
	indices = []
	for i in range(len(subtree)):
		if(isinstance(subtree[i], str)):
			indices.append(parent_index + [i])
		else:
			subtree_indices = getTreeIndices(subtree[i], parent_index+[i])
			indices += subtree_indices

	return indices



def SequenceToTreeIndices(completeTreeIndices, sequence_indices):
	return [completeTreeIndices[i] for i in sequence_indices]

def TreeToSequenceIndices(completeTreeIndices, tree_indices):
	return [completeTreeIndices.index(i) for i in tree_indices]



def tag_negated_in_sequence(sequence, negscope):
	return ["N_" + sequence[i] if i in negscope else sequence[i] for i in range(len(sequence))]

def neg_scope_from_tagged_sequence(sequence):
	return [i for i in range(len(sequence)) if "N_" in sequence[i]]

def tag_negated_in_tree(subtree, negscope, parent_index = []):
	
	''' iterates through the token tree and tags each token that is within negscope 
		[AKY 7/24 (TODO) don't iterate through every branch. start with negscope 
	'''

	indices = []
	for i in range(len(subtree)):
		if(isinstance(subtree[i], str)):
			if(parent_index+[i] in negscope):
				subtree[i] = "N_" + subtree[i]
		else:
			subtree_indices = tag_negated_in_tree(subtree[i], negscope, parent_index+[i])
			indices += subtree_indices

	return indices

def neg_scope_from_tagged_tree(subtree, parent_index = []):

	''' takes a token_tree that has its negated tokens tagged with N_ and returns tree indices on them '''

	indices = []
	for i in range(len(subtree)):
		if(isinstance(subtree[i], str)):
			if("N_" in subtree[i]):
				indices.append(parent_index + [i])
		else:
			subtree_indices = neg_scope_from_tagged_tree(subtree[i], parent_index+[i])
			indices += subtree_indices

	return indices

def get_1D_tokens_from_tree(tree):

	''' takes a token_tree and returns tokens_1D '''

	sequence = []
	for subtree in tree:
		if type(subtree) is list:
			sequence += get_1D_tokens_from_tree(subtree)
		else:
			sequence.append(subtree)
	return sequence














#*------------------------------------------------------------------------------------------*#
#*                             Negation Scope Detection Methods                             *#
#*------------------------------------------------------------------------------------------*#

def detect_neg_scope_tree(subtree, parent_index = []):

	''' takes NLTK tree and labels nodes as negated if previous sibling to the left is a negator 
		returns tree indices
	'''

	neg_scope = []
	for i in range(len(subtree)):
			isNegatorCheck1 = (subtree[i].height() < 3 and isNegator(subtree[i][0]))
			isNegatorCheck2 = (subtree[i].label() == "ADVP" and len(subtree[i]) == 1 and isNegator(subtree[i][0][0]))
			if(isNegatorCheck1 or isNegatorCheck2):
					for j in range(len(subtree)-(i+1)):
							neg_scope.append(parent_index+[j+i+1] )
			elif(subtree[i].height() > 2):
					neg_scope += detect_neg_scope_tree(subtree[i], parent_index+[i])

	return neg_scope


def detect_neg_scope_window(sentence_sequence, window_size = 0, stopwords = [sw.rstrip("\n") for sw in open("dictionaries/stopwords.txt").readlines()]):

	''' takes tokens_1D and labels the proceeding window_size number of tokens as negated
		there is a heuristic recursive way of dealing with double negatives
		returns a 1D list of 1D indices
	'''
	
	neg_scope = []
	num_scopes = 0
	last_scope_count = 0

	for i in range(len(sentence_sequence)):

			if(isNegator(sentence_sequence[i])):
					num_scopes += 1
					last_scope_count = 0

			elif(num_scopes > 0):
					for j in range(num_scopes):
							neg_scope.append(i)

					if(window_size > 0): #window_size = 0 signifies end-of-sentence scope
						#[aky 7/4 - ignore stopwords]
							if(sentence_sequence[i] not in stopwords):
									last_scope_count += 1
							if(last_scope_count >= window_size):
									num_scopes = 0
									last_scope_count = 0

	return neg_scope

def resolve_double_negative(neg_scope):
	
	''' resolves double negatives in a negscope for 1D indices '''

	new_thing = []
	for coord in neg_scope:
			if(neg_scope.count(coord)%2==1 and new_thing.count(coord) == 0):
					new_thing.append(coord)
	return new_thing















#*------------------------------------------------------------------------------------------*#
#*                                 Negation Resolution Methods                              *#
#*------------------------------------------------------------------------------------------*#

def negate(valence, affirm_word, neg_res, all_dicts = None):

	''' negates valence of given token based on neg_res '''

	if(neg_res == "INVERT"):
		weight = 1.0
		if(valence == -999):
			return -999
		else:
			return -weight*valence
	
	elif(neg_res == "AFFIRM_SHIFT"):
		if(valence == -999):
			return -999
		else:
			return -0.065916 -0.363218*valence
	
	elif(neg_res == "ANTONYM_LOOKUP"):
		if(affirm_word is None):
			print("ERROR! antonym lookup on null (parse + parse?)")
			return -999
		affirm_word = affirm_word.lstrip("N_")
		antonym_word = antonym_lookup(affirm_word)

		if(antonym_word is None):
			negated_valence = -999
		else:
			negated_valence = antonym_lookup_negate(antonym_word, all_dicts["valence"])
		return negated_valence

	elif(neg_res == "MEANINGSPEC_FREQ"):
		#[AKY 7/23 - Is this right? if valence is -999, ignore?]
		if(valence == -999):
			return -999
		else:
			return meaningSpec_freq(valence, all_dicts)	

	elif(neg_res == "MEANINGSPEC_FREQDP"):
		#[AKY 7/23 - Is this right? if valence is -999, ignore?]
		if(valence == -999):
			return -999
		else:
			if(affirm_word):
				affirm_word = affirm_word.lstrip("N_")
			return meaningSpec_freqdp(valence, affirm_word, all_dicts)














#*------------------------------------------------------------------------------------------*#
#*                               Sentiment Composition Methods                              *#
#*------------------------------------------------------------------------------------------*#


def tree_composition(valence_tree, token_tree, neg_scope, neg_res, all_dicts = None, parent_index = []):

	'''
		composes valences heirarchically bottom up in the tree structure negating valence of token or node
		#[aky 7/5 - need valence_tree and token_tree here to allow for affirm word to be passed]
	'''
	
	valence = []
	for i in range(len(valence_tree)):
		v_subtree = valence_tree[i]
		t_subtree = token_tree[i]
		current_index = parent_index+[i]

		if(isinstance(v_subtree, float) or isinstance(v_subtree, int)):
			if(current_index in neg_scope):
				affirm_word = t_subtree.lstrip("N_")
				v_subtree = negate(v_subtree, affirm_word, neg_res, all_dicts)
			
			if(v_subtree != -999):
				valence.append(v_subtree)
				
		else:
			valence_from_tree = tree_composition(v_subtree, t_subtree, neg_scope, neg_res, all_dicts, current_index)
			if(current_index in neg_scope):
				#[aky 7/5 - in tree_composition(): affirm_word == None because it is a node -- what happens when we pass in a None?]
				valence_from_tree = negate(valence_from_tree, None, neg_res, all_dicts)
			
			if(valence_from_tree != -999):
				valence.append(valence_from_tree)

	if(len(valence) == 0):
		return -999
	else:
		return sum(valence)/float(len(valence))

def flat_composition(valence_sequence):
	valence_sequence = list(filter(lambda a: a != -999, valence_sequence))
	
	if(len(valence_sequence) == 0):
		avg_valence = -999
	else:
		avg_valence = sum(valence_sequence)/len(valence_sequence)
	return avg_valence













#*--------------------------------------------------------------------------------------*#
#*                               Antonym Dictionary Method                              *#
#*--------------------------------------------------------------------------------------*#

def antonym_lookup_negate(word, valence_dict):
	try:
		antonym_word = antonym_lookup(word)
	except RuntimeError as re:
		antonym_word = None
		print("antonym_lookup({}) error: {}".format(word, re)) #antonym_lookup(us) error: maximum recursion depth exceeded
	
	if(antonym_word is None):
		antonym_valence = -999
	else:
		antonym_valence = getValence(antonym_word, valence_dict)

	return antonym_valence

def antonym_lookup(word):
	try:
		antonym= []
		keyword_synsets = wn.synsets(word)

		# print ('synsets', keyword_synsets)

		word_list_lemma = change_to_lemma(keyword_synsets)

		# print 'word_list_lemma', word_list_lemma
		antonym = check_has_antonym(word_list_lemma)

		#if there is no antonym for all keyword_synset in keyword_list,
		if antonym ==[]:
			antonym = mother(word_list_lemma)

		#antonym output as a list
		for anto in antonym:
			if '_' not in anto:
				return anto
	except RuntimeError as re:
		#print("antonym_lookup({}):{}".format(word, re)) #"us","me","50"
		return None
	# return antonym[0]

def mother(keyword_lemma):
	track_antonym= []

	if len(keyword_lemma) >25:
		return track_antonym

	# check synset (synonyms)
	keyword_synsets = check_has_synset(keyword_lemma)
	word_list_lemma = change_to_lemma(keyword_synsets)

	track_antonym = check_has_antonym(word_list_lemma)
	if track_antonym !=[]:
		return track_antonym

	#if there is no antonym for all keyword_synset in keyword_list,
	#get attribute of the keywords
	attribute_synset = check_has_attribute(keyword_lemma)

	#if there are keyword->attribute
	if attribute_synset !=[]:
		attribute_lemma = change_to_lemma(attribute_synset)
		track_antonym = check_has_antonym(attribute_lemma)

		if track_antonym != []:
			#if keyword->attribute has antonym, return that
			return track_antonym

	#if attribute_list==[] or keyword->attribute->antonym ==[]
	#check pertainym of keyword
	pertainym_list = check_has_pertainym(keyword_lemma)

	#if there are keyword->pertainym
	if pertainym_list!=[]:
		track_antonym = check_has_antonym(pertainym_list)

		if track_antonym !=[]:
			return track_antonym

		#if there is keyword->pertainym but no keyword->pertainym->antonym
		track_antonym = mother(pertainym_list)

		if track_antonym !=[]:
			return track_antonym

	#if pertainym_list ==[], check derivationally_related_forms
	derivation_list = check_has_derivation(keyword_lemma)

	if derivation_list !=[]:
		track_antonym = check_has_antonym(derivation_list)

		if track_antonym !=[]:
			return track_antonym

		#print 'derivation_list', derivation_list
		track_antonym = mother(derivation_list)
		if track_antonym !=[]:
			return track_antonym

	#if keyword->derivation_list or keyword->derivation->antonym ==[]
	#check similar to
	similar_list = check_has_similar(keyword_lemma)

	if similar_list !=[]:
		track_antonym = check_has_antonym(similar_list)

		if track_antonym !=[]:
			return track_antonym

		track_antonym = mother(similar_list)
		if track_antonym !=[]:
			return track_antonym

	#If all means exhausted and still no relation
	return track_antonym

def change_to_synset(lemma_list):
	synset_list=[]

	for lemma in lemma_list:
		synset= lemma.synset()
		synset_list.append(synset)

	# print 'synset_list', synset_list
	return synset_list

def change_to_lemma(keyword_synsets):
	#switch to lemma for antonyms
	word_list_lemma=[]
	for synset in keyword_synsets:
		temp_list_lemma= synset.lemmas()
		for temp in temp_list_lemma:
			word_list_lemma.append(temp)

	return word_list_lemma

def check_has_synset(keyword_lemma):
	# print 'check_has_synset'
	synset_list = []

	for lemma in keyword_lemma:
		temp_synset = wn.synsets(lemma.name())

		if temp_synset != []:
			for synset in temp_synset:
				synset_list.append(synset)
	return synset_list

def check_has_attribute(keyword_lemma):
	# print 'check_has_attribute'
	attribute_list=[]

	keyword_synset_list = change_to_synset(keyword_lemma)

	for keyword_synset in keyword_synset_list:
		temp_attribute_list= keyword_synset.attributes()
		if temp_attribute_list !=[]:
			for temp in temp_attribute_list:
				attribute_list.append(temp)

	return attribute_list

def check_has_antonym(word_list):
	# print 'check_has_antonym'
	antonym=[]

	for lemma in word_list:
		# if lemma.antonyms():
		# 	antonym.append(lemma.antonyms()[0].name())

		antonym_list =lemma.antonyms()

		if antonym_list != []:
			for antonym_word in antonym_list:
				antonym.append(antonym_word.name())

	return antonym

def check_has_pertainym(word_list_lemma):
	# print 'check_has_pertainym'
	pertainym_list=[]

	for lemma in word_list_lemma:
		temp_pertainym_list = lemma.pertainyms()

		if temp_pertainym_list!= []:
			for temp in temp_pertainym_list:
				pertainym_list.append(temp)

	# print 'pertainym_list',pertainym_list
	return pertainym_list

def check_has_derivation(word_list_lemma):
	# print 'check_has_derivation'

	derivation_list=[]

	for lemma in word_list_lemma:
		temp_derivation_list = lemma.derivationally_related_forms()

		if temp_derivation_list!= []:
			for temp in temp_derivation_list:
				derivation_list.append(temp)

	# print 'derivation_list',derivation_list
	return derivation_list
			
def check_has_similar(word_list):
	# print 'check_has_similar'
	similar_list= []

	for synset in word_list:
		temp_similar_list=synset.similar_tos()
		if temp_similar_list != []:
			for temp in temp_similar_list:
				similar_list.append(temp)

	# print 'similar_list', similar_list
	return similar_list

















#*--------------------------------------------------------------------------------------*#
#*                              Meaning Specificity Method                              *#
#*--------------------------------------------------------------------------------------*#

def get_cat(value):
	bins = [-1.0, -0.96, -0.92, -0.88, -0.84, -0.8, -0.76, -0.72, -0.68, -0.64, -0.6, -0.56, -0.52, -0.48, -0.44, -0.4, -0.36, -0.32, -0.28, -0.24, -0.2, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0]

	for i in range(len(bins)):
		if value == 1:
			return 1

		else:
			if bins[i] <= value < bins[i+1]:
				return bins[i]
				
def inferDist(valence, distribution_dict):
	cat = '%.3f'%(get_cat(valence))
	# if valence < 0:
	# 	cat = math.floor(valence*10)/10
	# else:
	# 	cat = math.ceil(valence*10)/10

	# get mu and sigma of distribution
	data = distribution_dict[str(cat)]

	random_freq_mu = data["frequency"][0]
	random_freq_sigma = data["frequency"][1]

	random_freq = np.random.normal(random_freq_mu, random_freq_sigma, 1)
	while (random_freq < 0):
		random_freq = np.random.normal(random_freq_mu, random_freq_sigma, 1)
	
	random_dp_mu = data["dispersion"][0]
	random_dp_sigma = data["dispersion"][1]

	random_dp = np.random.normal(random_dp_mu, random_dp_sigma, 1)
	while (random_dp < 0):
		random_dp = np.random.normal(random_dp_mu, random_dp_sigma, 1)

	# random_mi_mu = data["mi"][0]
	# random_mi_sigma = data["mi"][1]

	# random_mi = np.random.normal(random_mi_mu, random_mi_sigma, 1)
	
	return {"frequency" : random_freq[0], "dispersion": random_dp[0]}


def meaningSpec_freq(Affirm, all_dicts):
	Freq = inferDist(Affirm, all_dicts["dist"])["frequency"]
	return -7.747130e-02 -3.850748e-01*Affirm + Freq*5.326080e-09


# def meaningSpec_freqdp(Affirm, distribution_dict):
# 	inferred = inferDist(Affirm, distribution_dict)
# 	Freq = inferred["frequency"]
# 	DP = inferred["dispersion"]

# 	return -6.112665e-02 +Affirm*-3.851552e-01+Freq*7.751644e-09+DP*-2.260250e+00+Freq*DP*-1.976151e-06

def meaningSpec_freqdp(Affirm, affirm_word, all_dicts):
	Freq = -999
	DP = -999

	if (affirm_word in all_dicts["freq"]):
		Freq = all_dicts["freq"][affirm_word]
	if (affirm_word in all_dicts["dp"]):
		DP = all_dicts["dp"][affirm_word]

	if (Freq != -999 and DP != -999):
		return -6.112665e-02 +Affirm*-3.851552e-01+Freq*7.751644e-09+DP*-2.260250e+00+Freq*DP*-1.976151e-06

	else:
		try:
			inferred = inferDist(Affirm, all_dicts["dist"])

			if (Freq == -999):
				Freq = inferred["frequency"]
			if (DP == -999):
				DP = inferred["dispersion"]

			return -6.112665e-02 +Affirm*-3.851552e-01+Freq*7.751644e-09+DP*-2.260250e+00+Freq*DP*-1.976151e-06

		except:
			print ("No freq and dispersion infered and found. ")
			return Affirm
