from pycorenlp import StanfordCoreNLP
from SentimentModelFunctions import *
from nltk.tree import Tree
import itertools

class review_struct():
	''' 
	review_struct
	-------------
	
	the review object that holds all the data structures of parsed tokens and their corresponding valences

	'''

	def __init__(self, review = "", rating = None):
		self.review_string = review
		self.review_rating = rating

		self.list_tokenized_1D = []
		self.list_NLTK_trees = [] #NLTK Tree objects
		self.list_token_trees = [] #MD tokens
		self.list_tree_indices = []
		self.list_negscopes = []
		self.negtool_negscopes = []

		self.list_valence_1D = []
		self.list_valence_trees = []

		self.orig_list_tokenized_1D = ""
		self.orig_list_valence_1D = ""
		self.orig_list_token_trees = ""
		self.orig_list_valence_trees = ""
		self.orig_list_tree_indices = ""
		self.orig_list_negscopes = ""

		self.review_size = 0

	def create(self, corenlp):
		''' 
			parses the raw string review into sentences then tokens as well as a constituency parse 
			also intializes all the variables
		'''

		assert corenlp is not None

		output = corenlp.annotate(self.review_string, properties={
		'annotators': 'tokenize, ssplit, parse',
		'outputFormat': 'json',
		'parse.model' : 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
		})

		if(type(output) is str): #TypeError: eval() arg 1 must be a string, bytes or code object
			output = eval(output)
		self.size = len(output['sentences'])

		#organize into 1D and MD(multi-dimensional --> Tree)
		for i in range(self.size):
			tokenized_1D = [token_json['word'] for token_json in output['sentences'][i]['tokens']]
			self.list_tokenized_1D.append(tokenized_1D)

			parsetree = Tree.fromstring(output['sentences'][i]['parse'])
			self.list_NLTK_trees.append(parsetree) #NLTK Tree objects
			self.list_token_trees.append(map_token_tree(parsetree)) #MD tokens
			self.list_tree_indices.append(getTreeIndices(self.list_token_trees[i]))
			self.list_valence_1D.append([])
			self.list_valence_trees.append([])

		#save original as string json
		self.orig_list_token_trees = json.dumps({"Tree" : self.list_token_trees})
		self.orig_list_tokenized_1D = json.dumps({"1D" : self.list_tokenized_1D})
		self.orig_list_tree_indices = json.dumps({"Tree Indices" : self.list_tree_indices})
		self.orig_list_NLTK_trees = [tree.copy(deep = True) for tree in self.list_NLTK_trees]

	def set_negscopes(self, negscopes):
		''' saves the negscopes '''
		#AKY probably don't need this
		self.list_negscopes = negscopes
		self.orig_list_negscopes = json.dumps({"Negscopes" : self.orig_list_negscopes})

	def print(self):
		''' for debugging purposes '''

		print("\n"+"-"*50)

		print("Raw Review: {}".format(self.review_string))

		print("List Token Trees: {}".format(self.list_token_trees))
		print("List Valence Trees: {}".format(self.list_valence_trees))
		print("List Tree Indices: {}".format(self.list_tree_indices))
		
		print("List Tokenized 1D: {}".format(self.list_tokenized_1D))
		print("List Valence 1D: {}".format(self.list_valence_1D))
	
		print("List Negscopes: {}".format(self.list_negscopes))
		
		print("-"*50 + "\n")

class pipeline():

	'''
	the pipeline

		a singleton
		loads up all the dictionaries
		sets up the review and negtool iterators
		handles MANY kinds of sentiment analysis methods
		does negscope, neg_res, and composes valences
	'''

	def __init__(self, config):
		self.config = config

		self.review_id = 0
		self.review = None #holds one review at a time
		self.sentence_id = 0

		self.use_negtool = True
		

		#load up dictionaries
		self.valence_dict = json.loads(open(config["VALENCE_DICT"]).read())
		self.distribution_dict = json.loads(open(config["MEANING_SPEC_DISTRIBUTION_DICT_PATH"]).read())
		self.freq_dict = json.loads(open(config["MEANING_SPEC_FREQ_PATH"]).read())
		self.dp_dict = json.loads(open(config["MEANING_SPEC_DP_PATH"]).read())
		self.all_dicts = {	"valence":self.valence_dict,
							"dist":self.distribution_dict,
							"freq":self.freq_dict,
							"dp":self.dp_dict }
		self.stopwords = [sw.rstrip("\n") for sw in open(config["STOPWORDS_PATH"], 'r').readlines()]
		#parameter for the window negscope method
		self.negscope_window_size = 4


	'''the iterators'''

	def init_review_iterator(self, reviews_path = "", review_offset = 0, sentence_offset = 0, corenlp_server_port = "9000"):
		if(reviews_path == ""):
			reviews_path = self.config["REVIEWS_PATH"]
		self.review_iterator = review_iterator(reviews_path, review_offset, sentence_offset, corenlp_server_port)
		
	def init_negtool_iterator(self, negtool_negscope_path = "", sentence_offset = 0):
		if not self.use_negtool:
			return
		if(negtool_negscope_path == ""):
			negtool_negscope_path = self.config["NEGTOOL_NEGSCOPE"]
		self.negtool_iterator = negtool_iterator(negtool_negscope_path)

		self.negtool_iterator.get_next(review_size = sentence_offset)

	def next_review(self, test_review = ""):
		if(test_review != ""):
			self.review = self.review_iterator.get_test_review(test_review)
		else:
			self.review = self.review_iterator.get_next()
			
			if(self.use_negtool):
				self.review.negtool_negscopes = self.negtool_iterator.get_next(self.review.size, self.review_iterator.current_review_id, self.review_iterator.current_sentence_id)
			
			self.review_id = self.review_iterator.current_review_id
			self.sentence_id = self.review_iterator.current_sentence_id
		
		return self.review

	''' The Pipeline '''


	def detect_neg_scope(self):


		''' 
			calls the necessary negation scope detection methods in the right format 
			sometimes the structure/indices of the tree or sequence changes due to bigram pairing priorities 
		'''


		list_negscopes = []


		if(self.neg_scope_method == "WINDOW" and self.comp_method == "FLAT"):
			#RESET(self.review.list_tokenized_1D)
			self.review.list_tokenized_1D = json.loads(self.review.orig_list_tokenized_1D)["1D"]


			for i in range(self.review.size):
				self.review.list_tokenized_1D[i] = find_and_convert_bigrams2(self.review.list_tokenized_1D[i], self.valence_dict)
				list_negscopes.append(resolve_double_negative(detect_neg_scope_window(self.review.list_tokenized_1D[i], self.negscope_window_size, self.stopwords)))

				#convert 1D tokens into affirmative valence [AKY 7/24 - SAVE for RESET]
				valence_1D = [getValence(word, self.valence_dict) for word in self.review.list_tokenized_1D[i]]
				self.review.list_valence_1D[i] = valence_1D
			

			#SAVE(list_valence_1D FOR NEG_RES)
			self.review.orig_list_valence_1D = json.dumps({"1D_valence" : self.review.list_valence_1D})
				

		elif(self.neg_scope_method == "NEGTOOL" and self.comp_method == "FLAT"):
			#RESET(self.review.list_tokenized_1D)
			self.review.list_tokenized_1D = json.loads(self.review.orig_list_tokenized_1D)["1D"]

			self.review.set_negscopes(self.review.negtool_negscopes)

			list_neg_tagged_1D = []
			for i in range(self.review.size):
				#tag negated words
				list_neg_tagged_1D.append(tag_negated_in_sequence(self.review.list_tokenized_1D[i], self.review.negtool_negscopes[i]))
				#convert to bigrams with negation scope as priority
				self.review.list_tokenized_1D[i] = find_and_convert_bigrams2(list_neg_tagged_1D[i], self.valence_dict)
				#add updated negscope indices 
				list_negscopes.append(neg_scope_from_tagged_sequence(self.review.list_tokenized_1D[i]))
				#convert 1D tokens into affirmative valence [AKY 7/24 - SAVE for RESET]
				valence_1D = [getValence(word, self.valence_dict) for word in self.review.list_tokenized_1D[i]]
				self.review.list_valence_1D[i] = valence_1D
			
			#SAVE(list_valence_1D FOR NEG_RES)
			self.review.orig_list_valence_1D = json.dumps({"1D_valence" : self.review.list_valence_1D})
	

		# elif(self.neg_scope_method == "PARSETREE" and self.comp_method == "FLAT"):
		# 	#[AKY 7/24 - RESET(self.review.list_NLTK_trees)
		# 	self.review.list_NLTK_trees = [tree.copy(deep = True) for tree in self.review.orig_list_NLTK_trees]

		# 	for i in range(self.review.size):
		# 		self.review.list_NLTK_trees[i] = tree_bigram_pairer(self.review.list_NLTK_trees[i], self.valence_dict)
		# 		negscope = detect_neg_scope_tree(self.review.list_NLTK_trees[i], parent_index = [])
		# 		"""
		# 		[AKY 7/23 (TODO) -- tag all leaves in negscope and resolve double negatives, flatten, retrieve 1D negscope indices]
		# 		"""


		elif(self.neg_scope_method == "WINDOW" and self.comp_method == "PARSETREE"):
			#RESET(self.review.list_NLTK_trees)
			self.list_NLTK_trees = [tree.copy(deep = True) for tree in self.review.orig_list_NLTK_trees]

			for i in range(self.review.size):
				#convert to bigrams with tree heirarchy as priority
				self.review.list_NLTK_trees[i] = tree_bigram_pairer(self.review.list_NLTK_trees[i], self.valence_dict)
				#update token_trees
				self.review.list_token_trees[i] = map_token_tree(self.review.list_NLTK_trees[i])
				#update token_tree_indices
				self.review.list_tree_indices[i] = getTreeIndices(self.review.list_token_trees[i])
				#flatten new token_trees into 1D
				self.review.list_tokenized_1D[i] = get_1D_tokens_from_tree(self.review.list_token_trees[i])
				#perform window negscope with double negative resolution
				negscope = resolve_double_negative(detect_neg_scope_window(self.review.list_tokenized_1D[i], self.negscope_window_size, self.stopwords)) 
				#convert 1D negscopes to tree indice negscopes
				list_negscopes.append(SequenceToTreeIndices(self.review.list_tree_indices[i], negscope))

				#convert MD tokens into affirmative valences, take care of neg res in composition
				valence_tree = map_valence_tree(self.review.list_token_trees[i], self.valence_dict)
				self.review.list_valence_trees[i] = valence_tree
			
			#SAVE for NEGRES(list_valence_trees)
			self.review.orig_list_valence_trees = json.dumps({"Tree_valence" : self.review.list_valence_trees})
		
		

		elif(self.neg_scope_method == "NEGTOOL" and self.comp_method == "PARSETREE"):
			#RESET(self.review.list_token_trees)
			self.review.list_token_trees = json.loads(self.review.orig_list_token_trees)["Tree"]

			self.review.set_negscopes(self.review.negtool_negscopes)
			for i in range(self.review.size):
				#get tree indices from current unconverted(bigram) list_token_trees
				self.review.list_tree_indices[i] = getTreeIndices(self.review.list_token_trees[i])
				#convert negtool negscopes to tree indices
				negscope = SequenceToTreeIndices(self.review.list_tree_indices[i], self.review.negtool_negscopes[i])
				#tag negated words with "N_" prefix in token_tree]
				tag_negated_in_tree(self.review.list_token_trees[i], negscope)
				#make the valence_trees from token_trees once you've dealt with the negation handling (structure changes depending on what's in negation scope)]
				self.review.list_token_trees[i] = find_and_convert_bigrams_recursive(self.review.list_token_trees[i], self.valence_dict)
				#use new bigram_resolved with tagged negated words to update tree indice negscope
				list_negscopes.append(neg_scope_from_tagged_tree(self.review.list_token_trees[i]))

				#convert MD tokens into affirmative valences, take care of neg res in composition
				valence_tree = map_valence_tree(self.review.list_token_trees[i], self.valence_dict)
				self.review.list_valence_trees[i] = valence_tree
			
			#SAVE for NEGRES(list_valence_trees)
			self.review.orig_list_valence_trees = json.dumps({"Tree_valence" : self.review.list_valence_trees})


		elif(self.neg_scope_method == "PARSETREE" and self.comp_method == "PARSETREE"):
			#RESET(self.review.list_NLTK_trees)
			self.review.list_NLTK_trees = [tree.copy(deep = True) for tree in self.review.orig_list_NLTK_trees]

			for i in range(self.review.size):
				self.review.list_NLTK_trees[i] = tree_bigram_pairer(self.review.list_NLTK_trees[i], self.valence_dict)
				list_negscopes.append(detect_neg_scope_tree(self.review.list_NLTK_trees[i], parent_index = []))
				
				self.review.list_token_trees[i] = map_token_tree(self.review.list_NLTK_trees[i])
				
				#convert MD tokens into affirmative valences, take care of neg res in composition
				valence_tree = map_valence_tree(self.review.list_token_trees[i], self.valence_dict)
				self.review.list_valence_trees[i] = valence_tree

			#SAVE for NEGRES(list_valence_trees)
			self.review.orig_list_valence_trees = json.dumps({"Tree_valence" : self.review.list_valence_trees})

		self.review.set_negscopes(list_negscopes)




	def neg_resolution(self):


		''' 
			if flat, resolve negation
			if parse tree, just resets the valence tree
		'''

		
		if(self.comp_method == "FLAT"):
			#RESET_NEG_RES(list_valence_1D for negres)
			self.review.list_valence_1D = json.loads(self.review.orig_list_valence_1D)["1D_valence"]

			for i in range(self.review.size):
				
				#negation resolution
				for idx in self.review.list_negscopes[i]:
					valence = self.review.list_valence_1D[i][idx]
					affirm_word = self.review.list_tokenized_1D[i][idx]
					self.review.list_valence_1D[i][idx] = negate(self.review.list_valence_1D[i][idx], affirm_word, self.neg_res_method, self.all_dicts)

		elif(self.comp_method == "PARSETREE"):
			for i in range(self.review.size):
				#RESET for NEGRES(list_valence_trees)
				self.review.list_valence_trees = json.loads(self.review.orig_list_valence_trees)["Tree_valence"]
			

	def compose(self):


		''' 
			takes current review structure and aggregates the valences  
			if flat, does just that
			if parsetree, does neg res while doing that. ooo fancy
		'''

		
		sentiment_scores = []

		for i in range(len(self.review.list_tokenized_1D)):

			if (self.comp_method == "FLAT"):
				sentiment = flat_composition(self.review.list_valence_1D[i])

			elif(self.comp_method == "PARSETREE"):
				sentiment = tree_composition(self.review.list_valence_trees[i], self.review.list_token_trees[i], self.review.list_negscopes[i], self.neg_res_method, self.all_dicts)
			sentiment_scores.append(sentiment)

		sentiment_scores = list(filter(lambda a: a != -999, sentiment_scores))
		if(len(sentiment_scores) == 0):
			avg_valence = -999
		else:
			avg_valence = sum(sentiment_scores)/float(len(sentiment_scores))

		return avg_valence


class review_iterator():
	'''
		iterates through and returns individual reviews and their corresponding score
	'''
	def __init__(self, reviews_path, review_offset = -1, sentence_offset = -1, corenlp_server_port = "9000"):

		

		self.reviews_file = open(reviews_path, "r")

		#review iterator
		self.current_review_id = review_offset
		self.current_sentence_id = sentence_offset
		
		for i in range(review_offset):
			self.reviews_file.readline()

		#review structures
		self.corenlp = StanfordCoreNLP('http://localhost:%s' % corenlp_server_port)
		self.current_review = None

	def get_next(self):
		'''
		returns next review as review_struct or None if EOF
		'''
		
		
		line = self.reviews_file.readline()
		if(line == ""):
			print("EOF: current_review_id: %d" % self.current_review_id)
			return None

		review_json = json.loads(line)
		review = review_json["reviewText"]
		truth = review_json["overall"]

		self.current_review = review_struct(review, truth)
		self.current_review.create(self.corenlp)

		self.current_review_id += 1
		self.current_sentence_id += self.current_review.size

		return self.current_review

	def get_test_review(self, test_review):

		self.current_review = review_struct(test_review, 0)
		self.current_review.create(self.corenlp)

		return self.current_review

class negtool_iterator():
	
	''' iterates through negtool file returning negscopes per review '''

	def __init__(self, negtool_negscope_path):
		self.negtool_neg_scopes_file = open(negtool_negscope_path, "r")

	def get_next(self, review_size, review_id = -999, sentence_id = -999):
		negtool_negscopes = []

		for i in range(review_size):
			try:
				#Format #1
				#sentence_negscope = json.loads(self.negtool_neg_scopes_file.readline())[str(self.sentence_id+i)]["neg_scope"]

				#Format #2
				sentence_negscope = json.loads(self.negtool_neg_scopes_file.readline())
				sentence_negscope = sentence_negscope[list(sentence_negscope.keys())[0]]["neg_scope"]

				#Format #3
				#sentence_negscope = json.loads(self.negtool_neg_scopes_file.readline())["negscope"]

				negtool_negscopes.append(list(itertools.chain(*sentence_negscope)))
			except Exception as e:
				print("Error: {}... Could not load negtool_negscope @ review_id: {}, sentence_id: {}".format(e, review_id, sentence_id+i))

		if (len(negtool_negscopes) != review_size):
			print("Error: Negtool review size did not match raw review size!")

		return negtool_negscopes