"""
Sentiment Analysis Pipeline: a lexicon-based approach
	by UCI Computation of Language Laboratory 2018
"""

import argparse
import model
import os
import json
from nltk.tree import Tree
from SentimentModelFunctions import *
"""


"""
if __name__ == "__main__":

	'''define arguments '''
	pwd = os.getcwd()

	parser = argparse.ArgumentParser()
	parser.add_argument('-mode', type=str, default = "eval", help="eval or results")
	parser.add_argument('-config_path', type=str, default = "config.txt")
	parser.add_argument('-review_offset', type=int, default = 0)
	parser.add_argument('-sentence_offset', type=int, default = 0)
	parser.add_argument('-corenlp_server_port', type=str, default = "9000")
	parser.add_argument('-save_path', type=str, default = "results")

	args = parser.parse_args()
	config = json.load(open(args.config_path, "r"))

	
	''' init pipeline '''

	pipeline = model.pipeline(config)
	#"rid": 25, "sid": 80
	pipeline.init_review_iterator()#review_offset = 51, sentence_offset = 210)
	pipeline.init_negtool_iterator()#sentence_offset = 210)
	
	''' define pipeline method combinations '''

	composition_methods = [ "FLAT", "PARSETREE" ]
	neg_detection_methods = [[ "WINDOW", "NEGTOOL"], [ "WINDOW", "NEGTOOL", "PARSETREE"]]
	neg_res_methods = [ "ANTONYM_LOOKUP", "AFFIRM_SHIFT", "INVERT", "MEANINGSPEC_FREQDP"]

	''' to save results '''
	models_dict = json.load(open(config["MODELS_LIST"], "r"))

	if(args.mode == "results"):
		import datetime
		results_file_name = "results_{}".format(datetime.datetime.now().strftime("%Y-%m-%d"))
		
		ids = [int(fn[-5:-4]) for fn in os.listdir("results") if results_file_name in fn]
		if(len(ids) == 0):
			next_fn_id = 0
		else:
			next_fn_id = max(ids)+1
		
		results_file_name += "_%d.txt" % next_fn_id
		results_file = open("results/"+results_file_name, "a+")


	''' run the pipeline '''

	for i in range(3):

		pipeline.next_review()

		#init sentiment results for all models per review
		sentiment_results = [0 for i in range(len(models_dict.keys()))] 

		#run pipeline through all combinations
		for j in range(len(composition_methods)):
			pipeline.comp_method = composition_methods[j]

			for k in range(len(neg_detection_methods[j])):
				pipeline.neg_scope_method = neg_detection_methods[j][k]
				pipeline.detect_neg_scope()

				for neg_res_method in neg_res_methods:
					pipeline.neg_res_method = neg_res_method

					model_name = " ".join([pipeline.neg_scope_method, pipeline.neg_res_method, pipeline.comp_method])
					if model_name == "PARSETREE ANTONYM_LOOKUP PARSETREE":
						continue
					pipeline.neg_resolution()
					
					#get sentiment results and save it in the list
					sentiment_result = pipeline.compose()#round(pipeline.compose(),5)
					model_id = models_dict[model_name]
					sentiment_results[model_id] = sentiment_result

					if(args.mode == "eval"):
						print("{}-{} : {}".format(pipeline.review_id, model_name, sentiment_result))

		

		#write results to file
		if(args.mode == "results"):
			review_result_dict_line =  {"r_id": pipeline.review_id, "result": sentiment_results}
			write_line = json.dumps(review_result_dict_line) + "\n"
			print(write_line)
			results_file.write(write_line)

	if(args.mode == "results"):
		results_file.close()
				
'''
TO TEST CUSTOM REVIEWS:
# test_review = "I not ever really liked super super bad food ."
# pipeline.next_review(test_review)
'''

	


'''

TODO:
1. check each function, organise them into appropriate scripts, mark when checked
2. unit tests
3. window(n) flat
4. none
5. how do you want to analyze the results?

'''
