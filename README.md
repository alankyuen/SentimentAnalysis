# UCI CoLaLab Sentiment Anaylsis Pipeline
---------------------------------------

# Installation

	Anaconda 3
		conda create -n SAPipelineRE python=3.6

			    ca-certificates: 2018.03.07-0           
			    certifi:         2018.4.16-py36_0       
			    libcxx:          4.0.1-h579ed51_0       
			    libcxxabi:       4.0.1-hebd6815_0       
			    libedit:         3.1.20170329-hb402a30_2
			    libffi:          3.2.1-h475c297_4       
			    ncurses:         6.1-h0a44026_0         
			    openssl:         1.0.2o-h26aff7b_0      
			    pip:             10.0.1-py36_0          
			    python:          3.6.5-hc167b69_1       
			    readline:        7.0-hc1231fa_4         
			    setuptools:      39.1.0-py36_0          
			    sqlite:          3.23.1-hf1716c9_0      
			    tk:              8.6.7-h35a86e2_3       
			    wheel:           0.31.1-py36_0          
			    xz:              5.2.4-h1de35cc_4       
			    zlib:            1.2.11-hf3cbc9b_2   
			    
	Python 3.6
		- pathlib
			Successfully installed pathlib-1.0.1
		
		- pycorenlp
			Successfully installed chardet-3.0.4 idna-2.6 pycorenlp-0.3.0 requests-2.18.4 urllib3-1.22
		
		- NLTK (wordnet)
			python3.6
				>> import nltk
				>> nltk.download("wordnet")
			Successfully installed NLTK-3.3 six-1.11.0
		
		- numpy
			Successfully installed numpy-1.14.3

	# Change "config.txt"
	Replace all paths to fit your system

	# Change "models.txt"
	List all models to be evaluated in the pipeline:
		they must be in the exact format "NEGATION_SCOPE_DETECTION NEGATION_RESOLUTION AGGREGATION_METHOD"

	CoreNLP Server [https://stanfordnlp.github.io/CoreNLP/corenlp-server.html]
		Once you have the correct files (listed below are the folders we used when the code was written; any other version is not guaranteed to work)
			-stanford-corenlp-full-2018-02-27
			-stanford-ner-2018-02-27
			-stanford-parser-full-2018-02-27
			-stanford-postagger-2018-02-27

		Then you can run "python run_server.py -port 9000" on a separate thread (terminal; it must be running during your execution of the main python script)


	You can print sentiment output directly into the output by running
		python main.py -mode eval
	or you can have it write to a text file by running
		python main.py -mode results

	We supplied a sample file of reviews that are preprocessed, as well as an example negtool negscope file that tells the pipeline which words are being negated as determined by the 3rd party negtool program. For any other reviews, the user must supply their own files.


