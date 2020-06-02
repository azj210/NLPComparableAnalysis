import math
import edgar
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import copy
import json
import csv
from collections import defaultdict

#Obtaining SEC docs
def extract_docs(comp_name,cik):
	company = edgar.Company(comp_name, cik)
	tree = company.getAllFilings(filingType = "10-K")
	docs = edgar.getDocuments(tree, noOfDocuments=3)
	if ("This application relies heavily on JavaScript, you" in docs[0]):
		return docs[1]
	else:
		return docs[0]

#cleaning thet document to retain only the first half and to keep a unified format for all documents
def clean_doc(doc):
 #splitting the doc in two since most info later is balance sheet and numbers. Also eliminate most of intro.
	cleaned_doc = ""
	for i in range(5000,int(len(doc)//2)):
		cleaned_doc += doc[i]
	keep_letters = re.sub('[^a-zA-Z]', ' ', cleaned_doc)
	words = keep_letters.lower().split()
	stop_words = set(stopwords.words("english"))
	return_words =  [i for i in words if not i in stop_words]
	return_words2 = ' '.join(return_words)
	return return_words2

#stemmer to obtain stem words
stemmer = PorterStemmer()
def stem_words(words_list, stemmer):
 	return [stemmer.stem(word) for word in words_list]

#stemming the document
def stemmed_doc(text):
	tokens = nltk.word_tokenize(text)
	stems = stem_words(tokens, stemmer)
	stemmed = ""
	for i in stems:
		stemmed += i + " "
	return stemmed

#finalized doc
def finalize_doc(doc):
	part1 = clean_doc(doc)
	part2 = stemmed_doc(part1)
	#part2 = stem_words(part1, stemmer)
	return part2

"""
#training set
sectors = 6
trainSet = defaultdict(list)
with open("train.json") as json_file:
    data = json.load(json_file)
    for i in data:
    	trainSet[i["sector"]].append(finalize_doc(extract_docs(i["id"], i["cik"])))

tokenize = lambda doc: doc.lower().split(" ")
"""

def jaccard_similarity(query, document):
	intersection = set(query).intersection(set(document))
	union = set(query).union(set(document))
	return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
	return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
	count = tokenized_document.count(term)
	if count == 0:
		return 0
	return 1 + math.log(count)

def inverse_document_frequencies(tokenized_documents):
	idf_values = {}
	all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
	for tkn in all_tokens_set:
		contains_token = map(lambda doc: tkn in doc, tokenized_documents)
		idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
	return idf_values

def tfidf(documents):
	tokenized_documents = [tokenize(d) for d in documents]
	idf = inverse_document_frequencies(tokenized_documents)
	tfidf_documents = []
	for document in tokenized_documents:
	  	doc_tfidf = []
	  	for term in idf.keys():
	  		tf = sublinear_term_frequency(term, document)
	  		doc_tfidf.append(tf * idf[term])
	  	tfidf_documents.append(doc_tfidf)
	return tfidf_documents

def cosine_similarity(vector1, vector2):
	dot_product = sum(p*q for p,q in zip(vector1, vector2))
	magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
	if not magnitude:
		return 0
	return dot_product/magnitude

"""
#development set
devSet = defaultdict(list)
with open("develop.json") as json_file:
    data = json.load(json_file)
    for i in data:
    	devSet[i["sector"]].append(finalize_doc(extract_docs(i["id"], i["cik"])))
development_docs = []
for i in trainSet.keys():
	development_docs.append(trainSet[i] + devSet[i])
docsPerSector = len(development_docs[0])

similarities = {"technology":[],"biopharmaceuticals":[],"finance":[],"energy":[],"consumer_discretionary":[], "manufacturing":[]}
sim_list = list(similarities.keys())
"""

#6 sectors total
#8 docs in each sector 
def generate_cosine_similarities(doc,development_docs,similarity):
    #test, change all run_docs back to development docs
    run_docs = copy.deepcopy(development_docs)
    simCopy = copy.deepcopy(similarity)
    finalized_doc = finalize_doc(doc)
	#add test doc in for testing
    for i in range(len(run_docs)):
        run_docs[i].append(finalized_doc)
	#tfidf vectorization
    for i in range(len(run_docs)):
        temp_tfidf = tfidf(run_docs[i])
        average_cosim = 0
        for j in range(docsPerSector):
            hold = cosine_similarity(temp_tfidf[j],temp_tfidf[docsPerSector])
            average_cosim += hold
            simCopy[sim_list[i]].append(hold)
        simCopy[sim_list[i]].append(average_cosim/docsPerSector)
    return simCopy

def sector_ranking(sims):
	#tracking the sector with most similarity
	greatest = [0, 0]
	for i in sims.keys():
		if sims[i][docsPerSector] > greatest[1]:
			greatest[0] = i
			greatest[1] = sims[i][docsPerSector]
	#clear dictionary
	sims = sims.fromkeys(sims,[])
	return greatest

def gen_output(td,d,s):
	sim = generate_cosine_similarities(td,d,s)
	#display entire dictionary of similarities
	print(sim)
	return(sector_ranking(sim))


def main():
    #training set
    trainSet = defaultdict(list)
    with open("train.json") as json_file:
        data = json.load(json_file)
        for i in data:
        	trainSet[i["sector"]].append(finalize_doc(extract_docs(i["id"], i["cik"])))
    
    global tokenize  
    tokenize = lambda doc: doc.lower().split(" ")

    #development set
    devSet = defaultdict(list)
    with open("develop.json") as json_file:
        data = json.load(json_file)
        for i in data:
        	devSet[i["sector"]].append(finalize_doc(extract_docs(i["id"], i["cik"])))
    development_docs = []
    for i in trainSet.keys():
    	development_docs.append(trainSet[i] + devSet[i])
    global docsPerSector 
    docsPerSector= len(development_docs[0])
    
    global similarities 
    similarities = {"technology":[],"biopharmaceuticals":[],"finance":[],"energy":[],"consumer_discretionary":[], "manufacturing":[]}
    global sim_list 
    sim_list = list(similarities.keys())

    #test set
    with open("test.csv") as csvF:
        r = csv.reader(csvF, delimiter=',')
        for row in r:
            print(row[0])
            print(gen_output(extract_docs(row[0], row[1]), development_docs, similarities))

if __name__ == "__main__":
    main()
            

