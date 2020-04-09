from __future__ import division
import string
import math
import edgar
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#Obtaining SEC docs
def extract_docs(comp_name,cik):
	company = edgar.Company(comp_name, cik)
	tree = company.getAllFilings(filingType = "10-K")
	docs = edgar.getDocuments(tree, noOfDocuments=1)
	return docs

print('a')

#Development corpus


#working on creating a sector class that appends all the docs and perhaps finalizes them for me
""""
class Sector():
	def __init__(self,docs):
		self.docs = docs

	def generate_sector(docs):
		sec = []
		for i in range(docs):
			comp = input("enter a company")
			cik = input("enter the companies' cik")
			sec.append(extract_docs)
"""





#IT
itdoc1 = (extract_docs("APPLE INC", "0000320193"))
itdoc2 = (extract_docs("MICROSOFT CORPORATION", "0000789019"))
itdoc3 = (extract_docs("INTERNATIONAL BUSINESS MACHINES CORP", "0000051143"))
itdoc4 = (extract_docs("Oracle Corp", "0001341439"))

#Biopharmaceuticals
biodoc1 = (extract_docs("PFIZER INC", "0000078003"))
biodoc2 = (extract_docs("JOHNSON & JOHNSON", "0000200406"))
biodoc3 = (extract_docs("Biogen Inc.", "0000875045"))
biodoc4 = (extract_docs("MERCK & CO., INC.", "0000310158"))

#Finance
findoc1 = (extract_docs("CITIGROUP INC", "0000831001"))
findoc2 = (extract_docs("GOLDMAN SACHS GROUP INC", "0000886982"))
#findoc3 = (extract_docs("American Express Co", "0000004962"))
findoc3 = (extract_docs("WELLS FARGO & COMPANY", "0000072971"))
findoc4 = (extract_docs("BlackRock Inc.", "0001364742"))

#Energy
edoc1 = (extract_docs("EXXON MOBIL CORP", "0000034088"))
edoc2 = (extract_docs("CHEVRON CORP", "0000093410"))
edoc3 = (extract_docs("ENTERPRISE PRODUCTS PARTNERS L P", "0001061219"))
edoc4 = (extract_docs("Phillips 66", "0001534701"))

#Consumer Discretionary
cddoc1 = (extract_docs("General Motors Co", "0001467858"))
cddoc2 = (extract_docs("Target Corp", "0000027419"))
cddoc3 = (extract_docs("Marriott International Inc", "0001048286"))
cddoc4 = (extract_docs("CHIPOTLE MEXICAN GRILL", "0001058090"))


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



#building development set
it_documents = [finalize_doc(itdoc1),finalize_doc(itdoc2),finalize_doc(itdoc3),finalize_doc(itdoc4)]
bio_documents = [finalize_doc(biodoc1),finalize_doc(biodoc2),finalize_doc(biodoc3),finalize_doc(biodoc4)]
finance_documents = [finalize_doc(findoc1),finalize_doc(findoc2),finalize_doc(findoc3),finalize_doc(findoc4)]
energy_documents = [finalize_doc(edoc1),finalize_doc(edoc2),finalize_doc(edoc3),finalize_doc(edoc4)]
consumerdisc_documents= [finalize_doc(cddoc1),finalize_doc(cddoc2),finalize_doc(cddoc3),finalize_doc(cddoc4)]

development_docs = [it_documents,bio_documents,finance_documents,energy_documents,consumerdisc_documents]


tokenize = lambda doc: doc.lower().split(" ")

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

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

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



#Asking user for a company that they want to check
#user_company = input("Please input the name of a company you would like to check as it would appear on sec.gov ")
#user_cik = input("Please input the corresponding CIK# of that company ")

#Training Corpus
"""
train_doc = extract_docs("Oracle Corp", "0001341439")
train_sector = ""
train_doc2 = extract_docs("MERCK & CO., INC.", "0000310158")
train_doc3 = extract_docs("FACEBOOK INC", "0001326801")
train_doc4 = extract_docs("WELLS FARGO & COMPANY", "0000072971")
train_doc5 = extract_docs("Phillips 66", "0001534701")
train_doc6 = extract_docs("CHIPOTLE MEXICAN GRILL", "0001058090")
train_doc7 = extract_docs("BlackRock Inc.", "0001364742")
"""
#((user_company,user_cik)

#Test Corpus
test_doc1 = extract_docs("FACEBOOK INC","0001326801")
test_doc2 = extract_docs("LILLY ELI & CO", "0000059478")
test_doc3 = extract_docs("SVB FINANCIAL GROUP", "0000719739")
test_doc4 = extract_docs("VALERO ENERGY CORP/TX", "0001035002")
test_doc5 = extract_docs("NIKE INC", "0000320187")
test_doc6 = extract_docs("AMAZON COM INC", "0001018724")


# 0 = it, 1 = bio, 2 = finance, 3 = energy, 4 = consumer discretionary

similarities = {"technology":[],"biopharmaceuticals":[],"finance":[],"energy":[],"consumer_discretionary":[]}


#5 sectors total
#3 docs in each sector for testing plus 1 test doc
def generate_cosine_similarities(doc,development_docs,similarity):
  #test, change all run_docs back to development docs
	run_docs = development_docs.copy()
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
			similarity[sim_list[i]].append(hold)
		similarity[sim_list[i]].append(average_cosim/docsPerSector)

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
	generate_cosine_similarities(td,d,s)
	#display entire dictionary of similarities
	print(s)
	return(sector_ranking(s))


#facebook  = gen_output(test_doc1,development_docs,similarities)
#print(facebook)


#eli_lilly  = gen_output(test_doc2,development_docs,similarities)
#print(eli_lilly)


#svb = gen_output(test_doc3,development_docs,similarities)
#print(svb)


#valero = gen_output(test_doc4,development_docs,similarities)
#print(valero)

#nike = gen_output(test_doc5,development_docs,similarities)
#print(nike)

#amazon = gen_output(test_doc6, development_docs, similarities)
#print(amazon)








