import math
import edgar
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import copy

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


#training set
it_documents = [finalize_doc(extract_docs("APPLE INC", "0000320193")),finalize_doc(extract_docs("MICROSOFT CORPORATION", "0000789019")),finalize_doc(extract_docs("INTERNATIONAL BUSINESS MACHINES CORP", "0000051143")),finalize_doc(extract_docs("Oracle Corp", "0001341439")),finalize_doc(extract_docs("INTERNATIONAL BUSINESS MACHINES CORP", "0000051143"))]
bio_documents = [finalize_doc(extract_docs("PFIZER INC", "0000078003")),finalize_doc(extract_docs("JOHNSON & JOHNSON", "0000200406")),finalize_doc(extract_docs("Biogen Inc.", "0000875045")),finalize_doc(extract_docs("MERCK & CO., INC.", "0000310158")),finalize_doc(extract_docs("BAXTER INTERNATIONAL INC", "0000010456"))]
finance_documents = [finalize_doc(extract_docs("CITIGROUP INC", "0000831001")),finalize_doc(extract_docs("GOLDMAN SACHS GROUP INC", "0000886982")),finalize_doc(extract_docs("WELLS FARGO & COMPANY", "0000072971")),finalize_doc(extract_docs("BlackRock Inc.", "0001364742")),finalize_doc(extract_docs("AMERICAN INTERNATIONAL GROUP Inc", "0000005272"))]
energy_documents = [finalize_doc(extract_docs("EXXON MOBIL CORP", "0000034088")),finalize_doc(extract_docs("CHEVRON CORP", "0000093410")),finalize_doc(extract_docs("ENTERPRISE PRODUCTS PARTNERS L P", "0001061219")),finalize_doc(extract_docs("Phillips 66", "0001534701")),finalize_doc(extract_docs("DEVON ENERGY CORP/DE", "0001090012"))]
consumerdisc_documents = [finalize_doc(extract_docs("STARBUCKS CORP", "0000829224")),finalize_doc(extract_docs("Target Corp", "0000027419")),finalize_doc(extract_docs("Marriott International Inc", "0001048286")),finalize_doc(extract_docs("CHIPOTLE MEXICAN GRILL", "0001058090")),finalize_doc(extract_docs("NORDSTROM INC", "0000072333"))]
manufacturing_documents = [finalize_doc(extract_docs("General Motors Co", "0001467858")),finalize_doc(extract_docs("LOCKHEED MARTIN CORP", "0000936468")),finalize_doc(extract_docs("CATERPILLAR INC", "0000018230")),finalize_doc(extract_docs("FORD MOTOR CO", "0000037996")),finalize_doc(extract_docs("RELIANCE STEEL & ALUMINUM CO", "0000861884"))]


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


#development set
it_Dev = [finalize_doc(extract_docs("FACEBOOK INC","0001326801")), finalize_doc(extract_docs("INTEL CORP", "0000050863")), finalize_doc(extract_docs("CISCO SYSTEMS", "0000858877"))]
bio_Dev = [finalize_doc(extract_docs("LILLY ELI & CO", "0000059478")), finalize_doc(extract_docs("AbbVie Inc.", "0001551152")), finalize_doc(extract_docs("THERMO FISHER SCIENTIFIC INC.", "0000097745"))]
finance_Dev = [finalize_doc(extract_docs("SVB FINANCIAL GROUP", "0000719739")),  finalize_doc(extract_docs("JPMORGAN CHASE & CO", "0000019617")), finalize_doc(extract_docs("BANK OF AMERICA CORP /DE/", "0000070858"))]
energy_Dev = [finalize_doc(extract_docs("VALERO ENERGY CORP/TX", "0001035002")), finalize_doc(extract_docs("EOG RESOURCES INC", "0000821189")), finalize_doc(extract_docs("OCCIDENTAL PETROLEUM CORP /DE/", "0000797468"))]
consumerdisc_Dev = [finalize_doc(extract_docs("NIKE INC", "0000320187")), finalize_doc(extract_docs("TJX COMPANIES INC /DE/", "0000109198")), finalize_doc(extract_docs("MCDONALDS CORP", "0000063908"))]      
manufacturing_Dev = [finalize_doc(extract_docs("BOEING CO", "0000012927")), finalize_doc(extract_docs("GOODYEAR TIRE & RUBBER CO /OH/", "0000042582")) ,finalize_doc(extract_docs("LEAR CORP", "0000842162"))]

development_docs = [it_documents + it_Dev, bio_documents + bio_Dev, finance_documents + finance_Dev, energy_documents + energy_Dev, consumerdisc_documents + consumerdisc_Dev, manufacturing_documents + manufacturing_Dev]
docsPerSector = len(development_docs[0])

similarities = {"technology":[],"biopharmaceuticals":[],"finance":[],"energy":[],"consumer_discretionary":[], "manufacturing":[]}
sim_list = list(similarities.keys())


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

#test set
"""
print(gen_output(extract_docs("Booking Holdings Inc.", "0001075531"), development_docs, similarities))
print(gen_output(extract_docs("AMAZON COM INC", "0001018724"), development_docs, similarities))
print(gen_output(extract_docs("ADOBE INC.", "0000796343"), development_docs, similarities))
print(gen_output(extract_docs("Tesla, Inc.", "0001318605"), development_docs, similarities))
print(gen_output(extract_docs("Q2 Holdings, Inc.", "0001410384"), development_docs, similarities))
print(gen_output(extract_docs("MCKESSON CORP", "0000927653"), development_docs, similarities))
print(gen_output(extract_docs("Cellular Biomedicine Group, Inc.", "0001378624"), development_docs, similarities))
print(gen_output(extract_docs("AMERICAN EXPRESS CO", "0000004962"), development_docs, similarities))
print(gen_output(extract_docs("MORGAN STANLEY", "0000895421"), development_docs, similarities))
print(gen_output(extract_docs("HESS CORP", "0000004447"), development_docs, similarities))
print(gen_output(extract_docs("SOUTHERN CO", "0000092122"), development_docs, similarities))
print(gen_output(extract_docs("NORTHROP GRUMMAN CORP /DE/", "0001133421"), development_docs, similarities))
print(gen_output(extract_docs("Workday, Inc.", "0001327811"), development_docs, similarities))
print(gen_output(extract_docs("PayPal Holdings, Inc.", "0001633917"), development_docs, similarities))
print(gen_output(extract_docs("Mastercard Inc", "0001141391"), development_docs, similarities))
print(gen_output(extract_docs("BLACKLINE, INC.", "0001666134"), development_docs, similarities))
print(gen_output(extract_docs("HTIFFANY & CO", "0000098246"), development_docs, similarities))
print(gen_output(extract_docs("Macy's, Inc.", "0000794367"), development_docs, similarities))
print(gen_output(extract_docs("Blackstone Group Inc", "0001393818"), development_docs, similarities))
print(gen_output(extract_docs("Catalent, Inc.", "0001596783"), development_docs, similarities))
"""

