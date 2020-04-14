# NLPComparableAnalysis
File cleaning and extraction of SEC 10-k filings followed by TF-IDF and Cosine Similarity to test a company's industry similarity
Credit for Edgar SEC 10-k extraction code, edgar.py: https://github.com/joeyism/py-edgar 

**Overview of Main.py**

I started by extracting documents from SEC's edgar database. It should be noted that noOfDocuments is set to 2 because of all company filings in SEC's database, some files are not 10-k filings. While is is not gauranteed that the first filing extracted is the 10-k filings, it can be gauranteed that 10-k files can be found within the first 2 documents.

Next I cleaned the documents. I only took half of each 10-K filing since company business operations, financial data, and risk measures are outlined within the first few sections of the document. The latter half of document usually only contains information pertinent to executive compensation and corporation structure. By eliminating the unnecessary parts of documents, I found more accurate results and faster processing time for tf-idf vectorization and cosine similarity.

In terms of cleaning the documents, I made sure to 1. only take alphabeticals characters 2. make all words lower case 3. eliminate stop words 4. use nltk Word_Tokensize to eliminate contractions and white spaces 5. use nltk PorterStemmer to take the stem form of all words. 

I used 6 sectors in my analysis: Information Technology, Biopharmaceuticals, Financial Services, Energy, Consumer Discretionary, and Manufacturing.

I then proceeded to perform Term Frequency-Inverse Document Frequency (TFIDF) vectorization on each document within a sector. There are various TF methods that can be used. I decided to use sublinear term frequency since it reduces the weight of words that appear to often, something that might occur in documents like SEC 10-K filings where financial terms can be used in abundance. Once TFIDF was complete, I performed cosine similarity on input documents to compare how similar such a document is to a sector as a whole.

The complete output provides results for input company similarity to each company within the development set. The last number for each sector is the input company similarity to the sector as a whole. 
