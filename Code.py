
# coding: utf-8

# In[2]:


from IPython.core.display import HTML

rawHTML = "<h3>Import Required Dependencies</h3>"

HTML(rawHTML)


# In[280]:


dependencies = ["nltk", "numpy", "pandas", "scipy", "sklearn", "re"]

for module in dependencies:
    
    module_obj = __import__(module)
    globals()[module]  = module_obj


# In[4]:


rawHTML = "<h3>Reading CSV File as DataFrame</h3>"

HTML(rawHTML)


# In[281]:


import pandas as pd
dataframe = pd.read_csv("train_test.csv")
# dataframe


# In[5]:


rawHTML = """<h1>Cleaning Data Part</h1>
            <p>Remove the punctuation, any urls and numbers. Finally, convert every word to lower case.<p>"""
    
HTML(rawHTML)


# In[282]:


from string import punctuation

def cleanData(email):
    
    email = re.sub(r'http\S+', ' ', email)
    email = re.sub("\d+", " ", email)
    email = email.replace('\n', ' ')
    email = email.translate(str.maketrans("", "", punctuation))
    email = email.lower()
    return email

dataframe['text'] = dataframe['text'].apply(cleanData)

# dataframe


# In[29]:


rawHTML = """<h2>Prepare the Data</h2>
            <p>Split the text string into individual words and stem each word. Remove english stop words.<p>
            <h2>Split and Stem</h2>
            <p>Split the text by white spaces and link the different forms of the same word to each other, using stemming. For example “responsiveness” and “response” have the same stem/root - “respons”.<p>
            <h2>Remove Stop Words</h2>
            <p>Some words such as “the” or “is” appear in all emails and don’t have much content to them. These words are not going to help the algorithm distinguish spam from ham. Such words are called stopwords and they can be disregarded during classification.<p>"""
            

HTML(rawHTML)


# In[283]:


from nltk.stem.snowball import SnowballStemmer
# nltk.download("wordnet")
from nltk.corpus import wordnet as wn

def preprocessText(email):
    
    words = ""
    stemmer = SnowballStemmer("english")
    email = email.split()
    
    for word in email:
        
        words = words + stemmer.stem(word) + ' '
        
    return words

dataframe['text'] = dataframe['text'].apply(preprocessText)

# dataframe


# In[34]:


rawHTML = """<h2>Vectorize Words and Split Data to Train/Test Sets</h2>
            <p>Transform the words into a tf-idf matrix using the sklearn TfIdf transformation. Then, create train/test sets with the train_test_split function, using stratify parameter. The dataset is highly unbalanced and the stratify parameter will make a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify. For example, if variable y is 0 and 1 and there are 30% of 0’s and 70% of 1’s, stratify=y will make sure that the random split has 30% of 0’s and 75% of 1’s.<p>"""

HTML(rawHTML)


# In[284]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

Xs = dataframe['text'].values
Ys = dataframe['type'].values

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = "english")



Xs = vectorizer.fit_transform(Xs)

X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size = 0.2, shuffle = True,
                                                   random_state = 0,
                                                   stratify = Ys)


featureNames = vectorizer.get_feature_names()


# In[9]:


rawHTML = """<h2>Train Classifier With Scikit Learn Library</h2>
            <p>Train a Naive Bayes classifier and evaluate the performance with the accuracy score.<p>"""

HTML(rawHTML)


# In[285]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Accuracy: {}".format(clf.score(X_test, y_test)))


# In[251]:


rawHTML = """<h3>My Features</h3>
             <p>1- Length of emails<p>
             <p>2- All words in emails<p>"""

HTML(rawHTML)


# In[299]:


import matplotlib.pyplot as plt


xVal_1 = []
xVal_2 = []

for index, email in enumerate(X_trainI):
    emailLen = sum(len(word) for word in email)
    if emailLen > meanChrNo and y_train[index] == "spam":
        xVal_1.append(emailLen)
    if emailLen > meanChrNo and y_train[index] == "ham":
        xVal_2.append(emailLen)

maxLen = max(max(xVal_1), max(xVal_2))

fig, ax_0 = plt.subplots()
ax_0.hist(xVal_1, color="green")
ax_0.set_ylabel("Frequency of Spams")
ax_0.set_xlabel("Email Length")
ax_0.set_xlim((meanChrNo, maxLen))

fig, ax_1 = plt.subplots()
ax_1.hist(xVal_2, color="red")
ax_1.set_ylabel("Frequency of Hams")
ax_1.set_xlabel("Email Length")
ax_1.set_xlim((meanChrNo, maxLen))

plt.show()


# In[298]:


import matplotlib.pyplot as plt


xVal_1 = []
xVal_2 = []

for index, email in enumerate(X_trainI):
    emailLen = sum(len(word) for word in email)
    if emailLen < meanChrNo and y_train[index] == "spam":
        xVal_1.append(emailLen)
    if emailLen < meanChrNo and y_train[index] == "ham":
        xVal_2.append(emailLen)

maxLen = max(max(xVal_1), max(xVal_2))

fig, ax_0 = plt.subplots()
ax_0.hist(xVal_1, color="green")
ax_0.set_ylabel("Frequency of Spams")
ax_0.set_xlabel("Email Length")
ax_0.set_xlim((0, maxLen))

fig, ax_1 = plt.subplots()
ax_1.hist(xVal_2, color="red")
ax_1.set_ylabel("Frequency of Hams")
ax_1.set_xlabel("Email Length")
ax_1.set_xlim((0, maxLen))

plt.show()


# In[300]:


rawHTML = """<p>These histograms show that emailLength will be good feature.<p>"""
HTML(rawHTML)


# In[105]:


rawHTML = """<h3>Get Most Determinative </h3>
            <p>calling get_feature_names function<p>"""

HTML(rawHTML)


# In[286]:


mostDeterminativeFeatures = vectorizer.get_feature_names()


# In[200]:


# mostDeterminativeFeatures = []
# coefs = []

# def get_most_important_features(vectorizer, classifier, n=None):
    
#     featureNames = vectorizer.get_feature_names()
#     topFeatures = sorted(zip(classifier.coef_[0], featureNames))[-n:]
#     for coef, feat in topFeatures:
#         coefs.append(coef)
#         mostDeterminativeFeatures.append(feat)
        
# get_most_important_features(vectorizer, clf, 5)
# coefsMax = max([abs(coef) for coef in coefs])        
# mostDeterminativeFeatures = [[feature, abs(coefs[i]) / coefsMax] for i, feature in enumerate(mostDeterminativeFeatures)]
# # mostDeterminativeFeatures
# # coefs


# In[100]:


get_ipython().run_cell_magic('latex', '', '\n$$  Probability\\:of\\:Being\\:Ham\\:after\\:Watching\\:Features: $$\n$$ $$\n$$ \\Large P(ham\\;|\\;email) = \\frac{ P(ham) \\; \\Pi_{i = 0}^n P(feature[i]\\; | \\; ham)}{\\Pi_{i = 0}^n P(feature[i])} $$\n\n$$ $$\n$$ $$\n$$  Probability\\:of\\:Being\\:Spam\\:after\\:Watching\\:Features: $$\n$$ $$\n$$ \\Large P(Spam\\;|\\;email) = \\frac{ P(Spam) \\; \\Pi_{i = 0}^n P(feature[i]\\; | \\; Spam)}{\\Pi_{i = 0}^n P(feature[i])} $$')


# In[110]:


rawHTML = """<h3>Doing Naive Bayse Algorithm</h3>
            <p>geting ham and spam probability with above equations<p>"""

HTML(rawHTML)


# In[287]:


X_trainI = vectorizer.inverse_transform(X_train)
X_testI = vectorizer.inverse_transform(X_test)


# In[288]:


allChrNo = sum(len(word) for email in X_trainI for word in email)
meanChrNo = allChrNo / len(X_trainI)

spamBigE = sum(1 for i, email in enumerate(X_trainI) if sum(len(word) for word in email) > meanChrNo
               and y_train[i] == "spam")
spamBigEProb = spamBigE / sum(1 for t in y_train if t == "spam")

hamBigE = sum(1 for i, email in enumerate(X_trainI) if sum(len(word) for word in email) > meanChrNo 
              and y_train[i] == "ham")
hamBigEProb = hamBigE / sum(1 for t in y_train if t == "ham")

bigEProb = sum(1 for i, email in enumerate(X_trainI) if sum(len(word) for word in email) > meanChrNo) / len(X_trainI)

coefSpamProb = spamBigEProb / bigEProb
coefHamProb = hamBigEProb / bigEProb


# In[289]:


def getFeaturesProbs(_mostDeterminativeFeatures, X_train, y_train):
    
    yTrianSize = len(y_train)
    xTrainSize = yTrianSize

    hamNo = sum(1 for ham_spam_type in y_train if ham_spam_type == "ham")
    spamNo = yTrianSize - hamNo
    

    hamProb = hamNo / yTrianSize
    spamProb = 1.0 - hamProb
    
    featuresProb = []

    for feature in _mostDeterminativeFeatures:
        hamNoWithCurrFeature = 0
        spamNoWithCurrFeature = 0
        evidenceSeenNo = 0
    
        for index, email in enumerate(X_train):
            hamNoWithCurrFeature += feature in email and y_train[index] == 'ham'
            spamNoWithCurrFeature += feature in email and y_train[index] == 'spam'
            evidenceSeenNo += feature in email
        
        if evidenceSeenNo:
            featuresProb.append([feature,
                                hamNoWithCurrFeature,
                                spamNoWithCurrFeature,
                                hamNo,
                                spamNo,
                                xTrainSize,
                                evidenceSeenNo])
    
       
    return featuresProb


featuresProbs = getFeaturesProbs(mostDeterminativeFeatures, X_trainI, y_train)


# In[301]:


def determineSpamOrHam(email, myFeaturesProbs, hamProb, spamProb, _meanChrNo, _coefSpamProb, _coefHamProb):
    
    currEmailHamProb = hamProb
    currEmailSpamProb = spamProb
    
    if sum(len(word) for word in email) > _meanChrNo:
        currEmailHamProb = currEmailHamProb * _coefHamProb
        currEmailSpamProb = currEmailSpamProb * _coefSpamProb
    
    for featureDetails in myFeaturesProbs:
        if featureDetails[0] in email:
            currEmailHamProb = currEmailHamProb * ((featureDetails[1] * featureDetails[5]) / (featureDetails[3] * featureDetails[6]))
            currEmailSpamProb = currEmailSpamProb * ((featureDetails[2] * featureDetails[5]) / (featureDetails[4] * featureDetails[6]))
            

    if currEmailHamProb > currEmailSpamProb:
        return "ham"
    
    return "spam"


def testIt(X_train, y_train, X_test, y_test, myFeaturesProbs):
    
    yTrianSize = len(y_train)
    
    hamNo = sum(1 for ham_spam_type in y_train if ham_spam_type == "ham")
    spamNo = yTrianSize - hamNo
    
    hamProb = hamNo / yTrianSize
    spamProb = 1.0 - hamProb
    
    correctedDetectedSpams = sum(1 for index, email in enumerate(X_test) 
                                if determineSpamOrHam(email, myFeaturesProbs, hamProb, spamProb, meanChrNo, coefSpamProb, coefHamProb) == "spam"
                                and y_test[index] == "spam")
    
    spams = sum(1 for _type in y_test
                        if _type == "spam")
    recall = correctedDetectedSpams / spams
    
    detectedSpams = sum(1 for email in X_test 
                        if determineSpamOrHam(email, myFeaturesProbs, hamProb, spamProb, meanChrNo, coefSpamProb, coefHamProb) == "spam")
    precision = correctedDetectedSpams / detectedSpams
    
    correctDetected = sum(1 for index, email in enumerate(X_test) if determineSpamOrHam(email, myFeaturesProbs, hamProb, spamProb, meanChrNo, coefSpamProb, coefHamProb) == y_test[index])
    total = len(y_test)
    accuracy = correctDetected / total
    
    
    return [recall, precision, accuracy]
        
    

rsl = testIt(X_trainI, y_train, X_testI, y_test, featuresProbs)
print("Recall of Test Data: {}".format(rsl[0]))
print("Precision of Test Data : {}".format(rsl[2]))
print("Accuracy of Test Data: {}".format(rsl[2]))
# print("****************")
# rsl_1 = testIt(X_trainI, y_train, X_trainI, y_train, featuresProbs)
# print("Recall of Train Data: {}".format(rsl_1[0]))
# print("Precision of Train Data : {}".format(rsl_1[2]))
# print("Accuracy of Train Data: {}".format(rsl_1[2]))


# In[250]:


rawHTML = """<h3>How can we prevent from Overfiting</h3>
            <p>we should divide our data to train part and test part.If overfiting occurrs, we will getting low rate for accuracy of test part.
            For example If accuracy of train data is 99% and accuracy of test data is 20%, Overfiting had happend.
            By dividing data to test and train we can see whether overfiting occurs or not.
            In my program overfiting didn't occur<p>"""
HTML(rawHTML)


# In[112]:


rawHTML = """<h4>Reading Evaluation CSV File</h4>"""

HTML(rawHTML)


# In[245]:


evalDataFrame = pd.read_csv("/home/ehsana1998/PycharmProjects/AI_CA3/data/evaluate.csv")
evalDataFrame


# In[113]:


rawHTML = """<h3>Getting Result With Doing Steps</h3>
            <p>1- Cleaning data<p>
            <p>2- Processing text<p>
            <p>3- calling determineSpamOrHam function<p>"""

HTML(rawHTML)


# In[302]:



evalDataFrame['text'] = evalDataFrame['text'].apply(cleanData)
evalDataFrame['text'] = evalDataFrame['text'].apply(preprocessText)
# evalDataFrame

output = []
yTrianSize = len(y_train)
hamNo = sum(1 for ham_spam_type in y_train if ham_spam_type == "ham")
hamProb = hamNo / yTrianSize
spamProb = 1.0 - hamProb

for email in evalDataFrame['text']:
    
    output.append(determineSpamOrHam(email, featuresProbs, hamProb, spamProb, meanChrNo, coefSpamProb, coefHamProb))
    
evalDataFrame['type'] = output

outputData = []
for index, _type in enumerate(output):
    outputData.append([evalDataFrame['id'][index], _type])
    
outputDataFrame = pd.DataFrame(outputData, columns = ['id', 'type'])
outputDataFrame.to_csv("Output.csv", index = False)
outputDataFrame

