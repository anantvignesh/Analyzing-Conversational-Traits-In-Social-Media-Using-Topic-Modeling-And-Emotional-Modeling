import pandas as pd
import spacy
import re
import unidecode
import inflect
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from IPython.display import clear_output

#Download Stop Words If Necessary
nltk.download('stopwords')
nltk.download('words')
nltk.download('vader_lexicon')

#Global Variables
CONTRACTION_MAP = {
"ain't": "is not","aren't": "are not","can't": "cannot",
"can't've": "cannot have","'cause": "because","could've": "could have",
"couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not",
"don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",
"haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
"he'll've": "he he will have","he's": "he is","how'd": "how did","how'd'y": "how do you",
"how'll": "how will","how's": "how is","I'd": "I would","I'd've": "I would have",
"I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have",
"i'd": "i would","i'd've": "i would have","i'll": "i will","i'll've": "i will have",
"i'm": "i am","i've": "i have","isn't": "is not","it'd": "it would",
"it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have",
"mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not",
"mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock",
"oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
"shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will",
"she'll've": "she will have","she's": "she is","should've": "should have","shouldn't": "should not",
"shouldn't've": "should not have","so've": "so have","so's": "so as","that'd": "that would",
"that'd've": "that would have","that's": "that is","there'd": "there would","there'd've": "there would have",
"there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will",
"they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have",
"wasn't": "was not","we'd": "we would","we'd've": "we would have","we'll": "we will",
"we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not",
"what'll": "what will","what'll've": "what will have","what're": "what are","what's": "what is",
"what've": "what have","when's": "when is","when've": "when have","where'd": "where did",
"where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have",
"who's": "who is","who've": "who have","why's": "why is","why've": "why have",
"will've": "will have","won't": "will not","won't've": "will not have","would've": "would have",
"wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would",
"y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
"you'd've": "you would have","you'll": "you will","you'll've": "you will have","you're": "you are","you've": "you have"
}

#Slang Map For Words With Slang
with open('/content/drive/My Drive/Code/slang.txt') as file:
        slang_map = dict(map(str.strip, line.partition('\t')[::2]) for line in file if line.strip())

#Inflection Engine For Number To Word Conversion
inflectEngine = inflect.engine()

#Custom Functions

#Remove "RT" tag
def removeRT(text):
    text = text.replace("RT", "", 1)
    return text
    
#Removing HTML Tags
def stripHtmlTags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

#Remove URL
def removeURL(text):
    text = re.sub(r"http\S+", "", text)
    return text

#Remove Non English Words
def removeNonEnglish(text):
    words = set(nltk.corpus.words.words())
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    return text

#Contraction Expansion
def expandContractions(text, contraction_mapping = CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#Removing Accented Characters
def removeAccentedChars(text):
    text = unidecode.unidecode(text)
    return text

#Remove User Mentions
def removeUserMentions(text):
    text = re.sub(r"(?:\@)\S+", "", text)
    return text

#Convert Text To Lower Case
def lowerCaseConversion(text):
    text = " ".join(x.lower() for x in str(text).split())
    return text

#Remove special characters
def specialCharsRemoval(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

#Remove Emoticons
def removeEmoticons(text):
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

#Remove Stop Words
def stopWordsRemoval(text):
    stop = stopwords.words('english')
    stop.extend(['come','order','try','go','get','make','drink','plate','dish','restaurant','place',
                 'would','really','like','great','service','came','got'])
    text = " ".join([x for x in text.split() if x not in stop])
    return text

#Convert Numbers To Words
def numberToWords(text):  
    temp_str = text.split() 
    new_string = [] 
  
    for word in temp_str: 
        if word.isdigit(): 
            temp = inflectEngine.number_to_words(word) 
            new_string.append(temp)
        else: 
            new_string.append(word) 
  
    text = ' '.join(new_string) 
    return text

#Remove Numbers From Text
def removeNumbers(text):
    for word in text.split():
        if word.isnumeric():
            text = text.replace(word, '')
    return text

#Remove Adverbs
def removeAdverbs(text):
    adverbs = ['accidentally', 'always', 'angrily', 'anxiously', 'awkwardly', 'badly', 'blindly', 'boastfully', 'boldly', 'bravely', 'brightly', 'cheerfully', 'coyly', 'crazily', 'defiantly', 'deftly', 'deliberately', 'devotedly', 'doubtfully', 'dramatically', 'dutifully', 'eagerly', 'elegantly', 'enormously', 'evenly', 'eventually', 'exactly', 'faithfully', 'finally', 'foolishly', 'fortunately', 'frequently', 'gleefully', 'gracefully', 'happily', 'hastily', 'honestly', 'hopelessly', 'hourly', 'hungrily', 'innocently', 'inquisitively', 'irritably', 'jealously', 'justly', 'kindly', 'lazily', 'loosely', 'madly', 'merrily', 'mortally', 'mysteriously', 'nervously', 'never', 'obediently', 'obnoxiously', 'occasionally', 'often', 'only', 'perfectly', 'politely', 'poorly', 'powerfully', 'promptly', 'quickly', 'rapidly', 'rarely', 'regularly', 'rudely', 'safely', 'seldom', 'selfishly', 'seriously', 'shakily', 'sharply', 'silently', 'slowly', 'solemnly', 'sometimes', 'speedily', 'sternly', 'technically', 'tediously', 'unexpectedly', 'usually', 'victoriously', 'vivaciously', 'warmly', 'wearily', 'weekly', 'wildly', 'yearly', 'not', 'also', 'very', 'often', 'however', 'too', 'usually', 'really', 'early', 'never', 'always', 'sometimes', 'together', 'likely', 'simply', 'generally', 'instead', 'actually', 'again', 'rather', 'almost', 'especially', 'ever', 'quickly', 'probably', 'already', 'below', 'directly', 'therefore', 'else', 'thus', 'easily', 'eventually', 'exactly', 'certainly', 'normally', 'currently', 'extremely', 'finally', 'constantly', 'properly', 'soon', 'specifically', 'ahead', 'daily', 'highly', 'immediately', 'relatively', 'slowly', 'fairly', 'primarily', 'completely', 'ultimately', 'widely', 'recently', 'seriously', 'frequently', 'fully', 'mostly', 'naturally', 'nearly', 'occasionally', 'carefully', 'clearly', 'essentially', 'possibly', 'slightly', 'somewhat', 'equally', 'greatly', 'necessarily', 'personally', 'rarely', 'regularly', 'similarly', 'basically', 'closely', 'effectively', 'initially', 'literally', 'mainly', 'merely', 'gently', 'hopefully', 'originally', 'roughly', 'significantly', 'totally', 'twice', 'elsewhere', 'everywhere', 'obviously', 'perfectly', 'physically', 'successfully', 'suddenly', 'truly', 'virtually', 'altogether', 'anyway', 'automatically', 'deeply', 'definitely', 'deliberately', 'hardly', 'readily', 'terribly', 'unfortunately', 'forth', 'briefly', 'moreover', 'strongly', 'honestly', 'previously', 'as', 'there', 'when', 'how', 'so', 'up', 'out', 'no', 'only', 'well', 'then', 'first', 'where', 'why', 'now', 'around', 'once', 'down', 'off', 'here', 'tonight', 'away', 'today', 'far', 'quite', 'later', 'above', 'yet', 'maybe', 'otherwise', 'near', 'forward', 'somewhere', 'anywhere', 'please', 'forever', 'somehow', 'absolutely', 'abroad', 'yeah', 'nowhere', 'tomorrow', 'yesterday', 'the', 'to', 'in', 'on', 'by', 'more', 'about', 'such', 'through', 'new', 'just', 'any', 'each', 'much', 'before', 'between', 'free', 'right', 'best', 'since', 'both', 'sure', 'without', 'back', 'better', 'enough', 'lot', 'small', 'though', 'less', 'little', 'under', 'next', 'hard', 'real', 'left', 'least', 'short', 'last', 'within', 'along', 'lower', 'TRUE', 'bad', 'across', 'clear', 'easy', 'full', 'close', 'late', 'proper', 'fast', 'wide', 'item', 'wrong', 'ago', 'behind', 'quick', 'straight', 'direct', 'extra', 'morning', 'pretty', 'overall', 'alone', 'bright', 'flat', 'whatever', 'slow', 'clean', 'fresh', 'whenever', 'cheap', 'thin', 'cool', 'fair', 'fine', 'smooth', 'FALSE', 'thick', 'collect', 'nearby', 'wild', 'apart', 'none', 'strange', 'tourist', 'aside', 'loud', 'super', 'tight', 'gross', 'ill', 'downtown', 'honest', 'ok', 'pray', 'weekly']
    text = " ".join([x for x in text.split() if x not in stop])
    return text

#Remove Slang Words
def removeSlangWords(text):
    words = text.split()
    for word in words:
        if word in slang_map.keys():
            text = text.replace(word, slang_map[word])
    return text

#Remove Whitespace From Text 
def removeWhitespace(text): 
    return  " ".join(text.split())

#Main Data Preprocessing Method
def dataPreprocessing(corpus, RTremoval = True, userMentionRemoval = True, toLowerCase = True, stripHTML = True, URLRemoval = True, expandContraction = True, 
                      stripAccentedChars = True, NonEnglishRemoval = True, removeSplChars = True, removeStopWords = True, adverbsRemoval = True, 
                      numbersRemoval = True, numToWords = True, slangWordsRemoval = True, whiteSpaceRemoval = True):
    for i in range(len(corpus)):
        clear_output(wait=True)
        print(("%d of %d tweets processed...")%(i+1, len(corpus)))
        if RTremoval:
            corpus[i] = removeRT(corpus[i])
        if userMentionRemoval:
            corpus[i] = removeUserMentions(corpus[i])
        if toLowerCase:
            corpus[i] = lowerCaseConversion(corpus[i])
        if stripHTML:
            corpus[i] = stripHtmlTags(corpus[i])
        if URLRemoval:
            corpus[i] = removeURL(corpus[i])
        if expandContraction:
            corpus[i] = expandContractions(corpus[i])
        if stripAccentedChars:
            corpus[i] = removeAccentedChars(corpus[i])
        if NonEnglishRemoval:
            corpus[i] = removeNonEnglish(corpus[i])
        if removeSplChars:
            corpus[i] = specialCharsRemoval(corpus[i])
        if removeStopWords:
            corpus[i] = stopWordsRemoval(corpus[i])
        if adverbsRemoval:
            corpus[i] = removeAdverbs(corpus[i])
        if numbersRemoval:
            corpus[i] = removeNumbers(corpus[i])
        if numToWords:
            corpus[i] = numberToWords(corpus[i])
        if slangWordsRemoval:
            corpus[i] = removeSlangWords(corpus[i])
        if whiteSpaceRemoval:
            corpus[i] = removeWhitespace(corpus[i])
    return(corpus)