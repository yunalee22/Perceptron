import sys
import json

""" Class to classify text using perceptron model. """
class PerceptronClassifier():

	def __init__(self, lines):
		# Test data
		self.lines = lines
		self.vocabulary = set()		# Known words
		self.X = {}								# Test examples
		self.Y1 = {}							# True/fake (1/-1) prediction
		self.Y2 = {}							# Pos/neg (1/-1) prediction

		# Model parameters
		self.w1 = {}							# Weights for true/fake
		self.b1 = 0								# Bias for true/fake
		self.w2 = {}							# Weights for pos/neg
		self.b2 = 0								# Bias for pos/neg

		# For token preprocessing
		self.punctuation = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
		self.stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', \
			'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', \
			'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', \
			"didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', \
			'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', \
			'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', \
			'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', \
			'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", \
			'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', \
			'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', \
			"she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', \
			"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", \
			'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', \
			'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", \
			"we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', \
			"where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", \
			'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', \
			'yourself', 'yourselves']

	# Read perceptron model parameters from file
	def readModelFromFile(self, filename):
		with open(filename, "r", encoding="utf-8") as file_object:
			model = json.load(file_object)
			self.vocabulary = model['vocabulary']
			self.w1 = model['w1']
			self.b1 = model['b1']
			self.w2 = model['w2']
			self.b2 = model['b2']

	# Tokenize and preprocess test data
	def parseTestData(self):
		for line in self.lines:
			tokens = line.split()
			identifier = tokens[0]
			sentence = tokens[3:]
			x = self.sentenceToFeatureDict(sentence)
			self.X[identifier] = x

	# Converts a sentence to a feature dictionary (word:count)
	def sentenceToFeatureDict(self, sentence):
		features = {}
		# Translator to remove punctuation
		translator = str.maketrans('', '', self.punctuation)
		# Remove punctuation, convert to lowercase, exclude stop words
		for word in sentence:
			word = word.translate(translator).lower().strip()
			if word is "" or word.isdigit() or word in self.stopwords:
				continue
			if word in features:
				features[word] += 1
			else:
				features[word] = 1
		return features

	# Classify given test examples
	def classify(self):
		# Classify each test example
		for identifier in self.X:
			x = self.X[identifier]
			# Compute activations
			a1 = 0
			a2 = 0
			for d in x:
				if d in self.vocabulary:
					a1 += self.w1[d] * x[d] + self.b1
					a2 += self.w2[d] * x[d] + self.b2
			# Sign of activation is classification
			self.Y1[identifier] = "Fake" if a1 < 0 else "True"
			self.Y2[identifier] = "Neg" if a2 < 0 else "Pos"

	# Writes the final classifications to file
	def writeResultsToFile(self):
		with open("percepoutput.txt", "w", encoding="utf-8") as output_file:
			for i in self.X:
				output_file.write(i + " " + self.Y1[i] + " " + self.Y2[i] + "\n")


def main():
	# Get test data from file
	modelfile = sys.argv[1]
	datafile = sys.argv[2]
	with open(datafile) as file_object:
		lines = file_object.readlines()

	# Test Naive Bayes model
	classifier = PerceptronClassifier(lines)
	classifier.readModelFromFile(modelfile)
	classifier.parseTestData()
	classifier.classify()
	classifier.writeResultsToFile()


if __name__=="__main__":
	main()


