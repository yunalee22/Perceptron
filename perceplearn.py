import sys
import json

""" Class to represent a perceptron model. """
class PerceptronModel():

	def __init__(self, lines):
		# Training data
		self.lines = lines
		self.vocabulary = set()		# Known words
		self.identifiers = []			# Example identifiers
		self.X = {}								# Training examples
		self.Y1 = {}							# True/fake (1/-1) label
		self.Y2 = {}							# Pos/neg (1/-1) label

		# Model parameters for perceptron
		self.w1 = {}							# Weights for true/fake
		self.b1 = 0								# Bias for true/fake
		self.w2 = {}							# Weights for pos/neg
		self.b2 = 0								# Bias for pos/neg
		self.maxIterations = 30

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

	# Parse and preprocess training data
	def parseTrainingData(self):
		for line in self.lines:
			tokens = line.split()
			identifier = tokens[0]
			self.identifiers.append(identifier)
			y1 = 1 if tokens[1] == "True" else -1
			y2 = 1 if tokens[2] == "Pos" else -1
			sentence = tokens[3:]
			x = self.sentenceToFeatureDict(sentence)
			self.X[identifier] = x
			self.Y1[identifier] = y1
			self.Y2[identifier] = y2

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
			self.vocabulary.add(word)
		return features

	# Train vanilla perceptron and get weights
	def trainVanillaPerceptron(self):
		# Initialize weights and biases to 0
		for feature in self.vocabulary:
			self.w1[feature] = 0
			self.w2[feature] = 0
		self.b1 = 0
		self.b2 = 0
		# Learn perceptron weights
		for it in range(self.maxIterations):
			for identifier in self.identifiers:
				x = self.X[identifier]
				y1 = self.Y1[identifier]
				y2 = self.Y2[identifier]
				# Compute activations
				a1 = sum([self.w1[d] * x[d] + self.b1 for d in x])
				a2 = sum([self.w2[d] * x[d] + self.b2 for d in x])
				# Update weights and biases
				if y1 * a1 <= 0:
					for d in x:
						self.w1[d] += y1 * x[d]
					self.b1 += y1
				if y2 * a2 <= 0:
					for d in x:
						self.w2[d] += y2 * x[d]
					self.b2 += y2

	# Train averaged perceptron and get weights
	def trainAveragedPerceptron(self):
		# Initialize weights and biases to 0
		u1 = {}			# Cached weights 1
		u2 = {}			# Cached weights 2
		for feature in self.vocabulary:
			self.w1[feature] = 0
			self.w2[feature] = 0
			u1[feature] = 0
			u2[feature] = 0
		self.b1 = 0
		self.b2 = 0
		beta1 = 0		# Cached bias 1
		beta2 = 0		# Cached bias 2
		c = 1				# Example counter

		# Learn perceptron weights
		for it in range(self.maxIterations):
			for identifier in self.identifiers:
				x = self.X[identifier]
				y1 = self.Y1[identifier]
				y2 = self.Y2[identifier]
				# Compute activations
				a1 = sum([self.w1[d] * x[d] + self.b1 for d in x])
				a2 = sum([self.w2[d] * x[d] + self.b2 for d in x])
				# Update weights and biases
				if y1 * a1 <= 0:
					for d in x:
						self.w1[d] += y1 * x[d]
						u1[d] += y1 * c * x[d]
					self.b1 += y1
					beta1 += y1 * c
				if y2 * a2 <= 0:
					for d in x:
						self.w2[d] += y2 * x[d]
						u2[d] += y2 * c * x[d]
					self.b2 += y2
					beta2 += y2 * c
				# Increment counter
				c += 1

		# Average weights and biases
		for d in self.vocabulary:
			self.w1[d] -= u1[d] / float(c)
			self.w2[d] -= u2[d] / float(c)
		self.b1 -= beta1 / float(c)
		self.b2 -= beta2 / float(c)


	def writeModelToFile(self, filename):
		model = {'vocabulary': list(self.vocabulary), \
				'w1': self.w1, 'b1': self.b1, \
				'w2': self.w2, 'b2': self.b2}
		with open(filename, "w", encoding="utf-8") as file_object:
			file_object.write(json.dumps(model, ensure_ascii=False))


def main():
	# Get training data from file
	filename = sys.argv[1]
	with open(filename, "r", encoding="utf-8") as file_object:
		lines = file_object.readlines()

	# Train perceptron model
	model = PerceptronModel(lines)
	model.parseTrainingData()
	model.trainVanillaPerceptron()
	model.writeModelToFile("vanillamodel.txt")
	model.trainAveragedPerceptron()
	model.writeModelToFile("averagedmodel.txt")


if __name__=="__main__":
	main()