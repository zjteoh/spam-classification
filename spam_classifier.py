import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

'''
input: train_dir => path to directory of training emails

return: dictionary of 3000 most common words in the training set
'''
def make_dict(train_dir):
	all_email_paths = [os.path.join(train_dir,f) for f in sorted(os.listdir(train_dir))] 
	all_words = []

	for email_path in all_email_paths:
		with open(email_path) as lines:
			for idx, line in enumerate(lines):
				if idx == 2:
					words = line.split()
					words = [word for word in words if len(word) > 1 and word.isalpha()]
					all_words += words
          

	dictionary = Counter(all_words)
	dictionary = dictionary.most_common(3000)

	return dictionary

'''
input: dictionary => word dictionary, mail_dir => directory containing emails set

return: List[List[int]] => features matrix of Ax3000, where A = size of emails set
'''
def extract_features(dictionary, mail_dir):
	all_email_paths = [os.path.join(mail_dir,f) for f in sorted(os.listdir(mail_dir))] 
	features_matrix = np.zeros((len(all_email_paths), 3000))
	row = 0

	for email_path in all_email_paths:
		with open(email_path) as lines:
			for idx, line in enumerate(lines):
				if idx == 2:
					words = line.split()
					for word in words:
						for i, kv in enumerate(dictionary):
							_word, _count = kv
							if word == _word:
								features_matrix[row, i] = words.count(word)
		row += 1

	return features_matrix

train_dir = './ling-spam/train-mails'
test_dir  = './ling-spam/test-mails'
dictionary = make_dict(train_dir)

# features
train_features = extract_features(dictionary, train_dir)
test_features = extract_features(dictionary, test_dir)

# labels
train_labels = np.zeros(702)
train_labels[301:702] = 1
test_labels = np.zeros(260)
test_labels[130:260] = 1

# naive bayes 
model_nb = MultinomialNB()
model_nb.fit(train_features, train_labels)
result_labels_nb = model_nb.predict(test_features)

# svm
model_svc = SVC()
model_svc.fit(train_features, train_labels)
result_labels_svc = model_svc.predict(test_features)

# confusion matrix
print(confusion_matrix(test_labels, result_labels_nb))
print(confusion_matrix(test_labels, result_labels_svc))

