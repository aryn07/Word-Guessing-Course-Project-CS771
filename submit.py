import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier

def generate_bigrams(word):
    return [word[i:i+2] for i in range(len(word) - 1)]

def preprocess_words(word_list):
    bigram_to_words = defaultdict(list)
    word_to_bigrams = {}
    
    for word in word_list:
        bigrams = generate_bigrams(word)
        unique_bigrams = sorted(set(bigrams))
        if len(unique_bigrams) > 2:
            unique_bigrams = unique_bigrams[:2]
        bigram_key = tuple(unique_bigrams)
        bigram_to_words[bigram_key].append(word)
        word_to_bigrams[word] = bigram_key
    
    return bigram_to_words, word_to_bigrams

def create_feature_matrix(word_to_bigrams, unique_bigrams):
    feature_matrix = []
    labels = []
    bigram_index = {bigram: i for i, bigram in enumerate(unique_bigrams)}
    
    for word, bigrams in word_to_bigrams.items():
        features = [0] * len(unique_bigrams)
        for bigram in bigrams:
            if bigram in bigram_index:
                features[bigram_index[bigram]] = 1
        feature_matrix.append(features)
        labels.append(word)
    
    return np.array(feature_matrix), np.array(labels)

def generate_candidate_words(bigram_list):
    from itertools import permutations
    
    # Generate permutations of bigrams and try to form words
    candidates = set()
    for perm in permutations(bigram_list):
        candidate = perm[0]
        for bigram in perm[1:]:
            if candidate[-1] == bigram[0]:
                candidate += bigram[1]
            else:
                break
        candidates.add(candidate)
    return candidates

################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
#  Non Editable Region Ending  #
################################
    bigram_to_words, word_to_bigrams = preprocess_words(words)
    unique_bigrams = sorted(set(bigram for bigrams in word_to_bigrams.values() for bigram in bigrams))
    X, y = create_feature_matrix(word_to_bigrams, unique_bigrams)
    
    model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=10, min_samples_split=2)
    model.fit(X, y)
    
    return {
        'model': model,
        'unique_bigrams': unique_bigrams,
        'bigram_to_words': bigram_to_words,
        'word_to_bigrams': word_to_bigrams,
        'original_words': words  # Keep the original word list for validation
    }

################################
# Non Editable Region Starting #
################################
def my_predict(model_data, bigram_list):
################################
#  Non Editable Region Ending  #
################################
    model = model_data['model']
    unique_bigrams = model_data['unique_bigrams']
    original_words = model_data['original_words']
    
    # Generate candidate words from the given bigrams
    candidate_words = generate_candidate_words(bigram_list)
    
    # Filter candidate words to retain only those in the original dictionary
    valid_words = [word for word in candidate_words if word in original_words]
    
    # If no valid words found, fall back to the model's prediction
    if not valid_words:
        feature_vector = [0] * len(unique_bigrams)
        for bigram in bigram_list:
            if bigram in unique_bigrams:
                feature_vector[unique_bigrams.index(bigram)] = 1
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        prediction = model.predict(feature_vector)[0]
        return [prediction]

    return valid_words[:2]

# Test the model
if __name__ == "__main__":
    # Load the dictionary
    dictionary_path = 'dict_secret'
    with open(dictionary_path, 'r') as file:
        dictionary = [line.strip() for line in file]

    # Fit the model
    model = my_fit(dictionary)

    # Predict words from bigrams
    bigrams = ('ab', 'bo', 'ou', 'ut')
    predictions = my_predict(model, bigrams)
    print(predictions)
