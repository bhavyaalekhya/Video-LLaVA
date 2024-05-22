import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def are_sentences_similar_spacy(sentence1, sentence2, threshold=0.8):
    """
    Compare two sentences and determine if they are similar using spaCy.
    
    :param sentence1: First sentence
    :param sentence2: Second sentence
    :param threshold: Similarity threshold (0 to 1), default is 0.8
    :return: True if sentences are similar, False otherwise
    """
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    similarity = doc1.similarity(doc2)
    return similarity >= threshold

# Example usage
sentence1 = "Open-Open a can of tuna"
sentence2 = "Is the person opening a can of tuna?"

print(are_sentences_similar_spacy(sentence1, sentence2))
