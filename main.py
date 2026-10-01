import pandas as pd
import spacy, re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

if __name__ == '__main__': #apparently doing this will help with multithreading crashing, because without this it crashed on Windows
    file = "comcast_consumeraffairs_complaints.csv"
    text_column = "text"
    num_of_topics = 5
    top_n_words = 10

    #Load CSV file
    csv = pd.read_csv(file, encoding="latin1")
    csv = csv.dropna(subset=[text_column])
    raw_text = csv[text_column].to_list()[:2000]
    #remove the :2000 to go through the whole file. I put this there to test the code without going through everything every time

    #English NLP model
    nlp = spacy.load("en_core_web_sm")

    #Adding custom stop words to avoid clutter
    custom_stop_words = {'comcast', 'tell', 'say', 'call', 'company', 'technician', 'business', 'not'}
    nlp.Defaults.stop_words.update(custom_stop_words)

    for i in custom_stop_words:
        nlp.vocab[i].is_stop = True

    #Preprocessing, removing stop words, lemmatization
    def preprocessing(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        document = nlp(text)

        tokens = [token.lemma_ for token in document if not token.is_stop and not token.is_space and token.lemma_ not in custom_stop_words] #i added custom stop words but some would slip through anyway, so i added 'not in custom_stop_words'
        return tokens

    tokenized_doc_ = [preprocessing(text) for text in raw_text] #for gensim
    tokenized_doc = [" ".join(tokens) for tokens in tokenized_doc_] #for sklearn


    #TF-IDF Vectorizing, to be used later for LSA
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_doc)
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    #change max_df to affect how much is kept

    #BoW Vectorizing, to be used later for LDA
    bow_vectorizer = CountVectorizer(max_df=0.9, min_df=5, max_features=1000)
    bow_matrix = bow_vectorizer.fit_transform(tokenized_doc)
    bow_features = bow_vectorizer.get_feature_names_out()

    gensim_dictionary = Dictionary(tokenized_doc_)
    lsa_coherence_array = []
    lda_coherence_array = []

    def print_top_words(model, feature_names, num_words, append=""):
        for topic_idx, topic_weights in enumerate(model.components_):
            top_words_idx = topic_weights.argsort()[:-num_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            if append == "lsa":
                lsa_topics_to_grade.append(top_words)
            if append == "lda":
                lda_topics_to_grade.append(top_words)
            print(f"    Topic {topic_idx + 1}: {', '.join(top_words)}")

    for num_of_topics in range(3,11):

        lsa_model = TruncatedSVD(n_components=num_of_topics, random_state=42)
        lsa_model.fit(tfidf_matrix)

        lda_model = LatentDirichletAllocation(n_components=num_of_topics, random_state=42)
        lda_model.fit(bow_matrix)

        lsa_topics_to_grade = []
        lda_topics_to_grade = []

        print("\n LDA Topics using Bag of Words")
        print_top_words (lda_model, bow_features, top_n_words, "lda")
        cm = CoherenceModel(topics=lda_topics_to_grade, texts=tokenized_doc_, dictionary=gensim_dictionary, coherence='c_v')
        score = cm.get_coherence()
        print(f"Coherence: {score}")
        lda_coherence_array.append(score)

        print("\n LSA Topics using TF-IDF")
        print_top_words (lsa_model, tfidf_features, top_n_words, "lsa")
        cm = CoherenceModel(topics=lsa_topics_to_grade, texts=tokenized_doc_, dictionary=gensim_dictionary, coherence='c_v')
        score = cm.get_coherence()
        print(f"Coherence: {score}")
        lsa_coherence_array.append(score)

    lda_best_coherence_score = max(lda_coherence_array)
    lda_best_index = lda_coherence_array.index(lda_best_coherence_score)
    print(f"Best coherence for LDA/BOW is {lda_best_coherence_score} at n = {lda_best_index + 3}") #+3 because the array starts from zero, but we started from topics=3, making n=3 the zeroth index
    lsa_best_coherence_score = max(lsa_coherence_array)
    lsa_best_index = lsa_coherence_array.index(lsa_best_coherence_score)
    print(f"Best coherence for LSA/TF-IDF is {lsa_best_coherence_score} at n = {lsa_best_index + 3}")


    #matplotlib
    k_values = list(range(3,11))
    plt.figure(figsize=(8,5))

    plt.plot(k_values, lda_coherence_array, marker='o', color='blue',label='LDA/BOW')
    plt.plot(k_values, lsa_coherence_array, marker='s', color='orange',label='LSA/TF-IDF')

    plt.title('Optimal number of topics based on Coherence Score')
    plt.xlabel('Number of Topics (k)')
    plt.ylabel('Coherence Score')

    plt.xticks(k_values)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig("comparison.png")
    print("Graph saved as \'comparison.png\'")
