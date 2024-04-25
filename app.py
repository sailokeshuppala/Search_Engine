from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


model_distilbert = SentenceTransformer('distilbert-base-nli-mean-tokens')


subset_df = pd.read_csv('subtitles.csv', nrows =10000)


batch_size = 32
num_subtitles = len(subset_df)
subtitle_embeddings = []
for i in range(0, num_subtitles, batch_size):
    batch_subtitles = subset_df['clean_text_lemma'].iloc[i:i+batch_size].values
    batch_embeddings = model_distilbert.encode(batch_subtitles)
    subtitle_embeddings.append(batch_embeddings)

subtitle_embeddings = np.concatenate(subtitle_embeddings)


def semantic_search(query, top_n=5):
    
    query_embedding = model_distilbert.encode([query])[0]

    
    similarities = cosine_similarity([query_embedding], subtitle_embeddings)[0]

    
    top_indices = similarities.argsort()[-top_n:][::-1]

    
    return subset_df['name'].iloc[top_indices].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = semantic_search(query)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

