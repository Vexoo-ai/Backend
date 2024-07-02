import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
from serpapi import GoogleSearch, BingSearch
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LinearRegression, LogisticRegression

load_dotenv()

def call_google_search_engine(query):
    
    serpapi_api_key = os.getenv('serpapi_api_key')
    search_engines = ["google", "duckduckgo"]

    for engine in search_engines:
        params = {
            "q": query,
            "engine": engine,
            "hl": "en",
            "gl": "in",
            "google_domain": "google.com",
            "api_key": serpapi_api_key,
        }
        # Call the API with these parameters
    search = GoogleSearch(params)
    return search.get_dict()

def call_bing_search_engine(query):
   
    serpapi_api_key = os.getenv('serpapi_api_key')
    params = {
        "q": query,
        "hl": "en",
        "gl": "in",
        "api_key": serpapi_api_key,
    }

    search = BingSearch(params)
    return search.get_dict()

def call_search_engine(query):
    print("Google")
    google_result = call_google_search_engine(query)
    print("Bing")
    bing_result = call_bing_search_engine(query)
  
    merged_results = {**google_result, **bing_result}

    return rank_results(merged_results, query)  # Call the ranking function directly here

def rank_results(results, query):
    start_time = time.time()
    # Extract the search results from the dictionary
    search_results = results.get('organic_results', []) 

    # Initialize an empty list to store the snippets
    texts = []

    # Extract the 'snippet' and 'position' from each search result
    positions = []
    for result in search_results:
        try:
            snippet = result['snippet']
            position = result['position']  
            texts.append(snippet)
            positions.append(position)
        except KeyError:
            pass  

    # Add the query to the texts
    texts.append(query)

    # Create a TF-IDF Vectorizer and fit it to the texts
    vectorizer = TfidfVectorizer().fit(texts)

    # Transform the texts into a matrix of TF-IDF features
    tfidf_matrix = vectorizer.transform(texts)

    # Compute the cosine similarity between the query and each of the search results
    cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Normalize the positions to the same range as the cosine similarities
    positions = MinMaxScaler().fit_transform(np.array(positions).reshape(-1, 1)).flatten()

    # Combine the cosine similarities and positions to get a final score for each result
    final_scores = cosine_similarities.flatten() + positions

    # Create a DataFrame with the features
    features = pd.DataFrame({
        'cosine_similarity': cosine_similarities.flatten(),
        'position': positions,
        'snippet_length': [len(snippet) for snippet in texts[:-1]],
        'query_occurrences': [snippet.count(query) for snippet in texts[:-1]]
    })

    # Train a Logistic Regression model on all the data
    model = LinearRegression()
    model.fit(features, final_scores)

    # Use the model to predict the scores of the data
    predicted_scores = model.predict(features)

    # Get the indices that would sort the predicted_scores array in descending order
    sorted_indices = np.argsort(predicted_scores)[::-1]

    # Use these indices to sort the search results
    ranked_results = [search_results[i] for i in sorted_indices]
  
    # Wrap the ranked results in a dictionary under the key 'organic_results'
    ranked_results = {'organic_results': ranked_results}
    
    return ranked_results


def parse_published_date(published_date_str):
    # Use regular expression to find the number and time unit
    match = re.search(r"(\d+)\s*(hour|day|week|month)s?", published_date_str, re.I)
    if not match:
        return datetime.now()
    
    number = int(match.group(1))
    unit = match.group(2).lower()
    
    # Dictionary to map time units to corresponding timedelta arguments
    time_delta_args = {
        'hour': {'hours': number},
        'day': {'days': number},
        'week': {'weeks': number},
        'month': {'days': number * 30}  # Simplified assumption: 1 month = 30 days
    }
    
    # Calculate the timedelta based on extracted unit and number
    try:
        return datetime.now() - timedelta(**time_delta_args[unit])
    except KeyError:
        # If unit is not found in the dictionary, return the current time
        return datetime.now()

def call_image_search_engine(query):
    serpapi_api_key = os.getenv('serpapi_api_key')
    search_engine = "google_images"

    params = {
        "q": query,
        "engine": search_engine,
        "api_key": serpapi_api_key,
    }

    search = GoogleSearch(params)
    image_results = search.get_dict()
    results = image_results.get("images_results", [])

    for result in results:
        result['parsed_date'] = parse_published_date(result.get('published_date', ''))

    sorted_results = sorted(results, key=lambda x: x.get('parsed_date', datetime.now()), reverse=True)

    limited_results = sorted_results[:5]

    result_dict = {f"Image {idx+1}": {"Title": result.get('title', 'Unknown'), "Link": result.get('original', 'Unknown')} for idx, result in enumerate(limited_results)}

    return result_dict        
        

def call_youtube_search_engine(query):
    serpapi_api_key = os.getenv('serpapi_api_key')
    search_engine = "youtube"

    params = {
        "search_query": query,
        "engine": search_engine,
        "api_key": serpapi_api_key,
        "num": 5  # Limit the number of results to 5
    }

    search = GoogleSearch(params)
    youtube_results = search.get_dict()
    results = youtube_results["video_results"]
    
    relevant_results = []
    for movie in results:
        if 'title' in movie and 'link' in movie and 'published_date' in movie:
            published_date = parse_published_date(movie['published_date'])
            relevant_results.append((movie['title'], movie['link'], published_date))

    # Sort by publish date in descending order (most recent first)
    relevant_results.sort(key=lambda x: x[2], reverse=True)

    # Create a dictionary from the relevant results
    result_dict = {f"Video {idx+1}": {"Title": title, "Link": link, "Published Date": publish_date.strftime("%Y-%m-%d %H:%M:%S")} for idx, (title, link, publish_date) in enumerate(relevant_results[:5])}

    return result_dict
