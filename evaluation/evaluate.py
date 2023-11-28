import pandas as pd
from chat_gpt import ChatGPT

MAX_MOVIE_HISTORY = 15
BASE_PROMPT = """We are interested in predicting the serendipity of a movie recommendation for a user. We understand serendipity as the pleasent surprise in the recommendation. We will evaluate in a scale of 1 to 5. The following is a list of movies the user watched, and its corresponding rating.

"""
INSTRUCTIONS_PROMPT = """The following movie was recommended to the user:

"""
QUESTION_PROMPT = """
What would you rate the serendipity of this recommendation in a scale of 1 to 5? Please, first give your thoughts, and afterwards answer in the format "Serendipity: {value}". You must answer with a numeric value. If you are unsure, you still have to guess the serendipity value.
"""
RETRIES = 3

def build_movie_description(row, movies):
    movie = movies[movies['movieId'] == row['movieId']].iloc[0]
    prompt = ''
    prompt += f"Title: {movie['title']}\n" 
    prompt += f"Directed by: {movie['directedBy']}\n" 
    prompt += f"Starring: {movie['starring']}\n" 
    prompt += f"Genres: {movie['genres']}\n"
    prompt += f"The user rated it as: {row['rating']}/5\n"
    return prompt


def get_prompt(row, training, movies):
    watched_movies = training[training['userId'] == row['userId']][:MAX_MOVIE_HISTORY]

    prompt = BASE_PROMPT
    for _, movie in watched_movies.iterrows():
        prompt += build_movie_description(movie, movies) + '\n'
    prompt += INSTRUCTIONS_PROMPT
    prompt += build_movie_description(row, movies)
    prompt += QUESTION_PROMPT

    return prompt


if  __name__ == '__main__':
    training = pd.read_csv('data/serendipity-sac2018/training.csv')
    movies = pd.read_csv('data/serendipity-sac2018/movies.csv')
    answers = pd.read_csv('data/serendipity-sac2018/answers.csv')
    answers['gpt-rating'] = None
    chat_gpt = ChatGPT()

    i = 0
    try:
        for index, row in answers.iterrows():
            print('evaluating', i,'of', len(answers))
            for t in range(RETRIES):
                try:
                    prompt = get_prompt(row, training, movies)
                    rating = chat_gpt.rate_prompt(prompt)
                    answers.at[index, 'gpt-rating'] = rating
                    break
                except:
                    print('retrying')
            i += 1
    except Exception as e:
        print('got', e, 'error')

    answers.to_csv('data/evaluated_data.csv')

