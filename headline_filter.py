import os

_environ = dict(os.environ)  # or os.environ.copy()
try:

    # os.environ['CUDA_PATH'] = os.environ['CUDA_PATH_V9_0']
    # os.environ['Path'] = os.path.join(os.environ['CUDA_PATH_V9_0'], 'bin') + '; ' +\
    #                      os.path.join(os.environ['CUDA_PATH_V9_0'], 'libnvvp') + '; ' + os.environ['Path']

    # os.environ['Path'] = os.environ['Path'].replace(os.environ['CUDA_PATH'], os.environ['CUDA_PATH_V9_0'])
    # os.environ['CUDA_PATH'] = os.environ['CUDA_PATH_V9_0']
    # os.environ.update(os.environ)
    from sentiment_model_demo import get_prediction

finally:
    # os.environ.clear()
    # os.environ.update(_environ)
    os.environ.update(os.environ)
    pass
import sqlite3

# with sqlite3.connect('../Headline Trend Analysis/articles.db') as conn:
#     cur = conn.cursor()
#     # conn.row_factory = sqlite3.Row
#     cur.execute('select query, title, date from article where sentiment is null')
#     for query, title, date in cur.fetchall():
#         sentiment = get_prediction(title)
#         cur.execute('update article set sentiment = (?) where query = (?) and title = (?) and date = (?)',
#                     (sentiment, query, title, date))

# with sqlite3.connect('../Headline Trend Analysis/articles.db') as conn:
#     conn.row_factory = sqlite3.Row
#     cur = conn.cursor()
#     cur.execute('select query, title, date, full_text from article where full_text_sentiment is null')
#     insert_cur = conn.cursor()
#     while True:
#         insert_params = []
#         for row in cur.fetchmany(100):
#             sentiment = get_prediction(row['full_text'])
#             insert_params.append((sentiment, row['query'], row['title'], row['date']))
#         print(insert_params)
#         exit()
#         insert_cur.executemany('update article set full_text_sentiment = (?) where query = (?) and title = (?) and date = (?)',
#                     insert_params)
