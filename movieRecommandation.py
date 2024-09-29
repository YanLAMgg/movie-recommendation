import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 导入数据
movies = pd.read_csv('movie.csv')
ratings = pd.read_csv('rating.csv')

# 打印数据以确认加载成功
print(movies.head())
print(ratings.head())

# 选择前 1000 个用户，减少数据集大小
ratings_subset = ratings[ratings['userId'] <= 1000]

# 2. 构建稀疏用户-电影评分矩阵
# 使用 SciPy 的 csr_matrix 来构建稀疏矩阵
user_movie_matrix_sparse = csr_matrix((ratings_subset['rating'], (ratings_subset['userId'], ratings_subset['movieId'])))

# 3. 计算用户相似度矩阵（基于余弦相似度）
user_similarity = cosine_similarity(user_movie_matrix_sparse)


# 4. 推荐系统函数
def recommend_movies(user_id, user_movie_matrix_sparse, user_similarity, top_n=10):
    # 获取相似用户列表
    similar_users = np.argsort(-user_similarity[user_id - 1])  # 相似度从高到低排序
    similar_users = similar_users[:top_n]  # 获取最相似的 top_n 用户

    # 获取相似用户的平均评分
    similar_ratings = user_movie_matrix_sparse[similar_users].mean(axis=0).A1  # A1 将矩阵转为数组
    recommended_movies = pd.Series(similar_ratings).sort_values(ascending=False)

    # 获取用户已观看过的电影
    watched_movies = user_movie_matrix_sparse[user_id].nonzero()[1]

    # 排除已观看的电影
    recommended_movies = recommended_movies.drop(watched_movies)

    return recommended_movies.head(top_n)


# 5. 为某个用户推荐电影
user_id = 1
recommended = recommend_movies(user_id, user_movie_matrix_sparse, user_similarity)
print(f"movies recommended to users {user_id}:\n", recommended)

# 6. 合并推荐电影的详细信息
recommended_movie_ids = recommended.index  # 获取推荐电影的ID
recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)].copy()  # 合并电影信息

# 添加平均评分到推荐电影的 DataFrame 中
recommended_movies.loc[:, 'average_rating'] = recommended.values  # 将评分信息添加到 DataFrame

# 7. 可视化推荐结果
plt.figure(figsize=(10, 6))
sns.barplot(x='average_rating', y='title', data=recommended_movies)
plt.title(f"movies recommended to users {user_id} ")
plt.tight_layout()
plt.show()

