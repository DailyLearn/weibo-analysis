import re
import jieba
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#1.提取正文中的话题，以及包含“刺客（可修改为其他关键词）”的话题
def filter_topics_with_assassin(df, text_column='微博正文'):
    """
    从指定的文本列中提取话题，并筛选包含“电影节”的条目。

    参数:
    df (pd.DataFrame): 包含文本列的 DataFrame。
    text_column (str): 文本列的列名，默认值为 '微博正文'。

    返回:
    pd.DataFrame: 包含“电影节”话题的条目 DataFrame。
    """
    #确保微博正文是字符串
    df[text_column] = df[text_column].astype(str)

    # 定义提取话题的函数
    def extract_topics(text):
        # 匹配 "#xxx#" 格式的文本
        pattern = r'#(.*?)#'
        matches = re.findall(pattern, text)
        return matches if matches else None

    # 应用到指定文本列，提取话题
    
    df['话题'] = df[text_column].apply(extract_topics)

    # 筛选包含“刺客”的话题
    df['电影节话题'] = df['话题'].apply(lambda x: [topic for topic in x if '电影节' in topic] if x else None)

    # 筛选出包含“刺客”话题的条目
    filtered_df = df[df['电影节话题'].notnull()]

    return filtered_df

# 使用示例
# filtered_df = filter_topics_with_assassin(df2022)
# print(filtered_df)


#2.统计各话题频率
def count_assassin_topics(df, topic_column='电影节话题'):
    """
    统计 DataFrame 中每个“电影节话题”的频率。

    参数:
    df (pd.DataFrame): 包含话题数据的 DataFrame。
    topic_column (str): 包含“电影节话题”的列名，默认为 '电影节话题'。

    返回:
    pd.DataFrame: 统计每个话题出现频率的 DataFrame。
    """
    # 展平所有话题为一个列表
    all_topics = [topic for sublist in df[topic_column].dropna() for topic in sublist]
    
    # 使用 Counter 统计每个话题的频率
    topic_counts = Counter(all_topics)
    
    # 将统计结果转为 DataFrame 并按频率排序
    topic_count_df = pd.DataFrame(topic_counts.items(), columns=['话题', '频率']).sort_values(by='频率', ascending=False)
    
    return topic_count_df

# 使用示例
# 假设 df2022 是包含“电影节话题”的 DataFrame
# assassin_topic_counts = count_assassin_topics(df2022, '刺客话题')
# print(assassin_topic_counts)


#3.抽取话题包含指定关键词列表的内容
def filter_entries_by_existing_topics(df, keywords, topic_column='话题'):
    """
    根据输入的指定话题列表，通过现有的 '话题' 列筛选包含这些话题的条目。

    参数:
    df (pd.DataFrame): 包含话题的 DataFrame。
    keywords (list): 要筛选的指定话题。
    topic_column (str): 包含话题的列名，默认为 '话题'。

    返回:
    pd.DataFrame: 包含指定话题的条目 DataFrame。
    """
    #确保话题是字符串
    df[topic_column] = df[topic_column].astype(str)
    
    # 过滤出包含任意指定话题的条目
    filtered_df = df[df['话题'].apply(lambda keyword: keyword in keywords)]
    
    return filtered_df

# 使用示例
# 假设 df2022 有 "话题" 列
# topic = ''
# filtered_df = filter_entries_by_existing_topics(df2022, topic_list)
# print(filtered_df[['微博正文', '话题']])

#4.根据话题列表抽取对应内容
def filter_by_listed_topics(df, topic_list, topic_column='话题'):
    """
    根据输入的指定话题列表，通过现有的 '话题' 列筛选包含这些话题的条目。

    参数:
    df (pd.DataFrame): 包含话题的 DataFrame。
    topic_list (list): 要筛选的指定话题列表。
    topic_column (str): 包含话题的列名，默认为 '话题'。

    返回:
    pd.DataFrame: 包含指定话题的条目 DataFrame。
    """
    # 确保话题列是字符串
    df[topic_column] = df[topic_column].astype(str)
    
    # 过滤出包含指定话题列表中任意一个话题的条目
    filtered_df = df[df[topic_column].apply(lambda topics: any(topic in topics for topic in topic_list))]
    
    return filtered_df


#5.抽取包括特定关键词的话题
def filter_assassin_topics_with_keyword(df, keyword, topic_column='话题'):
    """
    筛选出包含特定关键词的“刺客话题”条目。

    参数:
    df (pd.DataFrame): 包含刺客话题的 DataFrame。
    keyword (str): 要查找的关键词（例如 '董宇辉'）。
    topic_column (str): 包含刺客话题的列名，默认为 '话题'。

    返回:
    pd.DataFrame: 包含该关键词的刺客话题的 DataFrame。
    """
    # 筛选出包含关键词的“刺客话题”
    filtered_df = df[df[topic_column].apply(lambda topics: any(keyword in topic for topic in topics) if topics else False)]
    
    return filtered_df

# 使用示例
# 假设 df2022 是包含“刺客话题”的 DataFrame
# filtered_df = filter_assassin_topics_with_keyword(df2022, '董宇辉', '话题')
# 输出包含“董宇辉”的“话题”条目
# print(filtered_df[['微博正文', '话题']])

#6.抽取指定话题下热度最高的几条微博
def calculate_top_n_hot_posts(df, n=5):
    """
    根据点赞数、转发数、评论数计算每条微博的热度，并筛选出热度最高的 n 条。

    参数:
    df (pd.DataFrame): 包含微博数据的 DataFrame，必须有'点赞数'、'转发数'、'评论数'列。
    n (int): 要返回的热度最高的微博条目数，默认是5。

    返回:
    pd.DataFrame: 包含热度最高的 n 条微博的 DataFrame。
    """
    # 定义热度计算公式
    df['热度'] = df['点赞数'] + 2 * df['转发数'] + 3 * df['评论数']

    # 筛选出热度最高的 n 条微博
    top_n_hot_posts = df.nlargest(n, '热度')

    # 返回包含微博正文及相关数据的 DataFrame
    return top_n_hot_posts[['微博正文', '点赞数', '转发数', '评论数', '热度']]

# 使用示例
# top_5_hot_posts = calculate_top_n_hot_posts(df, 5)
# print(top_5_hot_posts)


#7.数据表中微博正文TF-IDF关键词提取，需导入包
#import jieba
#from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_keywords(df, text_column, stopwords_path, max_features=30):
    """
    提取指定文本列的 TF-IDF 关键词。

    参数:
    df (pd.DataFrame): 包含文本数据的 DataFrame。
    text_column (str): 要处理的文本列的列名。
    stopwords_path (str): 停用词文件路径。
    max_features (int): 提取的关键词数量，默认为 100。

    返回:
    pd.DataFrame: 包含关键词和对应 TF-IDF 权重的 DataFrame。
    """
    # 读取停用词文件
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f]

    # 使用 jieba 分词
    df['分词结果'] = df[text_column].apply(lambda x: ' '.join(jieba.cut(x)))

    # 将所有分词结果合并为一个整体文本
    all_text = ' '.join(df['分词结果'].tolist())

    # 使用 TfidfVectorizer 进行 TF-IDF 计算，并去除停用词
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 stop_words=stop_words,
                                 token_pattern=r"(?u)\b\w{2,}\b")

    tfidf_matrix = vectorizer.fit_transform([all_text])

    # 提取关键词及其对应的权重
    tfidf_feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # 将结果转为 DataFrame
    tfidf_df = pd.DataFrame({'词语': tfidf_feature_names, 'TF-IDF权重': tfidf_scores})

    # 按照 TF-IDF 权重降序排序，返回前 n 个关键词
    top_keywords = tfidf_df.sort_values(by='TF-IDF权重', ascending=False).head(max_features)

    return top_keywords


#8.两个话题下文本余弦相似度比较，需要导入包
#import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity

def compare_topics_cosine_similarity(assassin_data, topic1, topic2, stopwords_path):
    """
    比较两个不同话题下微博文本的余弦相似度。
    
    参数:
        - assassin_data: 包含微博文本数据的DataFrame
        - topic1: 第一个话题的名称 (str)
        - topic2: 第二个话题的名称 (str)
        - stopwords_path: 停用词路径 (str)
    
    返回:
        - 两个话题之间的余弦相似度 (float)
    """
    
    # 提取两个不同话题下的微博文本
    topic1_data = assassin_data[assassin_data['微博正文'].str.contains(topic1)]
    topic2_data = assassin_data[assassin_data['微博正文'].str.contains(topic2)]

    # 生成每个话题下的TF-IDF关键词及权重
    top_keywords_topic1 = extract_tfidf_keywords(topic1_data, '微博正文', stopwords_path)
    top_keywords_topic2 = extract_tfidf_keywords(topic2_data, '微博正文', stopwords_path)

    # 从 DataFrame 中提取关键词和TF-IDF权重
    tfidf_dict_topic1 = dict(zip(top_keywords_topic1['词语'], top_keywords_topic1['TF-IDF权重']))
    tfidf_dict_topic2 = dict(zip(top_keywords_topic2['词语'], top_keywords_topic2['TF-IDF权重']))

    # 构建词汇表（所有关键词的集合）
    vocab = set(tfidf_dict_topic1.keys()).union(set(tfidf_dict_topic2.keys()))

    # 为两个话题生成词向量
    vector_topic1 = np.array([tfidf_dict_topic1.get(word, 0) for word in vocab])
    vector_topic2 = np.array([tfidf_dict_topic2.get(word, 0) for word in vocab])

    # 计算余弦相似度
    cos_sim = cosine_similarity([vector_topic1], [vector_topic2])

    # 返回余弦相似度
    return cos_sim[0, 0]

# 示例调用
#topic1 = '雪糕刺客'
#topic2 = '水果刺客'
#stopwords_path = 'path/to/stopwords.txt'

#similarity = compare_topics_cosine_similarity(assassin_data, topic1, topic2, stopwords_path)
#print(f"话题 '{topic1}' 和 '{topic2}' 之间的余弦相似度: {similarity:.4f}")


#9.在一定条件下抽样，如排除部分话题，并在剩下频率大于10的话题中抽样
import random

def remove_and_sample_topics(df, topic_list1, topic_column='话题', freq_column='频率', sample_size=20, min_frequency=10):
    """
    从 DataFrame 中删除 `topic_list1` 中的指定话题，并从频率大于等于 `min_frequency` 的话题中随机抽取 `sample_size` 个话题。

    参数:
    df (pd.DataFrame): 包含话题的 DataFrame。
    topic_list1 (list): 要删除的话题列表。
    topic_column (str): 包含话题的列名，默认为 '话题'。
    freq_column (str): 记录话题频率的列名，默认为 '频率'。
    sample_size (int): 要抽取的样本数量。
    min_frequency (int): 抽样的话题频率下限，默认为 10。

    返回:
    pd.DataFrame: 经过删除和随机抽样的 DataFrame。
    """
    # 删除 topic_list1 中的话题
    df_filtered = df[~df[topic_column].isin(topic_list1)]
    
    # 过滤出频率大于等于 min_frequency 的话题
    df_valid_freq = df_filtered[df_filtered[freq_column] >= min_frequency]
    
    # 从中随机抽取 sample_size 个话题
    sampled_df = df_valid_freq.sample(n=sample_size, random_state=42)
    
    return sampled_df

# 使用示例
# 假设 df 是包含 '话题' 和 '频率' 列的 DataFrame
# topic_list1 是要删除的话题列表
# sample_df = remove_and_sample_topics(df, topic_list1, '话题', '频率', 20, 10)
# print(sample_df)