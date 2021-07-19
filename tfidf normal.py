"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from pprint import pprint
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from data_loader.cnews_loader import *
"""
import os
from collections import Counter
import re
import jieba
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_file(filename):
    contents, labels = [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
        return contents, labels


# 设置数据读取、模型、结果保存路径
base_dir = 'D:\\collen\\tfidf\\dataset'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = 'D:\\collen\\tfidf\\dataset'
save_path = os.path.join(save_dir, 'best_validation')
# 读取数据
train_contents, train_labels = read_file(train_dir)
test_contents, test_labels = read_file(test_dir)


# 移除特殊字符


def clear_character(sentence):
    pattern1 = '\[.*?\]'
    pattern2 = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line1 = re.sub(pattern1, '', sentence)
    line2 = re.sub(pattern2, '', line1)
    new_sentence = ''.join(line2.split())
    return new_sentence


train_text = list(map(lambda s: clear_character(s), train_contents))
test_text = list(map(lambda s: clear_character(s), test_contents))
# jieba分词
train_seg_text = list(map(lambda s: jieba.lcut(s), train_text))
test_seg_text = list(map(lambda s: jieba.lcut(s), test_text))
# 读取停用词
stop_words_path = "D:/collen/tfidf/dataset/baidu_stopwords.txt"


def get_stop_words():
    file = open(stop_words_path, 'rb').read().decode('gbk').split('\r\n')
    return set(file)


stopwords = get_stop_words()


# 去掉文本中的停用词
def drop_stopwords(line, stopwords):
    line_clean = []
    for word in line:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean


train_st_text = list(map(lambda s: drop_stopwords(s, stopwords), train_seg_text))
test_st_text = list(map(lambda s: drop_stopwords(s, stopwords), test_seg_text))

# 标签映射
le = LabelEncoder()
le.fit(train_labels)
label_train_id = le.transform(train_labels)
label_test_id = le.transform(test_labels)

# tfidf
train_c_text = list(map(lambda s: ' '.join(s), train_st_text))
test_c_text = list(map(lambda s: ' '.join(s), test_st_text))
tfidf_model = TfidfVectorizer(binary=False, token_pattern=r"(?u)\b\w+\b")
train_Data = tfidf_model.fit_transform(train_c_text)
test_Data = tfidf_model.transform(test_c_text)
# LR分类训练
classifier = LogisticRegression()
classifier.fit(train_Data, label_train_id)
pred = classifier.predict(test_Data)
from sklearn.metrics import classification_report

print(classification_report(label_test_id, pred, digits=4))

# tuner
"""
parameters = {
    'tfidf__max_df': (0.75,),# 出现频率0.75以上的就把他忽略了
    # 'tfidf__stop_words':('english',stopwords),
    'tfidf__norm': ('l2',),
    'tfidf__use_idf': (True,),
    'tfidf__smooth_idf': (True,),
    'tfidf__max_features': (None,),
    # 'tfidf__ngram_range':((1, 1), (1, 2),(2, 2)),  # unigrams or bigrams

    #     'clf__max_iter': (20,),
    'clf__penalty': ('l1', 'l2'),
    # 'clf__tol': (0.0001,0.00001,0.000001),
    'clf__solver': ('liblinear', 'saga',),
}
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")),
    ('clf', LogisticRegression()),
])

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(train_c_text, label_train_id)
print("done in %0.3fs" % (time() - t0))
print()
"""
#
y_val = test_labels
y_pre = le.inverse_transform(pred)

confm = metrics.confusion_matrix(y_pre, y_val)
categories = le.classes_

plt.figure(figsize=(8, 8))
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False, linewidths=.8,
            cmap="YlGnBu")
plt.xlabel('True label', size=14)
plt.ylabel('Predicted label', size=14)
plt.xticks(np.arange(10) + 0.5, categories, size=12)
plt.yticks(np.arange(10) + 0.3, categories, size=12)
plt.show()
# DecisionTree
from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1.fit(train_Data, label_train_id)
pred1 = clf1.predict(test_Data)
print(classification_report(label_test_id, pred1, digits=4))
# MultinomialNB
from sklearn.naive_bayes import MultinomialNB
clf2 = MultinomialNB()
clf2.fit(train_Data, label_train_id)
pred2 = clf2.predict(test_Data)
print(classification_report(label_test_id, pred2, digits=4))