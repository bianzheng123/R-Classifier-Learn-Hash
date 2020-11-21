# 科学计算包
import numpy as np
# 画图包
from matplotlib import pyplot as plt
# 封装好的KMeans聚类包
from sklearn.cluster import KMeans

# 设置中文字体个负号正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
X = np.loadtxt(r'test.txt')


# 定义数据预处理函数
def preprocess(X):
    # 特征缩放
    X_min, X_max = np.min(X), np.max(X)
    X = (X - X_min) / (X_max - X_min)
    # 根据数据需要进行不同的处理
    return X


# 确定k值
k = np.arange(1, 11)
jarr = []
for i in k:
    model = KMeans(n_clusters=i)
    model.fit(X)
    jarr.append(model.inertia_)
    # 给这几个点打标
    plt.annotate(str(i), (i, model.inertia_))
plt.plot(k, jarr)
plt.show()
# 经确定，k=4
k = 4
# 正式定义模型
model1 = KMeans(n_clusters=k)
# 跑模型
model1.fit(X)
# 需要知道每个类别有哪些参数
C_i = model1.predict(X)
# 还需要知道聚类中心的坐标
Muk = model1.cluster_centers_
# 画图
plt.scatter(X[:, 0], X[:, 1], c=C_i, cmap=plt.cm.Paired)
# 画聚类中心
plt.scatter(Muk[:, 0], Muk[:, 1], marker='*', s=60)
for i in range(k):
    plt.annotate('中心' + str(i + 1), (Muk[i, 0], Muk[i, 1]))
plt.show()
