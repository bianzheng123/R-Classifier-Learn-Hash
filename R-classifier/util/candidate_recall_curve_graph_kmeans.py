import json
import matplotlib.pyplot as plt
import numpy as np


def get_cluster(json_dir):
    x_arr = []
    x_95_arr = []
    y_arr = []
    with open(json_dir, 'r') as file:
        json_data = json.load(file)
        for ele in json_data:
            if ele['recall'] == 0.0:
                continue
            x_arr.append(ele['n_candidates_avg'])
            x_95_arr.append(ele['n_candidates_95'])
            y_arr.append(ele['recall'])
    return x_arr, x_95_arr, y_arr


fname = '../result/graph_kmeans/'
dir_kmeans_0 = 'kmeans_0.json'
dir_learn_on_graph_0 = 'learn-on-graph_0.json'
dir_intersect = 'intersect_result.json'
dir_union = 'union_result.json'

cls_kmeans_0 = get_cluster(fname + dir_kmeans_0)
cls_learn_on_graph_0 = get_cluster(fname + dir_learn_on_graph_0)
cls_intersect = get_cluster(fname + dir_intersect)
cls_union = get_cluster(fname + dir_union)

# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
line1_1, = plt.plot(cls_kmeans_0[0], cls_kmeans_0[2], marker='o', linestyle='solid', color='#b9529f',
                    label='16 bin kmeans')
line1_2, = plt.plot(cls_kmeans_0[1], cls_kmeans_0[2], marker='v', linestyle='dotted', color='#b9529f',
                    label='16 bin kmeans 95-quantile')

line2_1, = plt.plot(cls_learn_on_graph_0[0], cls_learn_on_graph_0[2], marker='^', linestyle='solid', color='#3953a4',
                    label='16 bin hnsw_build_graph')
line2_2, = plt.plot(cls_learn_on_graph_0[1], cls_learn_on_graph_0[2], marker='v', linestyle='dotted', color='#3953a4',
                    label='16 bin hnsw_build_graph 95-quantile')

line3_1, = plt.plot(cls_intersect[0], cls_intersect[2], marker='<', linestyle='solid', color='#ed2024', label='intersect')
line3_2, = plt.plot(cls_intersect[1], cls_intersect[2], marker='v', linestyle='dotted', color='#ed2024',
                    label='intersect 95-quantile')

line4_1, = plt.plot(cls_union[0], cls_union[2], marker='<', linestyle='solid', color='#231f20', label='union')
line4_2, = plt.plot(cls_union[1], cls_union[2], marker='v', linestyle='dotted', color='#231f20',
                    label='union 95-quantile')

# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.title('graph kmeans')
plt.legend(loc='lower right', title="SIFT10K, 10-NN")

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
