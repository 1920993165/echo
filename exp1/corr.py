import pandas as pd
 
# 创建示例数据
data = {
    'x': [10, 15, 20, 15, 25, 30, 20, 25, 30, 35, 40, 45],
    'y': [100, 150, 200, 150, 250, 300, 200, 250, 300, 350, 400, 450]
}


data = {
    'x': [2.7,5.6,3.9,5.2,5.8,6.7],
    'y': [57,70.2,58.3,74.8,72.3,84]
}

# data = {
#     'x': [2.6,5.5,3.5,4.0,5.4,5.9],
#     'y': [56.3,64.3,52.5,65.1,67.7,74.5]
# }

data = {
    'x': [5.0,5.8,5.2,4.5,4.8,4.5,3.9,6.2],
    'y': [60.7,71.9,73.0,62.4,53.0,70.4,73.3,82.8]
}
data = {
    'x': [4.8,5.3,3.7,3.4,5.1,4.6,4.0,4.6],
    'y': [68.6,66.3,53.9,66.4,60.2,57.7,71.2,63.1]
}


# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 计算皮尔逊相关系数
pearson_corr = df['x'].corr(df['y'], method='pearson')
print("Pearson Correlation Coefficient:", pearson_corr)
 
# 计算Spearman相关系数
spearman_corr = df['x'].corr(df['y'], method='spearman')
print("Spearman Correlation Coefficient:", spearman_corr)
 
# 计算Kendall相关系数
kendall_corr = df['x'].corr(df['y'], method='kendall')
print("Kendall Correlation Coefficient:", kendall_corr)

