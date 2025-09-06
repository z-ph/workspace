import pandas as pd
import numpy as np

# 读取CSV文件，使用utf-8-sig编码避免中文乱码
df = pd.read_csv('data/附件_女胎检测数据.csv', encoding='utf-8-sig')

# 创建结果目录
import os
result_dir = 'result/data_analysis'
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)

# 1. 基本数据信息
with open(os.path.join(result_dir, 'data_analysis.md'), 'w', encoding='utf-8') as f:
    f.write('# 女胎检测数据结构与特征分析\n\n')
    
    # 数据基本信息
    f.write('## 1. 数据基本信息\n\n')
    f.write(f'- 总记录数: {len(df)}\n')
    f.write(f'- 总列数: {len(df.columns)}\n\n')
    
    # 列名和数据类型
    f.write('## 2. 列信息\n\n')
    f.write('| 列索引 | 列名 | 数据类型 | 非空值数量 |\n')
    f.write('|--------|------|----------|------------|\n')
    
    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()
        f.write(f'| {i} | {col} | {dtype} | {non_null_count} |\n')
    
    # 异常数据统计
    f.write('\n## 3. 染色体非整倍体异常情况\n\n')
    abnormal_counts = df['染色体的非整倍体'].value_counts()
    if not abnormal_counts.empty:
        f.write('| 异常类型 | 数量 |\n')
        f.write('|----------|------|\n')
        for abnormal_type, count in abnormal_counts.items():
            f.write(f'| {abnormal_type} | {count} |\n')
    else:
        f.write('无异常记录\n')
    
    # 特征相关性分析
    f.write('\n## 4. 可能影响结果的关键特征分析\n\n')
    f.write('以下是可能影响染色体非整倍体检测结果的重要特征：\n\n')
    
    # 染色体相关特征
    f.write('### 4.1 染色体Z值特征\n')
    f.write('- 13号染色体的Z值\n')
    f.write('- 18号染色体的Z值\n')
    f.write('- 21号染色体的Z值\n')
    f.write('- X染色体的Z值\n\n')
    
    # 技术指标特征
    f.write('### 4.2 技术指标特征\n')
    f.write('- 原始读段数\n')
    f.write('- 在参考基因组上比对的比例\n')
    f.write('- 重复读段的比例\n')
    f.write('- 唯一比对的读段数\n')
    f.write('- GC含量\n')
    f.write('- 被过滤掉读段数的比例\n')
    f.write('- 各染色体的GC含量（13号、18号、21号）\n\n')
    
    # 孕妇相关特征
    f.write('### 4.3 孕妇相关特征\n')
    f.write('- 年龄\n')
    f.write('- 身高\n')
    f.write('- 体重\n')
    f.write('- 孕妇BMI\n')
    f.write('- 检测孕周\n')
    f.write('- 检测抽血次数\n')
    f.write('- IVF妊娠（是否试管婴儿）\n\n')
    
    # 特殊说明
    f.write('## 5. 数据预处理建议\n\n')
    f.write('- 注意Unnamed列存在缺失值，可能需要处理\n')
    f.write('- 染色体非整倍体列为空表示无异常，有值表示有异常\n')
    f.write('- 建议将目标变量转换为二元分类（有异常/无异常）或多分类（具体异常类型）\n')

print(f'数据分析完成，请查看{result_dir}/data_analysis.md文件')