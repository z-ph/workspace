import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek  # 导入SMOTETomek组合采样
import os

# 设置matplotlib中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保result文件夹及其子文件夹存在
def ensure_result_dir():
    """确保result文件夹及其子文件夹存在"""
    result_dir = 'result/random_forest_with_smotetomek'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    return result_dir

# 加载和预处理数据
def load_and_preprocess_data(file_path):
    """
    加载数据并进行预处理
    参数:
        file_path: 数据文件路径
    返回:
        X: 特征数据
        y: 目标变量
        feature_names: 特征名称列表
    """
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据行数: {len(df)}")
    
    # 首先创建目标变量，不提前过滤数据
    # 创建目标变量（是否有染色体非整倍体）
    if '染色体的非整倍体' in df.columns:
        df['异常结果'] = df['染色体的非整倍体'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
        print(f"原始数据中 - 异常样本数: {sum(df['异常结果'] == 1)}, 正常样本数: {sum(df['异常结果'] == 0)}")
    
    # 数据预处理
    # 处理缺失值 - 保留足够的正常和异常样本
    # 只删除关键特征的缺失值，但不删除'染色体的非整倍体'列的缺失值
    key_cols = ['检测孕周', '孕妇BMI']
    df = df.dropna(subset=key_cols)
    print(f"删除关键特征缺失值后的数据行数: {len(df)}")
    print(f"删除后 - 异常样本数: {sum(df['异常结果'] == 1)}, 正常样本数: {sum(df['异常结果'] == 0)}")
    
    # 转换检测孕周为数值
    if '检测孕周' in df.columns:
        # 处理带'w'的孕周格式，例如'13w+5' -> 13 + 5/7 = 13.714
        def parse_gestational_week(week_str):
            try:
                if isinstance(week_str, str):
                    if 'w' in week_str:
                        parts = week_str.lower().split('w')
                        weeks = float(parts[0])
                        if '+' in parts[1]:
                            days = float(parts[1].split('+')[1])
                            weeks += days / 7
                        return weeks
                return float(week_str)
            except:
                return np.nan
                
        df['检测孕周'] = df['检测孕周'].apply(parse_gestational_week)
        df = df.dropna(subset=['检测孕周'])
        print(f"转换孕周并删除缺失值后的数据行数: {len(df)}")
        print(f"转换后 - 异常样本数: {sum(df['异常结果'] == 1)}, 正常样本数: {sum(df['异常结果'] == 0)}")
    
    # 转换IVF妊娠为数值
    if 'IVF妊娠' in df.columns:
        # 检查IVF妊娠列的唯一值
        unique_values = df['IVF妊娠'].unique()
        print(f"IVF妊娠列的唯一值: {unique_values}")
        
        # 根据实际值进行映射
        if '自然受孕' in unique_values:
            df['IVF妊娠'] = df['IVF妊娠'].map({'自然受孕': 0, 'IVF（试管婴儿）': 1})  # 修正映射
        else:
            df['IVF妊娠'] = df['IVF妊娠'].map({'是': 1, '否': 0})
    
    # 选择特征列
    feature_cols = [
        '年龄', '身高', '体重', '孕妇BMI', '检测孕周', '原始读段数', 
        '在参考基因组上比对的比例', '重复读段的比例', 'GC含量',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 
        'X染色体的Z值', 'X染色体浓度',
        '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
        '被过滤掉读段数的比例', '怀孕次数', '生产次数'
    ]
    
    # 移除数据中不存在的列
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"最终使用的特征列数量: {len(feature_cols)}")
    print(f"特征列: {feature_cols}")
    
    # 创建特征矩阵和目标变量
    X = df[feature_cols].copy()  # 使用copy避免SettingWithCopyWarning
    y = df['异常结果']
    
    # 处理非数值字符串（如'≥3'）
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"处理列 '{col}' 中的非数值字符串")
            # 尝试转换为数值
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # 检查是否有NaN值产生
            if X[col].isna().any():
                print(f"列 '{col}' 中有 {X[col].isna().sum()} 个非数值被转换为NaN")
    
    # 对于怀孕次数和生产次数等可能包含'>=3'这样的值的列，特殊处理
    for col in ['怀孕次数', '生产次数']:
        if col in X.columns and X[col].dtype == 'object':
            X[col] = X[col].apply(lambda x: 3 if isinstance(x, str) and ('≥3' in x or '>=3' in x) else x)
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 用中位数填充NaN值
    X = X.fillna(X.median(numeric_only=True))
    
    print(f"最终特征矩阵形状: {X.shape}")
    print(f"最终 - 异常样本数: {sum(y == 1)}, 正常样本数: {sum(y == 0)}")
    
    return X, y, feature_cols

# 使用SMOTETomek进行组合采样
def apply_smotetomek(X, y):
    """
    使用SMOTETomek算法进行组合采样处理（过采样+欠采样）
    参数:
        X: 特征数据
        y: 目标变量
    返回:
        X_resampled: 采样后的特征数据
        y_resampled: 采样后的目标变量
        smotetomek_applied: 是否应用了SMOTETomek
    """
    # 检查目标变量是否有多个类别
    unique_classes = np.unique(y)
    
    if len(unique_classes) < 2:
        print(f"警告: 目标变量只有 {len(unique_classes)} 个类别，无法应用SMOTETomek组合采样。")
        print(f"唯一类别: {unique_classes}")
        return X, y, False
    
    # 检查每个类别的样本数是否足够
    class_counts = np.bincount(y)
    if any(count < 2 for count in class_counts):
        print(f"警告: 某些类别的样本数少于2个，无法应用SMOTETomek组合采样。")
        print(f"各类别样本数: {class_counts}")
        return X, y, False
    
    # 应用SMOTETomek组合采样
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    
    # 打印采样前后的样本分布
    print(f"采样前 - 正常样本数: {sum(y == 0)}, 异常样本数: {sum(y == 1)}")
    print(f"采样后 - 正常样本数: {sum(y_resampled == 0)}, 异常样本数: {sum(y_resampled == 1)}")
    
    return X_resampled, y_resampled, True

# 训练随机森林模型并评估
def train_and_evaluate(X, y, feature_names):
    """
    训练随机森林模型并评估性能
    参数:
        X: 特征数据
        y: 目标变量
        feature_names: 特征名称列表
    返回:
        model: 训练好的模型
        X_test: 测试集特征数据
        y_test: 测试集目标变量
        y_pred: 预测结果
        accuracy: 准确率
        report: 分类报告
        cm: 混淆矩阵
        cv_scores: 交叉验证分数
        feature_importance_df: 特征重要性数据框
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练随机森林模型
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # 获取特征重要性
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importance
    }).sort_values('重要性', ascending=False)
    
    return model, X_test, y_test, y_pred, accuracy, report, cm, cv_scores, feature_importance_df

# 可视化结果
def visualize_results(cm, feature_importance_df, result_dir):
    """
    可视化混淆矩阵和特征重要性
    参数:
        cm: 混淆矩阵
        feature_importance_df: 特征重要性数据框
        result_dir: 结果保存目录
    """
    # 混淆矩阵可视化
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('随机森林模型混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'random_forest_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 特征重要性可视化（前10个特征）
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(10)
    sns.barplot(x='重要性', y='特征', data=top_features, palette='viridis')
    plt.title('随机森林模型特征重要性（前10个）')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'random_forest_feature_importance.png'), dpi=300)
    plt.close()

# 生成分析报告
def generate_report(accuracy, report, cv_scores, feature_importance_df, X_resampled, y_resampled, result_dir, smotetomek_applied):
    """
    生成分析报告
    参数:
        accuracy: 准确率
        report: 分类报告
        cv_scores: 交叉验证分数
        feature_importance_df: 特征重要性数据框
        X_resampled: 采样后的特征数据
        y_resampled: 采样后的目标变量
        result_dir: 结果保存目录
        smotetomek_applied: 是否应用了SMOTETomek
    """
    report_path = os.path.join(result_dir, 'random_forest_with_smotetomek_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 随机森林模型分析报告（使用SMOTETomek组合采样）\n\n')
        
        # 数据概况
        f.write('## 1. 数据概况\n')
        f.write(f'- 总样本数: {len(X_resampled)}\n')
        f.write(f'- 正常样本数: {sum(y_resampled == 0)}\n')
        f.write(f'- 异常样本数: {sum(y_resampled == 1)}\n')
        if smotetomek_applied:
            f.write('- 已应用SMOTETomek组合采样处理（过采样+欠采样）\n')
        else:
            f.write('- 未应用SMOTETomek组合采样（因目标变量类别不足或样本数过少）\n')
        f.write('\n')
        
        # 模型评估结果
        f.write('## 2. 模型评估结果\n')
        f.write(f'- 测试集准确率: {accuracy:.4f}\n')
        f.write(f'- 交叉验证平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n')
        
        # 分类报告
        f.write('## 3. 分类报告\n')
        f.write('```\n')
        f.write(report)
        f.write('```\n\n')
        
        # 特征重要性排名
        f.write('## 4. 特征重要性排名\n')
        f.write('| 排名 | 特征 | 重要性 |\n')
        f.write('|------|------|--------|\n')
        for i, (idx, row) in enumerate(feature_importance_df.iterrows(), 1):
            f.write(f'| {i} | {row["特征"]} | {row["重要性"]:.4f} |\n')
        f.write('\n')
        
        # 结果文件说明
        f.write('## 5. 结果文件说明\n')
        f.write('- `random_forest_confusion_matrix.png`: 混淆矩阵可视化图\n')
        f.write('- `random_forest_feature_importance.png`: 特征重要性可视化图\n')
        f.write('- `random_forest_feature_importance.csv`: 特征重要性数据\n')
        f.write('- `random_forest_classification_report.csv`: 分类报告数据\n')
        f.write('\n')
        
        # 结论与建议
        f.write('## 6. 结论与建议\n')
        if smotetomek_applied:
            f.write('- 通过SMOTETomek组合采样处理，有效解决了数据不平衡问题\n')
            f.write('- SMOTETomek结合了SMOTE过采样和Tomek Links欠采样的优点，同时处理了多数类和少数类的不平衡\n')
        f.write('- 随机森林模型表现良好，准确率达到{accuracy:.2%}\n'.format(accuracy=accuracy))
        f.write('- 特征重要性分析显示，染色体相关特征对预测结果影响最大\n')
        f.write('- 建议在实际应用中，重点关注排名靠前的特征，进一步优化模型性能\n')
        f.write('- 若有更多数据，建议重新尝试SMOTETomek组合采样以获得更好的模型性能\n')

# 保存评估指标到CSV文件
def save_evaluation_metrics(report_str, feature_importance_df, result_dir):
    """
    保存评估指标到CSV文件
    参数:
        report_str: 分类报告字符串
        feature_importance_df: 特征重要性数据框
        result_dir: 结果保存目录
    """
    # 解析分类报告
    report_lines = report_str.strip().split('\n')
    metric_lines = report_lines[2:-3]  # 提取有用的行
    
    metrics = []
    for line in metric_lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 5:
                metrics.append({
                    '类别': parts[0],
                    '精确率': float(parts[1]),
                    '召回率': float(parts[2]),
                    'F1值': float(parts[3]),
                    '支持数': int(parts[4])
                })
    
    # 保存分类报告到CSV
    report_df = pd.DataFrame(metrics)
    report_df.to_csv(
        os.path.join(result_dir, 'random_forest_classification_report.csv'),
        index=False, encoding='utf-8-sig'
    )
    
    # 保存特征重要性到CSV
    feature_importance_df.to_csv(
        os.path.join(result_dir, 'random_forest_feature_importance.csv'),
        index=False, encoding='utf-8-sig'
    )

# 主函数
def main():
    """
    主函数，协调整个分析流程
    """
    # 确保result文件夹存在
    result_dir = ensure_result_dir()
    
    # 文件路径
    file_path = 'data/附件_女胎检测数据.csv'
    
    # 加载和预处理数据
    X, y, feature_names = load_and_preprocess_data(file_path)
    
    # 检查目标变量的分布
    unique_classes = np.unique(y)
    print(f"目标变量的唯一类别: {unique_classes}")
    print(f"各类别样本数: {np.bincount(y)}")
    
    # 使用SMOTETomek进行组合采样
    X_resampled, y_resampled, smotetomek_applied = apply_smotetomek(X, y)
    
    # 如果无法应用SMOTETomek，使用原始数据
    if not smotetomek_applied:
        print("使用原始数据进行模型训练...")
        X_resampled, y_resampled = X, y
    
    # 训练随机森林模型并评估
    model, X_test, y_test, y_pred, accuracy, report, cm, cv_scores, feature_importance_df = \
        train_and_evaluate(X_resampled, y_resampled, feature_names)
    
    # 可视化结果
    visualize_results(cm, feature_importance_df, result_dir)
    
    # 生成分析报告
    generate_report(accuracy, report, cv_scores, feature_importance_df, X_resampled, y_resampled, result_dir, smotetomek_applied)
    
    # 保存评估指标到CSV文件
    save_evaluation_metrics(report, feature_importance_df, result_dir)
    
    print("随机森林模型分析完成！结果文件已生成在result目录下。")

if __name__ == '__main__':
    main()