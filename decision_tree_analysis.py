import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import os
from datetime import datetime
from sklearn import tree

# 设置中文字体和负号显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    try:
        # 加载CSV文件
        df = pd.read_csv(file_path)
        print(f"原始数据形状: {df.shape}")
        
        # 创建目标变量：异常样本（染色体的非整倍体有值）和正常样本（染色体的非整倍体为空）
        df['异常结果'] = df['染色体的非整倍体'].apply(lambda x: 1 if pd.notna(x) else 0)
        
        # 统计各类别样本数量
        abnormal_count = df['异常结果'].sum()
        normal_count = len(df) - abnormal_count
        print(f"原始数据：正常样本 {normal_count} 个，异常样本 {abnormal_count} 个")
        
        # 处理特殊格式的孕周
        def parse_gestational_week(week_str):
            if pd.isna(week_str):
                return np.nan
            try:
                # 处理类似 "13w+5" 的格式
                if isinstance(week_str, str) and 'w' in week_str:
                    parts = week_str.lower().split('w')
                    weeks = float(parts[0])
                    if len(parts) > 1 and '+' in parts[1]:
                        days = float(parts[1].split('+')[1]) / 7
                        weeks += days
                    return weeks
                # 尝试直接转换为数字
                return float(week_str)
            except:
                return np.nan
        
        # 应用孕周解析函数
        if '检测孕周' in df.columns:
            df['检测孕周'] = df['检测孕周'].apply(parse_gestational_week)
        
        # 处理IVF妊娠列
        if 'IVF妊娠' in df.columns:
            df['IVF妊娠'] = df['IVF妊娠'].apply(lambda x: 1 if str(x).strip() == 'IVF（试管婴儿）' else 0)
        
        # 处理怀孕次数和生产次数中的特殊值
        for col in ['怀孕次数', '生产次数']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 3 if str(x).strip() in ['≥3', '>=3'] else x)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 使用前五个重要特征（根据之前分析的结果）
        # X染色体浓度、孕妇BMI、13号染色体的GC含量、检测孕周、体重
        feature_columns = ['X染色体浓度', '孕妇BMI', '13号染色体的GC含量', '检测孕周', '体重']
        
        # 检查特征列是否存在，不存在的列使用空值填充
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # 创建特征矩阵和目标变量
        X = df[feature_columns].copy()
        y = df['异常结果']
        
        # 填充缺失值为中位数
        X = X.fillna(X.median(numeric_only=True))
        
        return X, y, df, feature_columns
        
    except Exception as e:
        print(f"数据加载和预处理出错: {e}")
        raise

def apply_smote(X, y):
    """应用SMOTE算法处理样本不平衡问题"""
    try:
        # 检查目标变量是否有多个类别
        classes = np.unique(y)
        if len(classes) < 2:
            print(f"目标变量只有1个类别，无法应用SMOTE")
            return X, y, False
        
        # 检查每个类别的样本数是否足够
        for cls in classes:
            cls_count = np.sum(y == cls)
            if cls_count < 2:
                print(f"类别 {cls} 的样本数太少（{cls_count}个），无法应用SMOTE")
                return X, y, False
        
        # 应用SMOTE过采样
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # 统计过采样后的样本数量
        abnormal_count = np.sum(y_resampled == 1)
        normal_count = np.sum(y_resampled == 0)
        print(f"SMOTE过采样后：正常样本 {normal_count} 个，异常样本 {abnormal_count} 个")
        
        return X_resampled, y_resampled, True
        
    except Exception as e:
        print(f"SMOTE应用出错: {e}")
        return X, y, False

def train_decision_tree(X, y, feature_columns):
    """训练决策树模型并生成决策规则"""
    try:
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # 训练决策树模型
        dt_classifier = DecisionTreeClassifier(
            max_depth=5,  # 限制树的深度，避免过拟合
            random_state=42,
            class_weight='balanced' if len(np.unique(y)) > 1 else None
        )
        dt_classifier.fit(X_train, y_train)
        
        # 生成决策规则
        decision_rules = generate_decision_rules(dt_classifier, feature_columns)
        
        # 模型评估
        y_pred = dt_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
        
        # 生成分类报告和混淆矩阵
        class_report = classification_report(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return dt_classifier, X_train, X_test, y_train, y_test, y_pred, \
               accuracy, cv_scores, class_report, conf_matrix, decision_rules
        
    except Exception as e:
        print(f"决策树训练出错: {e}")
        raise

def generate_decision_rules(tree, feature_names):
    """从决策树中提取决策规则"""
    try:
        # 获取树的内部结构
        tree_ = tree.tree_
        
        # 获取特征名称列表，处理未定义的特征
        feature_name = []
        for i in tree_.feature:
            if i != _tree.TREE_UNDEFINED:
                feature_name.append(feature_names[i])
            else:
                feature_name.append("undefined!")
        
        rules = []
        
        def recurse(node, depth, rule):
            """递归遍历决策树，提取规则"""
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # 非叶子节点，继续递归
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # 左子树规则 (特征 <= 阈值)
                left_rule = rule + [f"{name} <= {threshold:.4f}"]
                recurse(tree_.children_left[node], depth + 1, left_rule)
                
                # 右子树规则 (特征 > 阈值)
                right_rule = rule + [f"{name} > {threshold:.4f}"]
                recurse(tree_.children_right[node], depth + 1, right_rule)
            else:
                # 叶子节点，生成规则
                value = tree_.value[node][0]
                prob = value / value.sum() if value.sum() > 0 else [0] * len(value)
                class_idx = np.argmax(prob)
                confidence = prob[class_idx]
                
                rules.append({
                    'rule': ' AND '.join(rule) if rule else '无条件规则',
                    'predicted_class': class_idx,
                    'confidence': confidence,
                    'samples': int(value.sum())
                })
        
        # 从根节点开始递归
        recurse(0, 1, [])
        
        # 对规则按置信度排序
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        return rules
        
    except Exception as e:
        print(f"生成决策规则出错: {str(e)}")
        # 返回一个默认规则作为备用
        return [{
            'rule': '无法生成规则 (模型可能太简单)',
            'predicted_class': 0,
            'confidence': 0.5,
            'samples': len(tree.tree_.value[0][0]) if hasattr(tree.tree_, 'value') and len(tree.tree_.value) > 0 else 0
        }]

def visualize_results(dt_classifier, X_train, y_train, X_test, y_test, y_pred, 
                     conf_matrix, feature_columns, smote_applied, result_dir):
    """可视化分析结果"""
    try:
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
        plt.xlabel('预测类别')
        plt.ylabel('实际类别')
        plt.title(f'决策树混淆矩阵{"(SMOTE过采样)" if smote_applied else ""}')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 绘制决策树
        plt.figure(figsize=(20, 15))
        plot_tree(dt_classifier, feature_names=feature_columns, class_names=['正常', '异常'],
                 filled=True, rounded=True, fontsize=10)
        plt.title(f'决策树可视化{"(SMOTE过采样)" if smote_applied else ""}')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'decision_tree.png'), dpi=300)
        plt.close()
        
        # 绘制特征重要性
        importances = dt_classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=45)
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title('决策树特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"结果可视化出错: {e}")
        raise

def generate_report(X, y, dt_classifier, X_train, X_test, y_train, y_test, y_pred, 
                    accuracy, cv_scores, class_report, conf_matrix, decision_rules, 
                    feature_columns, smote_applied, result_dir):
    """生成分析报告"""
    try:
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成报告文件
        report_path = os.path.join(result_dir, 'decision_tree_analysis.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 决策树模型分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据基本信息
            f.write("## 数据基本信息\n")
            f.write(f"- 总样本数: {len(X)}\n")
            f.write(f"- 正常样本数: {np.sum(y == 0)}\n")
            f.write(f"- 异常样本数: {np.sum(y == 1)}\n")
            f.write(f"- 特征数量: {len(feature_columns)}\n")
            f.write(f"- 使用特征: {', '.join(feature_columns)}\n")
            if smote_applied:
                f.write("- 已应用SMOTE过采样处理样本不平衡问题\n")
            else:
                f.write("- 未应用SMOTE过采样\n")
            f.write(f"- 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}\n\n")
            
            # 模型评估指标
            f.write("## 模型评估指标\n")
            f.write(f"- 测试集准确率: {accuracy:.4f}\n")
            f.write(f"- 交叉验证平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
            
            # 分类报告
            f.write("## 分类报告\n")
            f.write("```\n")
            f.write(class_report)
            f.write("```\n\n")
            
            # 决策规则
            f.write("## 决策规则\n")
            if decision_rules:
                for i, rule in enumerate(decision_rules, 1):
                    class_name = '异常' if rule['predicted_class'] == 1 else '正常'
                    f.write(f"### 规则 {i}\n")
                    f.write(f"- 条件: {rule['rule']}\n")
                    f.write(f"- 预测结果: {class_name}\n")
                    f.write(f"- 置信度: {rule['confidence']:.4f}\n")
                    f.write(f"- 覆盖样本数: {rule['samples']}\n\n")
            else:
                f.write("未生成决策规则\n\n")
            
            # 特征重要性
            f.write("## 特征重要性\n")
            importances = dt_classifier.feature_importances_
            for i, (feature, importance) in enumerate(zip(feature_columns, importances)):
                f.write(f"- {i+1}. {feature}: {importance:.4f}\n")
            f.write("\n")
            
            # 结果文件说明
            f.write("## 生成的结果文件\n")
            f.write(f"1. 分析报告: {os.path.basename(report_path)}\n")
            f.write(f"2. 混淆矩阵图: confusion_matrix.png\n")
            f.write(f"3. 决策树可视化: decision_tree.png\n")
            f.write(f"4. 特征重要性图: feature_importance.png\n")
            f.write(f"5. 决策规则CSV: decision_rules.csv\n")
            f.write(f"6. 特征重要性CSV: feature_importance.csv\n\n")
            
            # 结论与建议
            f.write("## 结论与建议\n")
            if smote_applied:
                f.write("- 模型通过SMOTE过采样有效处理了样本不平衡问题\n")
            else:
                f.write("- 由于数据限制，未应用SMOTE过采样\n")
            f.write("- 决策树模型生成了可解释的分类规则，有助于理解影响胎儿染色体异常的关键因素\n")
            f.write("- 前五个重要特征对预测结果贡献最大，可作为重点关注指标\n")
            f.write("- 建议结合临床专业知识进一步验证模型规则的有效性\n")
        
        # 保存决策规则到CSV
        if decision_rules:
            rules_df = pd.DataFrame(decision_rules)
            rules_df['predicted_class_name'] = rules_df['predicted_class'].apply(lambda x: '异常' if x == 1 else '正常')
            rules_df.to_csv(os.path.join(result_dir, 'decision_rules.csv'), index=False, encoding='utf-8-sig')
        
        # 保存特征重要性到CSV
        importances_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': dt_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        importances_df.to_csv(os.path.join(result_dir, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
        
        print(f"分析报告已保存到: {report_path}")
        
    except Exception as e:
        print(f"生成报告出错: {e}")
        raise

def main():
    """主函数"""
    try:
        # 数据文件路径
        data_file = "data/附件_女胎检测数据.csv"
        
        # 结果目录
        result_dir = "result/decision_tree_analysis"
        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)
        
        # 加载和预处理数据
        print("=== 加载和预处理数据 ===")
        X, y, df, feature_columns = load_and_preprocess_data(data_file)
        
        # 应用SMOTE过采样
        print("\n=== 应用SMOTE过采样 ===")
        X_resampled, y_resampled, smote_applied = apply_smote(X, y)
        
        # 训练决策树模型
        print("\n=== 训练决策树模型 ===")
        dt_classifier, X_train, X_test, y_train, y_test, y_pred, \
        accuracy, cv_scores, class_report, conf_matrix, decision_rules = \
            train_decision_tree(X_resampled, y_resampled, feature_columns)
        
        # 可视化结果
        print("\n=== 可视化分析结果 ===")
        visualize_results(dt_classifier, X_train, y_train, X_test, y_test, y_pred, \
                         conf_matrix, feature_columns, smote_applied, result_dir)
        
        # 生成分析报告
        print("\n=== 生成分析报告 ===")
        generate_report(X_resampled, y_resampled, dt_classifier, X_train, X_test, y_train, y_test, y_pred, \
                       accuracy, cv_scores, class_report, conf_matrix, decision_rules, \
                       feature_columns, smote_applied, result_dir)
        
        print("\n=== 分析完成 ===")
        print(f"所有结果已保存到 {result_dir} 目录")
        print(f"- 生成的决策规则数量: {len(decision_rules)}")
        print(f"- 测试集准确率: {accuracy:.4f}")
        print(f"- 特征重要性第一位: {feature_columns[np.argmax(dt_classifier.feature_importances_)]} ({np.max(dt_classifier.feature_importances_):.4f})" )
        
    except Exception as e:
        print(f"分析过程出错: {e}")

if __name__ == "__main__":
    main()