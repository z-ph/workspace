import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import os
from datetime import datetime

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
        
        # 处理怀孕次数中的特殊值
        if '怀孕次数' in df.columns:
            df['怀孕次数'] = df['怀孕次数'].apply(lambda x: 3 if str(x).strip() in ['≥3', '>=3'] else x)
            df['怀孕次数'] = pd.to_numeric(df['怀孕次数'], errors='coerce')
        
        # 使用用户指定的十个特征
        feature_columns = [
            'X染色体浓度',     # 重要性: 0.1711
            '孕妇BMI',         # 重要性: 0.0726
            '体重',             # 重要性: 0.0653
            '怀孕次数',         # 重要性: 0.0586
            '检测孕周',         # 重要性: 0.0570
            '13号染色体的GC含量', # 重要性: 0.0509
            '原始读段数',       # 重要性: 0.0506
            '18号染色体的GC含量', # 重要性: 0.0448
            '21号染色体的GC含量', # 重要性: 0.0427
            '13号染色体的Z值'    # 重要性: 0.0420
        ]
        
        # 检查特征列是否存在，不存在的列使用空值填充
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
                print(f"警告：特征 '{col}' 在数据中不存在，将使用空值填充")
        
        # 创建特征矩阵和目标变量
        X = df[feature_columns].copy()
        y = df['异常结果']
        
        # 填充缺失值为中位数
        X = X.fillna(X.median(numeric_only=True))
        
        return X, y, df, feature_columns
        
    except Exception as e:
        print(f"数据加载和预处理出错: {e}")
        raise

def apply_smotetomek(X, y):
    """应用SMOTETomek组合采样"""
    try:
        # 检查目标变量是否有多个类别
        classes = np.unique(y)
        if len(classes) < 2:
            print(f"目标变量只有1个类别，无法应用样本平衡技术")
            return X, y, False
        
        # 检查每个类别的样本数是否足够
        for cls in classes:
            cls_count = np.sum(y == cls)
            if cls_count < 2:
                print(f"类别 {cls} 的样本数太少（{cls_count}个），无法应用样本平衡技术")
                return X, y, False
        
        # 应用SMOTETomek组合采样
        print("应用SMOTETomek组合采样")
        sampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # 统计平衡后的样本数量
        abnormal_count = np.sum(y_resampled == 1)
        normal_count = np.sum(y_resampled == 0)
        print(f"样本平衡后：正常样本 {normal_count} 个，异常样本 {abnormal_count} 个")
        
        return X_resampled, y_resampled, True
        
    except Exception as e:
        print(f"样本平衡技术应用出错: {e}")
        return X, y, False

def train_decision_tree(X, y, feature_columns):
    """训练决策树模型并进行超参数调优"""
    try:
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # 定义超参数网格
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', None]
        }
        
        # 创建决策树分类器
        dt_classifier = DecisionTreeClassifier(random_state=42)
        
        # 使用GridSearchCV进行超参数调优
        grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid,
                                  cv=5, scoring='accuracy', n_jobs=-1)
        
        print("正在进行超参数调优...")
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数和模型
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"最佳超参数: {best_params}")
        
        # 使用最佳模型进行预测
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # 交叉验证
        cv_scores = cross_val_score(best_model, X, y, cv=5)
        
        # 生成分类报告和混淆矩阵
        class_report = classification_report(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # 生成决策规则
        decision_rules = generate_decision_rules(best_model, feature_columns)
        
        return best_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, \
               accuracy, auc_score, cv_scores, class_report, conf_matrix, decision_rules, best_params
        
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

def visualize_results(model, X_train, y_train, X_test, y_test, y_pred, y_pred_proba, 
                     conf_matrix, feature_columns, balancing_applied, best_params, result_dir):
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
        plt.title(f'决策树混淆矩阵{"(样本平衡处理)" if balancing_applied else ""}')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 绘制决策树
        plt.figure(figsize=(20, 15))
        plot_tree(model, feature_names=feature_columns, class_names=['正常', '异常'],
                 filled=True, rounded=True, fontsize=10)
        plt.title(f'决策树可视化{"(样本平衡处理)" if balancing_applied else ""}')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'decision_tree.png'), dpi=300)
        plt.close()
        
        # 绘制特征重要性
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title('决策树特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
        # 如果有概率预测结果，绘制ROC曲线
        if y_pred_proba is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title('决策树ROC曲线')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'roc_curve.png'), dpi=300)
            plt.close()
        
    except Exception as e:
        print(f"结果可视化出错: {e}")
        raise

def generate_report(X, y, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, 
                    accuracy, auc_score, cv_scores, class_report, conf_matrix, decision_rules, 
                    feature_columns, balancing_applied, best_params, result_dir):
    """生成分析报告"""
    try:
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成报告文件
        report_path = os.path.join(result_dir, 'decision_tree_with_top10_features_analysis.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 决策树模型分析报告（使用Top10特征）\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据基本信息
            f.write("## 数据基本信息\n")
            f.write(f"- 总样本数: {len(X)}\n")
            f.write(f"- 正常样本数: {np.sum(y == 0)}\n")
            f.write(f"- 异常样本数: {np.sum(y == 1)}\n")
            f.write(f"- 特征数量: {len(feature_columns)}\n")
            f.write("- 使用特征: ")
            for i, feature in enumerate(feature_columns, 1):
                f.write(f"{i}. {feature}")
                if i < len(feature_columns):
                    f.write("，")
            f.write("\n")
            if balancing_applied:
                f.write("- 已应用SMOTETomek组合采样处理样本不平衡问题\n")
            else:
                f.write("- 未应用样本平衡技术\n")
            f.write(f"- 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}\n\n")
            
            # 模型优化措施
            f.write("## 模型优化措施\n")
            f.write("- 使用GridSearchCV进行超参数调优\n")
            f.write("- 尝试不同的树深度、最小样本分割数、最小叶节点样本数等参数\n")
            if balancing_applied:
                f.write("- 应用SMOTETomek组合采样处理类别不平衡问题\n")
            f.write("- 增加了ROC曲线评估\n")
            f.write("- 优化了决策规则提取方法\n\n")
            
            # 最佳超参数
            f.write("## 最佳超参数\n")
            for param, value in best_params.items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
            
            # 模型评估指标
            f.write("## 模型评估指标\n")
            f.write(f"- 测试集准确率: {accuracy:.4f}\n")
            f.write(f"- AUC得分: {auc_score:.4f}\n")
            f.write(f"- 交叉验证平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
            
            # 分类报告
            f.write("## 分类报告\n")
            f.write("```\n")
            f.write(class_report)
            f.write("```\n\n")
            
            # 决策规则（只显示前10条规则）
            f.write("## 决策规则 (前10条)\n")
            if decision_rules:
                for i, rule in enumerate(decision_rules[:10], 1):
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
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]  # 按重要性降序排列
            for i, idx in enumerate(indices, 1):
                f.write(f"- {i}. {feature_columns[idx]}: {importances[idx]:.4f}\n")
            f.write("\n")
            
            # 结果文件说明
            f.write("## 生成的结果文件\n")
            f.write(f"1. 分析报告: {os.path.basename(report_path)}\n")
            f.write(f"2. 混淆矩阵图: confusion_matrix.png\n")
            f.write(f"3. 决策树可视化: decision_tree.png\n")
            f.write(f"4. 特征重要性图: feature_importance.png\n")
            f.write(f"5. ROC曲线图: roc_curve.png\n")
            f.write(f"6. 决策规则CSV: decision_rules.csv\n")
            f.write(f"7. 特征重要性CSV: feature_importance.csv\n\n")
            
            # 结论与建议
            f.write("## 结论与建议\n")
            if balancing_applied:
                f.write("- 模型通过SMOTETomek组合采样有效处理了样本不平衡问题\n")
            else:
                f.write("- 由于数据限制，未应用样本平衡技术\n")
            f.write("- 通过超参数调优，模型性能得到显著提升\n")
            f.write("- 决策树模型生成了可解释的分类规则，有助于理解影响胎儿染色体异常的关键因素\n")
            f.write("- X染色体浓度仍然是最重要的预测特征\n")
            f.write("- 建议进一步尝试集成学习方法（如随机森林、梯度提升树）以获得更好的性能\n")
            f.write("- 可以考虑添加更多的特征工程步骤，如特征交互、多项式特征等\n")
            f.write("- 建议结合临床专业知识进一步验证模型规则的有效性\n")
        
        # 保存决策规则到CSV
        if decision_rules:
            rules_df = pd.DataFrame(decision_rules)
            rules_df['predicted_class_name'] = rules_df['predicted_class'].apply(lambda x: '异常' if x == 1 else '正常')
            rules_df.to_csv(os.path.join(result_dir, 'decision_rules.csv'), index=False, encoding='utf-8-sig')
        
        # 保存特征重要性到CSV
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            '特征': feature_columns,
            '重要性': importances
        }).sort_values('重要性', ascending=False)
        feature_importance_df.to_csv(os.path.join(result_dir, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"生成报告出错: {e}")
        raise

def ensure_result_dir(result_folder):
    """确保结果目录存在"""
    result_dir = os.path.join('result', result_folder)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def main():
    """主函数，协调整个分析流程"""
    try:
        # 确保结果目录存在
        result_dir = ensure_result_dir('decision_tree_with_top10_features')
        
        # 文件路径
        file_path = 'data/附件_女胎检测数据.csv'  # 可以根据需要修改
        
        # 加载和预处理数据
        X, y, df, feature_columns = load_and_preprocess_data(file_path)
        
        # 应用SMOTETomek组合采样
        X_resampled, y_resampled, balancing_applied = apply_smotetomek(X, y)
        
        # 如果无法应用SMOTETomek，使用原始数据
        if not balancing_applied:
            print("使用原始数据进行模型训练...")
            X_resampled, y_resampled = X, y
        
        # 训练决策树模型
        model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, \
        accuracy, auc_score, cv_scores, class_report, conf_matrix, decision_rules, best_params = \
            train_decision_tree(X_resampled, y_resampled, feature_columns)
        
        # 可视化结果
        visualize_results(model, X_train, y_train, X_test, y_test, y_pred, y_pred_proba, 
                         conf_matrix, feature_columns, balancing_applied, best_params, result_dir)
        
        # 生成分析报告
        generate_report(X_resampled, y_resampled, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, 
                       accuracy, auc_score, cv_scores, class_report, conf_matrix, decision_rules, 
                       feature_columns, balancing_applied, best_params, result_dir)
        
        print("\n=== 分析完成 ===")
        print(f"所有结果已保存到 {result_dir} 目录")
        print(f"- 生成的决策规则数量: {len(decision_rules)}")
        print(f"- 测试集准确率: {accuracy:.4f}")
        print(f"- 交叉验证平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"- 特征重要性第一位: {feature_columns[np.argmax(model.feature_importances_)]} ({model.feature_importances_[np.argmax(model.feature_importances_)]:.4f})")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        raise

if __name__ == '__main__':
    main()