import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
import os
import json

# 设置中文字体和负号显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class BMIAnalyzer:
    """分析BMI对Y染色体浓度达标时间影响的类"""
    
    def __init__(self, file_path):
        """初始化分析器，加载数据"""
        self.file_path = file_path
        self.data = None
        self.output_dir = "c:\\Users\\30513\\Desktop\\workspace\\result\\bmi_y_chromosome_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """加载CSV数据并进行基本预处理"""
        self.data = pd.read_csv(self.file_path, encoding='utf-8-sig')
        
        # 预处理：将怀孕次数中的'≥3'转换为3
        self.data['怀孕次数'] = self.data['怀孕次数'].replace('≥3', 3)
        
        # 将'生产次数'转换为数值型
        self.data['生产次数'] = pd.to_numeric(self.data['生产次数'], errors='coerce')
        
        # 为每位孕妇记录达到4%浓度的最早时间
        self._identify_reach_4_percent_time()
    
    def _identify_reach_4_percent_time(self):
        """识别每位孕妇Y染色体浓度达到或超过4%的最早时间"""
        # 按孕妇代码和检测孕天排序
        self.data = self.data.sort_values(['孕妇代码', '检测孕天'])
        
        # 标识是否达到4%浓度
        self.data['达到4%浓度'] = self.data['Y染色体浓度'] >= 0.04
        
        # 为每位孕妇找出最早达到4%浓度的时间
        first_reach_time = self.data.groupby('孕妇代码').apply(
            lambda x: x.loc[x['达到4%浓度'], '检测孕天'].min() if x['达到4%浓度'].any() else np.nan
        ).reset_index(name='最早达标时间')
        
        # 合并回原数据
        self.data = self.data.merge(first_reach_time, on='孕妇代码', how='left')
        
        # 标记是否在当前检测时已达标
        self.data['当前是否已达标'] = self.data.apply(
            lambda row: row['检测孕天'] >= row['最早达标时间'] if pd.notna(row['最早达标时间']) else False,
            axis=1
        )
        
        # 找出最终是否达标
        final_reach_status = self.data.groupby('孕妇代码').apply(
            lambda x: x['达到4%浓度'].any()
        ).reset_index(name='最终是否达标')
        
        self.data = self.data.merge(final_reach_status, on='孕妇代码', how='left')
    
    def analyze_data_structure(self):
        """分析数据结构和类型"""
        data_info = {
            '记录总数': len(self.data),
            '孕妇数量': self.data['孕妇代码'].nunique(),
            '列总数': len(self.data.columns),
            '数值型列': [],
            '分类型列': [],
            '日期型列': [],
            '空值情况': self.data.isnull().sum().to_dict()
        }
        
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                data_info['数值型列'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]) or '日期' in col:
                data_info['日期型列'].append(col)
            else:
                data_info['分类型列'].append(col)
        
        return data_info
    
    def create_data_dictionary(self):
        """创建数据字典"""
        # 基于列名和数据内容推断每列的含义
        data_dict = {
            '序号': '数据记录的序号',
            '孕妇代码': '每位孕妇的唯一标识符',
            '年龄': '孕妇的年龄（岁）',
            '身高': '孕妇的身高（cm）',
            '体重': '孕妇的体重（kg）',
            '末次月经': '孕妇最后一次月经的日期',
            'IVF妊娠': '是否为体外受精妊娠',
            '检测日期': '进行检测的日期',
            '检测抽血次数': '该孕妇本次是第几次抽血检测',
            '检测孕周': '检测时的孕周，格式为"周数w+天数"',
            '孕妇BMI': '孕妇的体重指数，计算公式：体重(kg)/身高(m)²',
            '原始读段数': '测序获得的原始DNA片段数量',
            '在参考基因组上比对的比例': '能够比对到人类参考基因组的DNA片段比例',
            '重复读段的比例': '测序中重复出现的DNA片段比例',
            '唯一比对的读段数': '能够唯一比对到基因组特定位置的DNA片段数量',
            'GC含量': '测序读段中的GC碱基对比例',
            '13号染色体的Z值': '13号染色体剂量的Z评分',
            '18号染色体的Z值': '18号染色体剂量的Z评分',
            '21号染色体的Z值': '21号染色体剂量的Z评分',
            'X染色体的Z值': 'X染色体剂量的Z评分',
            'Y染色体的Z值': 'Y染色体剂量的Z评分',
            'Y染色体浓度': '母血中胎儿Y染色体的浓度比例',
            'X染色体浓度': '母血中X染色体的浓度比例',
            '13号染色体的GC含量': '13号染色体上的GC碱基对比例',
            '18号染色体的GC含量': '18号染色体上的GC碱基对比例',
            '21号染色体的GC含量': '21号染色体上的GC碱基对比例',
            '被过滤掉读段数的比例': '质量控制过程中被过滤掉的DNA片段比例',
            '染色体的非整倍体': '检测到的染色体非整倍体情况（如T13、T18、T21等）',
            '怀孕次数': '孕妇的怀孕次数',
            '生产次数': '孕妇的生产次数',
            '胎儿是否健康': '胎儿是否健康',
            '检测孕天': '检测时的怀孕天数'
        }
        
        return data_dict
    
    def analyze_bmi_distribution(self):
        """分析BMI的分布情况"""
        # 计算BMI的基本统计量
        bmi_stats = self.data['孕妇BMI'].describe()
        
        # 可视化BMI分布
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['孕妇BMI'], bins=20, kde=True)
        plt.title('孕妇BMI分布')
        plt.xlabel('BMI')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'bmi_distribution.png'), dpi=300)
        plt.close()
        
        return bmi_stats
    
    def analyze_y_chromosome_concentration(self):
        """分析Y染色体浓度的分布和特征"""
        # Y染色体浓度的基本统计量
        y_conc_stats = self.data['Y染色体浓度'].describe()
        
        # 可视化Y染色体浓度分布
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Y染色体浓度'], bins=30, kde=True)
        plt.axvline(x=0.04, color='red', linestyle='--', label='4%浓度阈值')
        plt.title('Y染色体浓度分布')
        plt.xlabel('Y染色体浓度')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'y_chromosome_concentration_distribution.png'), dpi=300)
        plt.close()
        
        # 达标情况统计
        reach_4_percent_stats = {
            '总达标率': self.data.groupby('孕妇代码')['达到4%浓度'].any().mean(),
            '未达标率': 1 - self.data.groupby('孕妇代码')['达到4%浓度'].any().mean()
        }
        
        return y_conc_stats, reach_4_percent_stats
    
    def analyze_bmi_vs_reach_time(self):
        """分析BMI与达标时间的关系"""
        # 筛选出有达标时间的孕妇数据
        has_reach_time = self.data.drop_duplicates('孕妇代码').dropna(subset=['最早达标时间'])
        
        # 相关性分析
        corr, p_value = stats.pearsonr(has_reach_time['孕妇BMI'], has_reach_time['最早达标时间'])
        
        # 可视化BMI与达标时间的关系
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='孕妇BMI', y='最早达标时间', data=has_reach_time, alpha=0.6)
        sns.regplot(x='孕妇BMI', y='最早达标时间', data=has_reach_time, scatter=False, color='red')
        plt.title(f'BMI与Y染色体浓度达标时间的关系 (r={corr:.3f}, p={p_value:.3f})')
        plt.xlabel('BMI')
        plt.ylabel('最早达标时间（天）')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'bmi_vs_reach_time.png'), dpi=300)
        plt.close()
        
        return {'相关系数': corr, 'p值': p_value}
    
    def perform_bmi_clustering(self, n_clusters=4):
        """使用K-means聚类对BMI进行分组"""
        # 准备数据
        bmi_data = self.data[['孕妇BMI']].dropna()
        
        # 标准化数据
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_bmi = scaler.fit_transform(bmi_data)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_bmi)
        
        # 添加聚类结果到数据中
        self.data['BMI聚类'] = clusters
        
        # 分析每个聚类的BMI范围和达标时间
        cluster_stats = self.data.groupby('BMI聚类').agg({
            '孕妇BMI': ['min', 'max', 'mean'],
            '最早达标时间': ['mean', 'std', 'count']
        }).round(2)
        
        # 可视化每个聚类的BMI范围和达标时间
        plt.figure(figsize=(12, 6))
        
        # BMI箱线图
        plt.subplot(1, 2, 1)
        sns.boxplot(x='BMI聚类', y='孕妇BMI', data=self.data)
        plt.title('各聚类的BMI分布')
        plt.xlabel('BMI聚类')
        plt.ylabel('BMI')
        
        # 达标时间箱线图
        plt.subplot(1, 2, 2)
        sns.boxplot(x='BMI聚类', y='最早达标时间', data=self.data.dropna(subset=['最早达标时间']))
        plt.title('各聚类的达标时间分布')
        plt.xlabel('BMI聚类')
        plt.ylabel('最早达标时间（天）')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bmi_clusters_analysis.png'), dpi=300)
        plt.close()
        
        return cluster_stats
    
    def determine_optimal_nipt_time(self):
        """根据BMI分组确定最佳NIPT检测时点"""
        # 首先进行基于医学标准的BMI分组
        def bmi_medical_group(bmi):
            if bmi < 18.5:
                return '偏瘦 (<18.5)'
            elif 18.5 <= bmi < 24:
                return '正常 (18.5-23.9)'
            elif 24 <= bmi < 28:
                return '超重 (24-27.9)'
            else:
                return '肥胖 (≥28)'
        
        self.data['BMI医学分组'] = self.data['孕妇BMI'].apply(bmi_medical_group)
        
        # 分析每组的达标情况
        group_analysis = self.data.groupby(['BMI医学分组']).apply(
            lambda x: pd.Series({
                '样本数': len(x['孕妇代码'].unique()),
                '平均达标时间': x.dropna(subset=['最早达标时间'])['最早达标时间'].mean(),
                '达标率': x.groupby('孕妇代码')['达到4%浓度'].any().mean(),
                '不同孕天的达标率': self._calculate_reach_rate_by_gestational_days(x)
            })
        )
        
        # 可视化每组的不同孕天达标率
        plt.figure(figsize=(12, 8))
        
        # 创建孕天范围
        gestational_days = range(70, 180, 10)
        
        # 为每个BMI组绘制达标率曲线
        for group in group_analysis.index:
            reach_rates = []
            for day in gestational_days:
                # 计算该组中在day天时已达标的孕妇比例
                group_data = self.data[self.data['BMI医学分组'] == group]
                reached_by_day = group_data.groupby('孕妇代码').apply(
                    lambda x: x[x['检测孕天'] <= day]['达到4%浓度'].any()
                ).mean()
                reach_rates.append(reached_by_day)
            
            plt.plot(gestational_days, reach_rates, marker='o', label=group)
        
        plt.axhline(y=0.95, color='red', linestyle='--', label='95%达标率阈值')
        plt.title('不同BMI分组在各孕天的Y染色体浓度达标率')
        plt.xlabel('检测孕天')
        plt.ylabel('达标率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'reach_rate_by_bmi_group.png'), dpi=300)
        plt.close()
        
        return group_analysis
    
    def _calculate_reach_rate_by_gestational_days(self, group_data):
        """计算某组数据在不同孕天的达标率"""
        # 以10天为间隔计算达标率
        result = {}
        for day in range(80, 180, 10):
            # 计算在day天内已达标的孕妇比例
            reached_by_day = group_data.groupby('孕妇代码').apply(
                lambda x: x[x['检测孕天'] <= day]['达到4%浓度'].any()
            ).mean()
            result[f'{day}天'] = reached_by_day
        
        return result
    
    def analyze_detection_error(self):
        """分析检测误差对结果的影响"""
        # 计算Y染色体浓度的变异系数作为误差指标
        cv_by_patient = self.data.groupby('孕妇代码')['Y染色体浓度'].apply(
            lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0
        )
        
        error_analysis = {
            '总体平均变异系数': cv_by_patient.mean(),
            '变异系数分布': cv_by_patient.describe()
        }
        
        # 分析BMI与检测误差的关系
        cv_df = cv_by_patient.reset_index(name='Y染色体浓度变异系数')
        merged_data = pd.merge(
            self.data.drop_duplicates('孕妇代码'), 
            cv_df, 
            on='孕妇代码', 
            how='left'
        )
        
        corr_cv_bmi, p_cv_bmi = stats.pearsonr(merged_data['孕妇BMI'], merged_data['Y染色体浓度变异系数'])
        
        # 可视化BMI与检测误差的关系
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='孕妇BMI', 
            y='Y染色体浓度变异系数', 
            data=merged_data, 
            alpha=0.6
        )
        sns.regplot(
            x='孕妇BMI', 
            y='Y染色体浓度变异系数', 
            data=merged_data, 
            scatter=False, 
            color='red'
        )
        plt.title(f'BMI与Y染色体浓度检测误差的关系 (r={corr_cv_bmi:.3f}, p={p_cv_bmi:.3f})')
        plt.xlabel('BMI')
        plt.ylabel('Y染色体浓度变异系数')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'bmi_vs_detection_error.png'), dpi=300)
        plt.close()
        
        error_analysis['BMI与误差相关性'] = {'相关系数': corr_cv_bmi, 'p值': p_cv_bmi}
        
        return error_analysis
    
    def generate_md_report(self):
        """生成完整的Markdown报告"""
        # 加载和分析数据
        self.load_data()
        data_structure = self.analyze_data_structure()
        data_dict = self.create_data_dictionary()
        bmi_stats = self.analyze_bmi_distribution()
        y_conc_stats, reach_stats = self.analyze_y_chromosome_concentration()
        bmi_reach_corr = self.analyze_bmi_vs_reach_time()
        cluster_stats = self.perform_bmi_clustering()
        optimal_time_analysis = self.determine_optimal_nipt_time()
        error_analysis = self.analyze_detection_error()
        
        # 创建Markdown内容
        md_content = "# 男胎孕妇BMI与Y染色体浓度达标时间分析报告\n\n"
        
        # 1. 数据概览
        md_content += "## 1. 数据概览\n\n"
        md_content += "| 指标 | 值 |\n"
        md_content += "|-----|-----|\n"
        md_content += f"| 总记录数 | {data_structure['记录总数']} |\n"
        md_content += f"| 孕妇数量 | {data_structure['孕妇数量']} |\n"
        md_content += f"| 列总数 | {data_structure['列总数']} |\n"
        md_content += f"| 数值型列数量 | {len(data_structure['数值型列'])} |\n"
        md_content += f"| 分类型列数量 | {len(data_structure['分类型列'])} |\n"
        md_content += f"| 日期型列数量 | {len(data_structure['日期型列'])} |\n\n"
        
        # 2. 数据字典
        md_content += "## 2. 数据字典\n\n"
        md_content += "| 列名 | 数据类型 | 含义 |\n"
        md_content += "|-----|---------|------|\n"
        
        for col in self.data.columns:
            if col in data_structure['数值型列']:
                col_type = '数值型'
            elif col in data_structure['分类型列']:
                col_type = '分类型'
            else:
                col_type = '日期型'
            
            # 处理新增的列
            if col not in data_dict:
                if col == '最早达标时间':
                    col_desc = 'Y染色体浓度达到或超过4%的最早检测孕天'
                elif col == '达到4%浓度':
                    col_desc = '本次检测Y染色体浓度是否达到或超过4%'
                elif col == '当前是否已达标':
                    col_desc = '在当前检测时是否已达到4%浓度'
                elif col == '最终是否达标':
                    col_desc = '该孕妇最终是否达到4%浓度'
                elif col == 'BMI聚类':
                    col_desc = '基于K-means聚类的BMI分组'
                elif col == 'BMI医学分组':
                    col_desc = '基于医学标准的BMI分组'
                else:
                    col_desc = '分析过程生成的列'
            else:
                col_desc = data_dict[col]
            
            md_content += f"| {col} | {col_type} | {col_desc} |\n"
        md_content += "\n"
        
        # 3. BMI分布分析
        md_content += "## 3. BMI分布分析\n\n"
        md_content += "### BMI统计摘要\n\n"
        md_content += "| 统计指标 | 值 |\n"
        md_content += "|---------|-----|\n"
        for stat_name, value in bmi_stats.items():
            md_content += f"| {stat_name} | {value:.2f} |\n"
        md_content += "\n"
        
        md_content += "### BMI分布图\n\n"
        md_content += "![BMI分布图](bmi_distribution.png)\n\n"
        
        # 4. Y染色体浓度分析
        md_content += "## 4. Y染色体浓度分析\n\n"
        md_content += "### Y染色体浓度统计摘要\n\n"
        md_content += "| 统计指标 | 值 |\n"
        md_content += "|---------|-----|\n"
        for stat_name, value in y_conc_stats.items():
            md_content += f"| {stat_name} | {value:.6f} |\n"
        md_content += "\n"
        
        md_content += "### Y染色体浓度分布\n\n"
        md_content += "![Y染色体浓度分布](y_chromosome_concentration_distribution.png)\n\n"
        
        md_content += "### 浓度达标情况\n\n"
        md_content += "| 指标 | 值 |\n"
        md_content += "|-----|-----|\n"
        md_content += f"| 总达标率 | {reach_stats['总达标率']:.2%} |\n"
        md_content += f"| 未达标率 | {reach_stats['未达标率']:.2%} |\n\n"
        
        # 5. BMI与达标时间的关系
        md_content += "## 5. BMI与Y染色体浓度达标时间的关系\n\n"
        md_content += "### 相关性分析\n\n"
        md_content += "| 指标 | 值 |\n"
        md_content += "|-----|-----|\n"
        md_content += f"| 相关系数(r) | {bmi_reach_corr['相关系数']:.3f} |\n"
        md_content += f"| p值 | {bmi_reach_corr['p值']:.3f} |\n"
        
        # 解释相关性结果
        corr_interpretation = ""
        if bmi_reach_corr['p值'] < 0.05:
            if abs(bmi_reach_corr['相关系数']) < 0.1:
                corr_interpretation = "极弱相关，但具有统计学意义"
            elif abs(bmi_reach_corr['相关系数']) < 0.3:
                corr_interpretation = "弱相关，但具有统计学意义"
            elif abs(bmi_reach_corr['相关系数']) < 0.5:
                corr_interpretation = "中等相关，且具有统计学意义"
            else:
                corr_interpretation = "强相关，且具有统计学意义"
        else:
            corr_interpretation = "相关性不具有统计学意义"
        
        direction = "正相关" if bmi_reach_corr['相关系数'] > 0 else "负相关"
        md_content += f"| 相关性解释 | {direction}，{corr_interpretation} |\n\n"
        
        md_content += "### BMI与达标时间关系图\n\n"
        md_content += "![BMI与达标时间关系](bmi_vs_reach_time.png)\n\n"
        
        # 6. BMI分组与最佳NIPT检测时点
        md_content += "## 6. BMI分组与最佳NIPT检测时点\n\n"
        md_content += "### 基于医学标准的BMI分组分析\n\n"
        md_content += "| BMI分组 | 样本数 | 平均达标时间（天） | 达标率 |\n"
        md_content += "|--------|-------|----------------|-------|\n"
        for group, row in optimal_time_analysis.iterrows():
            md_content += f"| {group} | {int(row['样本数'])} | {row['平均达标时间']:.1f} | {row['达标率']:.2%} |\n"
        md_content += "\n"
        
        md_content += "### 不同BMI分组在各孕天的达标率\n\n"
        md_content += "![各BMI分组达标率曲线](reach_rate_by_bmi_group.png)\n\n"
        
        # 7. 检测误差分析
        md_content += "## 7. 检测误差分析\n\n"
        md_content += "### 总体误差情况\n\n"
        md_content += "| 指标 | 值 |\n"
        md_content += "|-----|-----|\n"
        md_content += f"| 总体平均变异系数 | {error_analysis['总体平均变异系数']:.3f} |\n"
        md_content += f"| BMI与误差相关系数 | {error_analysis['BMI与误差相关性']['相关系数']:.3f} |\n"
        md_content += f"| 相关性p值 | {error_analysis['BMI与误差相关性']['p值']:.3f} |\n\n"
        
        md_content += "### BMI与检测误差的关系\n\n"
        md_content += "![BMI与检测误差关系](bmi_vs_detection_error.png)\n\n"
        
        # 8. 结论与建议
        md_content += "## 8. 结论与建议\n\n"
        
        conclusions = []
        
        # BMI分组建议
        conclusions.append("### BMI分组建议\n")
        conclusions.append("根据医学标准和数据分析结果，建议将孕妇BMI分为以下四组：\n")
        conclusions.append("1. **偏瘦组**：BMI < 18.5\n")
        conclusions.append("2. **正常组**：18.5 ≤ BMI < 24\n")
        conclusions.append("3. **超重组**：24 ≤ BMI < 28\n")
        conclusions.append("4. **肥胖组**：BMI ≥ 28\n\n")
        
        # 最佳检测时点建议
        conclusions.append("### 最佳NIPT检测时点建议\n")
        for group, row in optimal_time_analysis.iterrows():
            # 找到该组达标率达到95%的最早孕天
            reach_rate_dict = row['不同孕天的达标率']
            suggested_day = None
            for day_str, rate in sorted(reach_rate_dict.items()):
                day = int(day_str.replace('天', ''))
                if rate >= 0.95:
                    suggested_day = day
                    break
            
            if suggested_day:
                conclusions.append(f"- **{group}**：建议在孕{suggested_day}天（约{suggested_day//7}周）进行检测，此时达标率可达95%以上。\n")
            else:
                max_day = max(int(d.replace('天', '')) for d in reach_rate_dict.keys())
                max_rate = max(reach_rate_dict.values())
                conclusions.append(f"- **{group}**：即使在孕{max_day}天（约{max_day//7}周），达标率仅为{max_rate:.1%}，建议适当延迟检测或增加检测次数。\n")
        conclusions.append("\n")
        
        # 潜在风险最小化建议
        conclusions.append("### 潜在风险最小化建议\n")
        conclusions.append("1. 对于BMI较高的孕妇，建议适当延迟检测时间，以提高Y染色体浓度达标率。\n")
        conclusions.append("2. 对于肥胖孕妇，可考虑增加测序深度或采用更敏感的检测技术，降低假阴性风险。\n")
        conclusions.append("3. 检测报告中应注明孕妇BMI对检测结果的潜在影响，为临床决策提供参考。\n")
        conclusions.append("4. 建立基于BMI的个性化检测时间窗口，提高检测的准确性和稳定性。\n")
        conclusions.append("5. 对于检测结果不确定的高BMI孕妇，建议进行随访检测或结合其他检测方法。\n\n")
        
        # 检测误差对结果的影响分析
        conclusions.append("### 检测误差对结果的影响\n")
        if abs(error_analysis['BMI与误差相关性']['相关系数']) > 0.3 and error_analysis['BMI与误差相关性']['p值'] < 0.05:
            conclusions.append(f"分析发现，BMI与Y染色体浓度的检测误差呈{('正' if error_analysis['BMI与误差相关性']['相关系数'] > 0 else '负')}相关，这表明随着BMI的{('增加' if error_analysis['BMI与误差相关性']['相关系数'] > 0 else '降低')}，检测结果的变异程度也会相应{('增加' if error_analysis['BMI与误差相关性']['相关系数'] > 0 else '增加')}。\n")
        else:
            conclusions.append("未发现BMI与检测误差之间存在显著相关性，表明检测误差在不同BMI组之间相对稳定。\n")
        conclusions.append("\n")
        
        # 整合结论到报告中
        md_content += "\n".join(conclusions)
        
        # 保存Markdown报告
        report_path = os.path.join(self.output_dir, 'bmi_y_chromosome_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write(md_content)
        
        return report_path

# 主函数
if __name__ == "__main__":
    file_path = "c:\\Users\\30513\\Desktop\\workspace\\data\\附件_男胎检测数据_10_25.csv"
    analyzer = BMIAnalyzer(file_path)
    report_path = analyzer.generate_md_report()
    print(f"分析报告已生成：{report_path}")