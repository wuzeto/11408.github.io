import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ----------------------------------------------------------------------------
# 1. 模型参数设定（参考知识库[[3]][[6]][[7]]）
# ----------------------------------------------------------------------------
# 专业类型参数（专硕/学硕、文科/理工科）
PROFESSIONS = {
    "专硕_理工科": {"alpha": 1.2, "beta": 0.7},  # 扩招且竞争较小（如计算机专硕[[3]]）
    "学硕_文科": {"alpha": 0.8, "beta": 1.5},   # 微缩且竞争激烈（如哲学[[6]]）
    "新兴学科": {"alpha": 1.1, "beta": 0.9}     # 适度扩招（如人工智能[[4]]）
}

# 初始分数与时间步长
INITIAL_SCORES = {"专硕_理工科": 350, "学硕_文科": 320, "新兴学科": 340}
TIME_STEPS = 5  # 预测未来5年
dt = 1  # 时间步长（年）

# ----------------------------------------------------------------------------
# 2. 动态分数调整模型（微分方程求解）
# ----------------------------------------------------------------------------
def score_model(S, t, k, alpha, beta):
    """分数动态调整方程：dS/dt = -k*(S - S_avg) * (alpha/beta)"""
    S_avg = np.mean(S)  # 当前平均分数
    return -k * (S - S_avg) * (alpha / beta)

def predict_scores(initial_scores, k=0.1, alpha=1.0, beta=1.0, years=5):
    """预测未来分数变化"""
    t = np.linspace(0, years, years+1)
    results = odeint(score_model, initial_scores, t, args=(k, alpha, beta))
    return results.flatten()

# ----------------------------------------------------------------------------
# 3. 风险评估函数（参考[[3]][[6]]）
# ----------------------------------------------------------------------------
def calculate_risk(current_score, avg_score, alpha, beta):
    """综合风险值：分数波动 + 竞争与扩招影响"""
    score_risk = abs((current_score - avg_score)/avg_score) * 100
    competition_risk = 1 - (alpha / beta)  # 扩招系数与竞争的比值
    return 0.7*score_risk + 0.3*competition_risk  # 权重分配

# ----------------------------------------------------------------------------
# 4. Streamlit交互界面
# ----------------------------------------------------------------------------
st.title("考研分数预测与风险评估系统")
st.write("基于扩招政策与学科竞争的动态模型 [[3]][[6]]")

# 用户输入参数
selected_profession = st.selectbox("选择专业类型", list(PROFESSIONS.keys()))
k = st.slider("放水调整系数（k）", 0.01, 0.5, 0.1)
years = st.slider("预测年数（1-5年）", 1, 5, 3)

# 获取参数
alpha = PROFESSIONS[selected_profession]["alpha"]
beta = PROFESSIONS[selected_profession]["beta"]
initial_score = INITIAL_SCORES[selected_profession]

# 预测分数
predicted_scores = predict_scores(initial_score, k, alpha, beta, years)[-1]  # 最终年分数

# 风险计算
current_score = initial_score
avg_score = np.mean([v for v in INITIAL_SCORES.values()])
risk = calculate_risk(current_score, avg_score, alpha, beta)

# 可视化结果
st.subheader("预测结果")
col1, col2 = st.columns(2)
col1.metric("当前分数", f"{current_score:.1f}")
col1.metric("预测分数", f"{predicted_scores:.1f}")
col2.metric("风险指数", f"{risk:.1f}/100", delta=f"{'↑高风险' if risk>50 else '↓低风险'}")

# 绘制趋势图
st.subheader("分数变化趋势（未来5年）")
t = np.arange(0, years+1)
scores = predict_scores(initial_score, k, alpha, beta, years)
plt.figure(figsize=(10, 4))
plt.plot(t, scores, marker='o', linestyle='--')
plt.title(f"{selected_profession} 分数预测（k={k:.2f}）")
plt.xlabel("年份")
plt.ylabel("平均分数")
st.pyplot(plt)

# 风险提示
st.warning(f"**风险提示**：{selected_profession}专业竞争系数β={beta:.1f}，"
           f"扩招系数α={alpha:.1f}，建议关注{'' if alpha>1 else '调剂'}[[3]][[6]]")