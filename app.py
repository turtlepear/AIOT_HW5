import streamlit as st
from transformers import pipeline
import numpy as np
import random

# ----------------------
# Streamlit UI 設定
# ----------------------
st.set_page_config(page_title="AI vs Human Detector", layout="wide")

st.markdown("""
<style>
body {background-color: #f8f9fa; font-family: "Segoe UI", sans-serif;}
.stButton>button {background: linear-gradient(90deg,#4facfe,#00f2fe); color:white; height:3em; font-size:16px; border-radius:12px; border:none;}
.stTextArea>div>textarea {font-size:16px;}
.progress-bar {border-radius:10px; height:20px;}
.card {background-color:#ffffff; padding:20px; border-radius:15px; box-shadow:0 4px 20px rgba(0,0,0,0.1); text-align:center; margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

st.title("📝 AI vs Human 文章偵測器 (英文版)")
st.write("輸入英文文本，我們會預測它是 AI 生成還是人類撰寫。")

# ----------------------
# 示範文本分組
# ----------------------
demo_texts = {
    "Human": [
        "How are you?",
        "This college essay tip is by Dhivya Arumugham, Kaplan Test Prep's director of SAT and ACT programs.",
        "Reflect on a time when you questioned or challenged a belief or idea. What prompted your thinking? What was the outcome?"
    ],
    "AI": [
        "Generative AI models have revolutionized content creation, producing coherent outputs across multiple domains.",
        "In the era of digital transformation, artificial intelligence demonstrates unparalleled capabilities in automating complex processes efficiently.",
        "Machine learning algorithms, when applied to large-scale datasets, can identify patterns that human analysts might overlook, thereby enhancing decision-making processes."
    ]
}

# ----------------------
# 下拉選擇 Human / AI
# ----------------------
col1, col2 = st.columns([1,3])
with col1:
    demo_choice = st.selectbox("示範類別", [""] + list(demo_texts.keys()))
    if st.button("🎲 隨機示範文本"):
        if demo_choice:
            user_input = random.choice(demo_texts[demo_choice])
        else:
            user_input = ""
with col2:
    if demo_choice:
        user_input = random.choice(demo_texts[demo_choice])
    else:
        user_input = ""
    user_input = st.text_area("請輸入文章內容", value=user_input, height=200)

# ----------------------
# 模型載入
# ----------------------
@st.cache_resource
def load_model():
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    return classifier

classifier = load_model()

# ----------------------
# 判斷按鈕
# ----------------------
if st.button("判斷"):
    if user_input.strip() == "":
        st.warning("⚠️ 請先輸入文本！")
    else:
        result = classifier(user_input[:512])[0]
        ai_prob = np.clip(result['score'], 0.0, 1.0) if result['label'] == 'POSITIVE' else np.clip(1 - result['score'], 0.0, 1.0)
        human_prob = 1 - ai_prob

        # ----------------------
        # 美化結果卡片
        # ----------------------
        st.subheader("📊 判斷結果")
        col_ai, col_human = st.columns(2)
        with col_ai:
            st.markdown(f"""
                <div class="card">
                <h3>AI 生成概率</h3>
                <h1 style='color:#ff4b4b'>{ai_prob*100:.2f}%</h1>
                <progress class="progress-bar" value="{ai_prob*100}" max="100" style="width:100%; background:#ffe5e5;"></progress>
                </div>
            """, unsafe_allow_html=True)
        with col_human:
            st.markdown(f"""
                <div class="card">
                <h3>Human 撰寫概率</h3>
                <h1 style='color:#4CAF50'>{human_prob*100:.2f}%</h1>
                <progress class="progress-bar" value="{human_prob*100}" max="100" style="width:100%; background:#e5ffe5;"></progress>
                </div>
            """, unsafe_allow_html=True)

        # 條形圖視覺化
        st.bar_chart({"AI": ai_prob, "Human": human_prob})

# ----------------------
# 清空文本按鈕
# # ----------------------
# if st.button("🧹 清空文本"):
#     st.experimental_rerun()
