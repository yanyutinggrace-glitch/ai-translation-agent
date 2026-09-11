import streamlit as st
from zhipuai import ZhipuAI
import json
import os
import pandas as pd
from datetime import datetime

# -----------------------------
# 1. 页面基本配置
# -----------------------------
st.set_page_config(
    page_title="AI笔译助教",
    page_icon="📝",
    layout="wide"
)

# 【修改1：精简了标题，并更新了副标题包含“英汉互译”】
st.title("📝 AI笔译助教")
st.caption("英汉互译多维质量评估与自主学习系统")

# -----------------------------
# 2. 侧边栏配置与数据持久化
# -----------------------------
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    # 【修改2：在后台直接固定 API Key。请把下面引号里的内容换成你真实的 API Key！】
    api_key = st.secrets["ZHIPU_API_KEY"] 
    
    model_name = st.selectbox("选择大模型", ["glm-4-flash", "glm-4"], index=0)
    
    st.divider()
    st.header("🔄 翻译诊断模式")
    translation_direction = st.radio("请选择当前练习的翻译方向：", ["汉译英 (C-E)", "英译汉 (E-C)"])

# 设定学术科研数据记录文件
LOG_FILE = "student_evaluation_logs.csv"

def save_log(student_id, direction, source_text, target_text, eval_data):
    """将每次诊断结果写入本地 CSV，便于导出用于实证研究与数据分析"""
    records = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    errors = eval_data.get("errors", [])
    
    if not errors:
        records.append({
            "Timestamp": timestamp,
            "Student_ID": student_id,
            "Direction": direction,
            "Source_Text": source_text,
            "Target_Text": target_text,
            "Category": "None",
            "Error_Code": "0.0",
            "Error_Name": "No Error",
            "Severity": "None",
            "Reason": "No major errors identified.",
            "Revised_Target": eval_data.get("revised_translation", "")
        })
    else:
        for err in errors:
            records.append({
                "Timestamp": timestamp,
                "Student_ID": student_id,
                "Direction": direction,
                "Source_Text": source_text,
                "Target_Text": target_text,
                "Category": err.get("category", ""),
                "Error_Code": err.get("error_code", ""),
                "Error_Name": err.get("error_name", ""),
                "Severity": err.get("severity", ""),
                "Reason": err.get("reason", ""),
                "Revised_Target": eval_data.get("revised_translation", "")
            })
            
    df_new = pd.DataFrame(records)
    # 【修复报错：强制所有读写操作使用 utf_8_sig 编码，防止 Windows 报 ascii 错误】
    if not os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, index=False, encoding="utf_8_sig")
    else:
        df_new.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf_8_sig")

# -----------------------------
# 3. 核心 Prompt 与评估逻辑
# -----------------------------
def analyze_translation(client, source_text, target_text, model, direction):
    if direction == "汉译英 (C-E)":
        prompt_template = """
        你是一名权威的高校翻译专业学术审稿人与教学专家。请基于以下 MQM 框架对学生的【汉译英】译文进行质量评估：
        
        【MQM 错误分类及严重度定义】
        1. Accuracy:
           - 1.1 Mistranslation (Major): 核心语义误译。
           - 1.2 Omission (Major): 关键信息漏译。
           - 1.3 Addition (Minor): 无中生有的增译。
        2. Fluency:
           - 2.1 Grammatical Error (Minor): 英语语法错误（时态、冠词、主谓一致等）。
           - 2.2 Lexical Error (Minor): 选词不当或搭配不自然。
           - 2.3 Stylistic Error (Major): 中式英语 (Chinglish)，表达生硬。
           - 2.4 Misordering (Minor): 语序错误。
        3. Locale:
           - 3.1 Orthographic Error (Trivial): 拼写、大小写或标点错误。
        4. Style:
           - 3.2 Pragmatic Error (Minor): 语用错误，语气或正式程度不符。

        【特别诊断指令】
        在诊断中国哲学语料（如涉及“理”、“气”等概念的王夫之哲学文本）时，请特别关注文化内涵的准确传递。严禁使用集装箱式隐喻 (container metaphors) 和二元对立的句法结构，必须采用重过程、非二元对立的翻译策略，确保价值对齐。

        【输入内容】
        - 中文原文: {source_text}
        - 学生英译文: {target_text}
        """
    else:
        prompt_template = """
        你是一名权威的高校翻译专业学术审稿人与教学专家。请基于以下 MQM 框架对学生的【英译汉】译文进行质量评估：
        
        【MQM 错误分类及严重度定义】
        1. Accuracy:
           - 1.1 Mistranslation (Major): 核心语义误译（如习语、多义词误判）。
           - 1.2 Omission (Major): 关键信息漏译。
           - 1.3 Addition (Minor): 误导读者的多余增译。
        2. Fluency (重点关注汉语地道性):
           - 2.1 Grammatical Error (Minor): 汉语语法错误（如成分残缺、搭配不当）。
           - 2.2 Lexical Error (Minor): 选词不当（如书面语与口语混用，成语使用不当）。
           - 2.3 Stylistic Error (Major): 翻译腔 / 欧化汉语 (Translationese)。如生硬的结构、滥用“被”字句、滥用代词等。
           - 2.4 Misordering (Minor): 句法结构错乱。定语从句或状语从句未按汉语逻辑前置或拆分。
        3. Locale:
           - 3.1 Orthographic Error (Trivial): 错别字，中英文标点混用（如中文句末使用英文句点）。
        4. Style:
           - 3.2 Pragmatic Error (Minor): 语用及文体不符。

        【输入内容】
        - 英文原文: {source_text}
        - 学生中译文: {target_text}
        """

    json_instruction = """
    【要求】
    请务必以严格的 JSON 格式输出，包含以下键值：
    {
        "errors": [
            {
                "category": "主分类（如 Fluency）",
                "error_code": "编号（如 2.3）",
                "error_name": "名称（如 Stylistic Error）",
                "severity": "严重度（Major / Minor / Trivial）",
                "problematic_segment": "译文中出错的具体词或句段",
                "reason": "结合翻译理论与错误成因的深度剖析"
            }
        ],
        "overall_comment": "针对学生本次翻译的综合教学评语",
        "revised_translation": "规范化、地道的参考修改译文"
    }
    """
    
    full_prompt = prompt_template.format(source_text=source_text, target_text=target_text) + json_instruction
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional translation evaluation AI. Always output valid JSON strictly matching the requested format."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# -----------------------------
# 4. 界面交互布局
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 翻译实践区")
    student_id = st.text_input("学生学号/标识", value="Student_01", help="自动记录至后台实证数据池")
    
    src_label = "中文原文" if "C-E" in translation_direction else "英文原文"
    tgt_label = "学生英译文" if "C-E" in translation_direction else "学生中译文"
    
    source_text = st.text_area(src_label, height=120)
    student_text = st.text_area(tgt_label, height=120)
    
    submit_btn = st.button("🚀 提交 AI 诊断", type="primary", use_container_width=True)

with col2:
    # 【修改3：更新面板标题】
    st.subheader("📋 AI诊断报告")
    if submit_btn:
        # 添加容错：如果用户忘记在代码里改 API Key
        if not api_key or api_key == "在这里填入你申请的智谱API_Key":
            st.warning("⚠️ 教师尚未在后台配置有效的 API Key，系统暂时无法提供诊断。")
        elif not source_text.strip() or not student_text.strip():
            st.warning("⚠️ 原文与译文均不能为空。")
        else:
            with st.spinner(f"AI 助教正在依据 MQM 标准进行 {translation_direction} 深度诊断..."):
                try:
                    client = ZhipuAI(api_key=api_key)
                    result = analyze_translation(client, source_text, student_text, model_name, translation_direction)
                    
                    save_log(student_id, translation_direction, source_text, student_text, result)
                    
                    errors = result.get("errors", [])
                    if errors:
                        st.error(f"🔍 检测到 {len(errors)} 处典型错误：")
                        for idx, err in enumerate(errors, 1):
                            with st.expander(f"[{err.get('error_code')}] {err.get('error_name')} ({err.get('severity')})", expanded=True):
                                st.markdown(f"**🔴 出错片段：** `{err.get('problematic_segment', 'N/A')}`")
                                st.markdown(f"**📖 成因剖析：** {err.get('reason', '')}")
                    else:
                        st.success("🎉 译文质量极佳，未检测到显著错误！")

                    st.markdown("---")
                    # 【修改3：更新优化建议标题】
                    st.markdown("#### 💡 优化建议")
                    st.info(f"**参考译文：** {result.get('revised_translation')}")
                    st.caption(f"**教学评语：** {result.get('overall_comment')}")
                    st.toast("评估完成，结果已自动归档至科研数据集！")

                except Exception as e:
                    st.error(f"系统调用异常: {str(e)}")

# -----------------------------
# 5. 教师后台数据池
# -----------------------------
st.divider()
if os.path.exists(LOG_FILE):
    with st.expander("📁 教师后台：翻译过程语料与诊断日志下载 (CSV)"):
        # 【修复报错：读取 CSV 时也加上 utf_8_sig 编码】
        df_logs = pd.read_csv(LOG_FILE, encoding="utf_8_sig")
        st.dataframe(df_logs.tail(10), use_container_width=True)
        csv_data = df_logs.to_csv(index=False, encoding='utf_8_sig')
        st.download_button(
            label="⬇️ 导出完整标注数据集 (用于实证类论文分析)",
            data=csv_data,
            file_name="MQM_AI_TPACK_Dataset.csv",
            mime="text/csv",
        )
