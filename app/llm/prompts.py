"""提示词模板：RAG 问答（引用+拒答）、OCR 纠错、评测集问题生成。"""

RAG_SYSTEM = (
    "你是 DocBrain，一个严谨的本地文档问答助手。"
    "你只能依据用户提供的编号资料回答问题，绝不编造。"
    "输出必须是一个合法的 JSON 对象。"
)

RAG_USER_TMPL = """资料片段（编号引用）：
{context}

问题：{question}

输出要求（严格遵守）：
1. 只输出一个 JSON 对象，格式：{{"answer": "...", "evidence_sufficient": true}}
2. answer 中每个结论后用 [n] 标注来源编号，例如：该功能于 2023 年上线[1]，支持多格式解析[2]。
3. 资料不足以回答问题时，evidence_sufficient 置为 false，answer 用一句话说明无法从文档中确定，不要猜测、不要编造。
4. 使用与问题相同的语言作答。"""

OCR_CORRECT_SYSTEM = (
    "你是 OCR 文本校对助手。用户会给出 OCR 识别原文，请纠正其中明显的识别错误"
    "（形近汉字、0/O 与 1/l/I 混淆、全半角混乱、断行粘连等），"
    "保持原有信息、数字与格式不变，不得增删或改写内容，没有把握的地方保持原样。"
    "只输出纠正后的文本，不要任何解释。"
)

QA_GEN_PROMPT = """阅读以下文档片段，生成一个具体的问题。要求：
1. 问题必须能仅凭该片段直接回答；
2. 问题要具体（涉及片段中的关键事实/数字/结论），不要笼统；
3. 不要出现"片段""上文""文档"等字样，不要直接照抄原文句子；
4. 只输出问题本身，一行。

文档片段：
{text}"""


def build_rag_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    """context_chunks: [{doc_name, page, text}, ...]（已按引用顺序编号）"""
    parts = []
    for i, c in enumerate(context_chunks, 1):
        loc = f"《{c['doc_name']}》"
        if c.get("page"):
            loc += f" 第{c['page']}页"
        parts.append(f"[{i}] {loc}\n{c['text']}")
    context = "\n\n".join(parts)
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": RAG_USER_TMPL.format(context=context, question=question)},
    ]
