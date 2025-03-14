import pandas as pd
import re

def clean_text(text):
    """
    对文本进行基本清洗：
      1. 去除 quoted-printable 的软换行符号 "=\n"
      2. 合并多个换行符为一个
      3. 去除每行开头的 ">" 标记及其后空格
    """
    # 如果文本不是字符串，先转换为字符串
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'=\n', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'^\s*>+\s?', '', text, flags=re.MULTILINE)
    return text.strip()

def filter_body(text):
    """
    过滤掉邮件头部信息行，只保留正文内容。
    删除那些以常见邮件头关键词开头的行。
    """
    headers = [
        "Return-Path:", "Received:", "Message-ID:", "Date:",
        "From:", "To:", "Subject:", "Mime-Version:",
        "Content-Type:", "Content-Transfer-Encoding:", "X-Mailer:"
    ]
    lines = text.split("\n")
    filtered_lines = []
    for line in lines:
        if any(line.strip().startswith(prefix) for prefix in headers):
            continue
        filtered_lines.append(line.strip())
    return "\n".join(filtered_lines).strip()

def process_text(text):
    """
    对文本进行综合处理：先清洗后过滤邮件头信息，
    仅保留正文的纯自然语言内容。
    """
    cleaned = clean_text(text)
    filtered = filter_body(cleaned)
    return filtered

if __name__ == "__main__":
    # 读取已有的 CSV 文件，文件中应包含 "text" 列
    input_csv = "spamassassin_emails_body_final.csv"
    output_csv = "spamassassin_emails_plain.csv"
    
    df = pd.read_csv(input_csv, encoding="utf-8")
    
    # 对 "text" 列应用 process_text 处理
    df["text"] = df["text"].apply(process_text)
    
    # 保存新的纯文本 CSV 文件
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Finished creating plain text CSV: {output_csv}")
