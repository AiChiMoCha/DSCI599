import os
import csv
import re
import email
from email import policy
from bs4 import BeautifulSoup

def clean_text(text):
    """
    对提取的正文进行进一步清洗：
      1. 去除 quoted-printable 中的软换行标记 "=\n"。
      2. 将连续的换行符替换为一个换行符。
      3. 去除每行开头的 ">" 符号及其后可能的空格。
      4. 删除首尾多余的空白字符。
    """
    text = re.sub(r'=\n', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'^\s*>+\s?', '', text, flags=re.MULTILINE)
    return text.strip()

def extract_body(msg):
    """
    从邮件对象msg中提取正文部分。
    优先提取text/plain部分；如果没有，再查找text/html并转换为纯文本。
    """
    body = ""
    if msg.is_multipart():
        # 尝试提取 text/plain 内容
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disp = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disp:
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="ignore")
                    break
                except Exception:
                    continue
        # 如果未找到 text/plain，则查找 text/html 部分
        if not body:
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disp = str(part.get("Content-Disposition"))
                if content_type == "text/html" and "attachment" not in content_disp:
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or "utf-8"
                        html = payload.decode(charset, errors="ignore")
                        soup = BeautifulSoup(html, "html.parser")
                        body = soup.get_text(separator="\n", strip=True)
                        break
                    except Exception:
                        continue
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                content = payload.decode(charset, errors="ignore")
            except Exception:
                content = payload.decode("utf-8", errors="ignore")
            if content_type == "text/html":
                soup = BeautifulSoup(content, "html.parser")
                body = soup.get_text(separator="\n", strip=True)
            else:
                body = content

    return clean_text(body)

def parse_email(file_path):
    """
    解析单个RFC 822格式邮件文件，提取发件人、收件人、主题、日期和正文内容。
    """
    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
        msg = email.message_from_file(f, policy=policy.default)
    
    sender = msg.get('From', '')
    recipient = msg.get('To', '')
    subject = msg.get('Subject', '')
    date = msg.get('Date', '')
    body = extract_body(msg)
    
    return sender, recipient, subject, date, body

def create_email_csv(base_dir, output_csv):
    """
    遍历 SpamAssassin 数据集下的指定子文件夹，
    将每封邮件的发件人、收件人、主题、日期、正文内容和标签写入 CSV 文件中。
    CSV 文件包含六列：From, To, Subject, Date, Body, Label
    标签定义如下：
      - easy_ham/easy_ham 和 hard_ham/hard_ham 均归为 ham
      - spam_2/spam_2 归为 spam
    """
    # 定义子目录及其对应的标签
    subfolders = [
        ("easy_ham/easy_ham", "ham"),
        ("hard_ham/hard_ham", "ham"),
        ("spam_2/spam_2", "spam")
    ]
    fieldnames = ["From", "To", "Subject", "Date", "Body", "Label"]
    
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for subpath, label in subfolders:
            folder_path = os.path.join(base_dir, subpath)
            if not os.path.isdir(folder_path):
                print(f"目录未找到: {folder_path}")
                continue
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if not os.path.isfile(file_path) or filename.startswith('.') or filename == "__MACOSX":
                    continue
                try:
                    sender, recipient, subject, date, body = parse_email(file_path)
                    writer.writerow({
                        "From": sender,
                        "To": recipient,
                        "Subject": subject,
                        "Date": date,
                        "Body": body,
                        "Label": label
                    })
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{e}")

if __name__ == "__main__":
    # 假定 SpamAssassin 文件夹与脚本处于同一目录下
    base_dir = "SpamAssassin"
    output_csv = "emails_extracted.csv"
    create_email_csv(base_dir, output_csv)
    print(f"CSV文件已生成: {output_csv}")
