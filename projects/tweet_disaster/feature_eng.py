import re

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_username(text):
    user = re.compile(r'@[A-Za-z0-9_]+')
    return user.sub(r'', text)

def feature_engineering(text):
    text = remove_URL(text)
    text = remove_numbers(text)
    text = remove_html(text)
    text = remove_username(text)
    return " ".join(text.split())
