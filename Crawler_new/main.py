from bs4 import BeautifulSoup
import requests
import re
import json
import os
from mongoDB_config import MongoDB_config


def read_txt_urls(file_name):
    file = open(file_name, 'r')
    list_urls = []
    for i in file:
        list_urls.append(i.replace('\n', ""))
    return list_urls


def where_json(file_name):
    return os.path.exists(file_name)


def write_json_file(file_name, data):
    if where_json(file_name) == False or os.path.getsize(file_name) == 0:
        with open(file_name, 'w', encoding='utf8') as fout:
            json.dump(data, fout, indent=4, ensure_ascii=False)
    else:
        with open(file_name, 'r', encoding='utf8') as fin:
            d = json.loads(fin.read())
        for i in data:
            d.append(i)
        with open(file_name, 'w', encoding='utf8') as fout:
            json.dump(d, fout, indent=4, ensure_ascii=False)


def read_json_file(file_name):
    f = open(file_name, encoding='utf8')
    data = json.load(f)
    for i in data:
        print(i)
    f.close()


def crawl(urls, label):
    data = []

    for url in urls:
        print(url)
        collect = MongoDB_config("Web_Crawl", "Web_Struct").get_collections_web().find()
        for obj in collect:
            if obj["dommain"] in url:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")
                if (len(soup.select(obj["title"])) == 0
                        or len(soup.select(obj["summary"])) == 0
                        or len(soup.select(obj["conetent"])) == 0):
                    break
                title_ = soup.select(obj["title"])[0].getText()
                summary_ = soup.select(obj["summary"])[0].getText()
                content_ = " ".join([x.getText() for x in soup.select(obj["conetent"])])
                dict = {}
                dict['title'] = re.sub(r'([\s]{2,})', "", title_)
                dict['summary'] = re.sub(r'([\s]{2,})', "", summary_)
                dict['content'] = re.sub(r'([\s]{2,})', "", content_)
                dict['label'] = label
                data.append(dict)
            else:
                continue
    return data


if __name__ == '__main__':
    label = "neutral"
    write_json_file("data_news.json", crawl(read_txt_urls('urls.txt'), label))
    read_json_file("data_news.json")
