from bs4 import BeautifulSoup
import requests

def crawl(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    all_link = soup.findAll("h3.article-title")
    print(soup)
    for i in all_link:
        ans = i.select("a").getText()
        print(ans)

if __name__ == "__main__":
    crawl("https://dantri.com.vn/du-lich/kham-pha/trang-16.htm")
