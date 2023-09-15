import re
import urllib
import requests
from bs4 import BeautifulSoup

def search(query:str) -> str:
    # Search
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url)

    # Get responce
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

    else :
        return response.status_code
    
    # regex for find url
    regex = re.compile(r"(?<=url\?q=)(.*?)(?=&sa=U)")
    
    # Collect Informations
    links = list()
    for elem in soup.find_all("h3"):
        container_elem = elem.parent.parent.parent.parent.parent
        regex_result = regex.search(str(container_elem.find("a").get("href")))
        if regex_result == None:
            continue
        url = regex_result.group()
        url = urllib.parse.unquote(url)
        title = container_elem.find("h3").text
        content = container_elem.find_all("div")[-1].get_text()
        content = content if content != "" else "youtube"
        content = content if "ago" not in content else "news"

        links = list()
        if content not in ["youtube", "news"]:
            link = {
                "url": url,
                "title": title,
                "content": content,
            }
            links.append(link)
    return links[:3]