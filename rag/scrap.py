import os
import requests
from bs4 import BeautifulSoup

KNOWLEDGE_DIR = "../knowledge/"


def scrape(url: str):
    response = requests.get(url)
    html = response.text


# parse and extract the txt
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

# remove extra whitespaces
    lines = [line.strip() for line in text.splitlines()]
    clean = "\n\n".join(lines)

# create filename and save in Knowledge dir
    filename = url.split("/")[-1] or "untitled"
    filepath = os.path.join(KNOWLEDGE_DIR, f"{filename}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(clean)

    print(f"Saved: {filepath}")
    print(f"Chars count: {len(clean)}")

# read
if __name__ == "__main__":
    url = input("Paste URL: ")
    scrape(url)
