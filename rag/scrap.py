import os

import requests
from bs4 import BeautifulSoup

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "../knowledge/")


def scrape(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text

        # parse and extract the txt
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines()]
        clean = "\n\n".join(line for line in lines if line)

        # create filename and save in Knowledge dir
        filename = url.split("/")[-1] or "untitled"
        filepath = os.path.join(KNOWLEDGE_DIR, f"{filename}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(clean)

        print(f"Saved: {filepath}")
        print(f"Chars count: {len(clean)}")

    # site returns error
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")

    # ctrl + c to exit
    except KeyboardInterrupt:
        print("\nExited")


# read
if __name__ == "__main__":
    try:
        url = input("Paste URL: ")
        scrape(url)
    except KeyboardInterrupt:
        print("\nExited.")
