import requests

url = "https://evan-finnigan.github.io"
r = requests.get(url,timeout=5)
if r.status_code == 200:
    for cookie in r.cookies:
        print(cookie)            # Use "print cookie" if you use Python 2.
