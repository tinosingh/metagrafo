import requests


def test():
    url = "http://localhost:9001/health"
    try:
        r = requests.get(url, timeout=5)
        print(f"Server status: {r.status_code}\n{r.text}")
    except Exception as e:
        print(f"Server unreachable: {str(e)}")


if __name__ == "__main__":
    test()
