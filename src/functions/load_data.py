import pandas as pd

def load_data(base, file_urls):
    data = []
    n = len(file_urls)
    count = 0

    for url in file_urls:
        df = pd.read_csv(base + url)
        data.append(df)
        count +=1
        print(f"\r{count}/{n} files loaded..", end="", flush=True)

    return data