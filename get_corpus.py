import os
import requests

# Use the GitHub raw URLs to download the files individually
# If you know the exact filenames, you can automate this
files = [
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/alice_wonderland.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/artofwar.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/dracula.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/frankenstein.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/greatgatsby.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/grimm.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/iliad.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/lesmiserables.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/metamorphosis.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/moby_dick.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/montecristo.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/odyssey.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/prideandprejudice.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/romeoandjuliet.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/sherlock_holmes.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/sleepyhollow.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/therepublic.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/treasure_island.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/warandpeace.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/warofworlds.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/whinniethepooh.txt",
    "https://raw.githubusercontent.com/jghawaly/CSC7809_FoundationModels/main/Project2/data/raw/wizardofoz.txt",
]

os.makedirs("raw", exist_ok=True)

with open("corpus.txt", "w", encoding="utf-8") as outfile:
    for url in files:
        filename = os.path.join("raw", url.split("/")[-1])
        r = requests.get(url)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(r.text)
        outfile.write(r.text + "\n")