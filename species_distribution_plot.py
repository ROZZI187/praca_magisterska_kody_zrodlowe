import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\soundscape_data\annotations_clean.csv"
df = pd.read_csv(path)

species_col = "Species eBird Code"

counts = df[species_col].value_counts()

top_species = counts.head(15)

plt.figure(figsize=(12,6))

top_species.plot(kind="bar")

plt.title("Najczęściej występujące gatunki w zbiorze SSW")
plt.xlabel("Kod gatunku (eBird)")
plt.ylabel("Liczba adnotacji")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("rys_4_1_licznosc_gatunkow.png", dpi=300)

plt.show()