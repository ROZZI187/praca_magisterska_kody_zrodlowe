import pandas as pd

input_path = r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\soundscape_data\annotations.csv"

output_path = r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\soundscape_data\annotations_clean.csv"

df = pd.read_csv(input_path)

print("Liczba rekordów przed filtrowaniem:", len(df))

df_clean = df[df["Species eBird Code"] != "????"]

print("Liczba rekordów po filtrowaniu:", len(df_clean))
unknown = (df["Species eBird Code"] == "????").sum()

print("Liczba rekordów z nieznanym gatunkiem:", unknown)
df_clean.to_csv(output_path, index=False)

print("Zapisano oczyszczony plik:", output_path)