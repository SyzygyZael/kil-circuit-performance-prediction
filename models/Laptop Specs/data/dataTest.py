import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Kevin Nesbitt/Documents/Coding/Python/Deep Learning/Laptop Specs/data/laptop_data.csv")

print(df.corr()["Price"].sort(ascending=False))