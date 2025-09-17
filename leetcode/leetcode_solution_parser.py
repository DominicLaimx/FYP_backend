import pandas as pd

df = pd.read_csv(r"C:\Users\ASUS\Downloads\FYP interview\FYP_backend\leetcode\leetcode_category_list.csv")
print(len(df))
print(df["solution"][0])