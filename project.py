"""World Sustainability Dataset"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read csv and assign it to variable df
df = pd.read_csv(
    "/workspaces/World-Sustainability-Dataset/WorldSustainabilityDataset.csv"
)
print(df.head())

# only keep country name, year, final consumption expenditure, renewable energy consumption,
# annual CO2 emissions, continent, and income classification
df = df[
    [
        "Country Name",
        "Year",
        "Final consumption expenditure (% of GDP) - NE.CON.TOTL.ZS",
        "Renewable energy consumption (% of total final energy consumption) - EG.FEC.RNEW.ZS",
        "Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes",
        "Continent",
        "Income Classification (World Bank Definition)",
    ]
]

print(df.head())

# rename columns
df.rename(
    columns={
        "Final consumption expenditure (% of GDP) - NE.CON.TOTL.ZS": "Final Consumption",
        "Renewable energy consumption (% of total final energy consumption) - EG.FEC.RNEW.ZS": "Renewable Energy Consumption",
        "Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes": "Annual CO2 Emissions",
        "Income Classification (World Bank Definition)": "Income Classification",
    },
    inplace=True,
)

print(df.head())

# rearrange columns
df = df[
    [
        "Continent",
        "Country Name",
        "Income Classification",
        "Year",
        "Annual CO2 Emissions",
        "Renewable Energy Consumption",
        "Final Consumption",
    ]
]

print(df.head())

# check to see how many null objects are in the dataset
print(df.isnull().sum())

# quick summary of each column in the dataframe
print(df.describe())

# data dimensions
print(df.shape)

# turn year into str
df["Year"] = df["Year"].astype("str")

# calculate mean of renewable energy consumption grouped by continent and year
print(df.groupby(["Continent", "Year"])["Renewable Energy Use"].mean())

# create new dataframe for mean
cont_REC_df = df.groupby(["Continent", "Year"])["Renewable Energy Use"].mean()\
    .reset_index(name = "Renewable Energy Use (%)")
print(cont_REC_df)

# plot
g = sns.relplot(
    kind = "line", 
    data = cont_REC_df, 
    x = "Year", 
    y = "Renewable Energy Use (%)", 
    hue = "Continent", 
    aspect = 1.75
).set(title = "World Renewable Energy Use from 2000-2018 by Continent")
g.set_xticklabels(rotation = 30)
plt.show()