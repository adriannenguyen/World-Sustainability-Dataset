"""World Sustainability Dataset"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file_path):
    # Read dataset from CSV file
    return pd.read_csv(file_path)

def clean_data(df):
    # Select relevant columns
    df = df[["Country Name", "Year", "Renewable energy consumption (% of total final energy consumption) - EG.FEC.RNEW.ZS",
             "Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes",
             "Continent", "Income Classification (World Bank Definition)"]]

    # Rename columns
    df.columns = ["Country", "Year", "Renewable Energy Use (%)", "Annual CO2 Emissions (Mt)", "Continent", "Income Classification"]

    # Reorder columns
    df = df[["Continent", "Country", "Income Classification", "Year", "Renewable Energy Use (%)", "Annual CO2 Emissions (Mt)"]]

    # Drop rows with any missing values
    df.dropna(axis=0, how="any", inplace=True)

    return df

def plot_data(df, x, y, hue, title):
    # Plot line graph
    g = sns.relplot(kind="line", data=df, x=x, y=y, hue=hue, aspect=1.75).set(title=title)
    g.set_xticklabels(rotation=30)
    plt.show()

def main():
    file_path = "/workspaces/World-Sustainability-Dataset/WorldSustainabilityDataset.csv"
    df = read_data(file_path)
    df = clean_data(df)

    # Convert "Year" to str
    df["Year"] = df["Year"].astype("str")

    # Group and aggregate data
    avg_df = df.groupby(["Continent", "Year"]).agg({
        "Renewable Energy Use (%)": "mean",
        "Annual CO2 Emissions (Mt)": "mean"
    }).reset_index()


    # Plot graphs
    plot_data(avg_df, "Year", "Renewable Energy Use (%)", "Continent", "Average World Renewable Energy Use from 2000-2018 by Continent")
    plot_data(avg_df, "Year", "Annual CO2 Emissions (Mt)", "Continent", "Average Annual World CO2 Emissions from 2000-2018 by Continent")

if __name__ == "__main__":
    main()