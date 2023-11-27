"""World Sustainability Dataset"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file_path):
    # Read dataset from CSV file
    return pd.read_csv(file_path)

def clean_data(df):
    # Select relevant columns
    df = df[["Country Name", "Year", "Access to electricity (% of population) - EG.ELC.ACCS.ZS",
             "Renewable electricity output (% of total electricity output) - EG.ELC.RNEW.ZS",
             "Renewable energy consumption (% of total final energy consumption) - EG.FEC.RNEW.ZS",
             "Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes",
             "Continent", "Income Classification (World Bank Definition)"]] 

    # Rename columns
    df.columns = ["Country", "Year", "Access to Electricity (% of population)", 
                  "Renewable Electricity Output (% of total electricity output)",
                  "Renewable Energy Use (%)", "Annual CO2 Emissions (Mt)", 
                  "Continent", "Income Classification"
                  ]

    # Reorder columns
    df = df[["Continent", "Country", "Income Classification", "Year", 
             "Access to Electricity (% of population)",
             "Renewable Electricity Output (% of total electricity output)",
             "Renewable Energy Use (%)", 
             "Annual CO2 Emissions (Mt)"]]

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Fill missing Continent for Timor-Leste
    df.loc[df["Country"] == "Timor-Leste", "Continent"] = "Asia"
    df.loc[df["Country"] == "Timor-Leste", "Income Classification"] = "Lower-middle income"

    # Convert "Year" to int
    df["Year"] = df["Year"].astype("int")

    return df

def plot_data(df, x, y, hue, title, xlim=None, ylim=None):
    # Filter the dataframe if xlim is specified
    if xlim:
        df = df[df[x] <= xlim]

    # Plot line graph for historical data
    g = sns.relplot(kind="line", data=df, x=x, y=y, hue=hue, aspect=1.75).set(title=title)

    # Set x-axis limit if specified
    if xlim:
        g.set(xlim=(df[x].min(), xlim))

    # Set y-axis limit if specified
    if ylim:
        g.set(ylim=(df[y].min(), ylim))

    g.set_xticklabels(rotation=30)
    plt.show()

def main():
    df = read_data(file_path)
    df = clean_data(df)
    print(df)

    # Group and aggregate data by Continent
    cont_avg_df = df.groupby(["Continent", "Year"]).mean().reset_index()
    print(cont_avg_df)

    # Plot line graphs by Continent
    plot_data(cont_avg_df, "Year", "Access to Electricity (% of population)", "Continent", 
              "Average Access to Electricity from 2000-2018 by Continent", xlim=2018)
    plot_data(cont_avg_df, "Year", "Renewable Electricity Output (% of total electricity output)", "Continent", 
              "Average Renewable Electricity Output from 2000-2018 by Continent", xlim=2015, ylim=100)
    plot_data(cont_avg_df, "Year", "Renewable Energy Use (%)", "Continent", 
              "Average Renewable Energy Use from 2000-2018 by Continent", xlim=2018, ylim=100)
    plot_data(cont_avg_df, "Year", "Annual CO2 Emissions (Mt)", "Continent", 
              "Average Annual CO2 Emissions from 2000-2018 by Continent", xlim=2018, ylim=500)

    # Group and aggregate data by Income Classification
    inc_avg_df = df.groupby(["Income Classification", "Year"]).mean().reset_index()
    print(inc_avg_df)

    # Plot bar graphs by Income Classification
    plot_data(inc_avg_df, "Year", "Access to Electricity (% of population)", "Income Classification", 
              "Average Access to Electricity from 2000-2018 by Income Classification", xlim=2018)
    plot_data(inc_avg_df, "Year", "Renewable Electricity Output (% of total electricity output)", "Income Classification", 
              "Average Renewable Electricity Output from 2000-2018 by Income Classification", xlim=2015, ylim=100)
    plot_data(inc_avg_df, "Year", "Renewable Energy Use (%)", "Income Classification", 
              "Average Renewable Energy Use from 2000-2018 by Income Classification", xlim=2018, ylim=100)
    plot_data(inc_avg_df, "Year", "Annual CO2 Emissions (Mt)", "Income Classification", 
              "Average Annual CO2 Emissions from 2000-2018 by Income Classification", xlim=2018, ylim=500)

if __name__ == "__main__":
    file_path = "/workspaces/World-Sustainability-Dataset/WorldSustainabilityDataset.csv"
    main()