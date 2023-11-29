"""World Sustainability Dataset"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from IPython.display import display


def read_data(file_path):
    # Read dataset from CSV file
    return pd.read_csv(file_path)


def clean_data(df):
    # Select relevant columns
    df = df[
        [
            "Country Name",
            "Year",
            "Access to electricity (% of population) - EG.ELC.ACCS.ZS",
            "Renewable electricity output (% of total electricity output) - EG.ELC.RNEW.ZS",
            "Renewable energy consumption (% of total final energy consumption) - EG.FEC.RNEW.ZS",
            "Annual production-based emissions of carbon dioxide (CO2), measured in million tonnes",
            "Continent",
            "Income Classification (World Bank Definition)",
        ]
    ]

    # Rename columns
    df.columns = [
        "Country",
        "Year",
        "Access to Electricity (% of population)",
        "Renewable Electricity Output (% of total electricity output)",
        "Renewable Energy Use (%)",
        "Annual CO2 Emissions (Mt)",
        "Continent",
        "Income Classification",
    ]

    # Reorder columns
    df = df[
        [
            "Continent",
            "Country",
            "Income Classification",
            "Year",
            "Access to Electricity (% of population)",
            "Renewable Electricity Output (% of total electricity output)",
            "Renewable Energy Use (%)",
            "Annual CO2 Emissions (Mt)",
        ]
    ]

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Fill missing Continent for Timor-Leste
    df.loc[df["Country"] == "Timor-Leste", "Continent"] = "Asia"
    df.loc[
        df["Country"] == "Timor-Leste", "Income Classification"
    ] = "Lower-middle income"

    # Convert "Year" to int
    df["Year"] = df["Year"].astype("int")

    return df

def save_to_csv(df, file_name):
    df.to_csv(f"output/{file_name}.csv")

def plot_data(df, x, y, hue, title, xlim=None, ylim=None):
    # Filter the dataframe if xlim is specified
    if xlim:
        df = df[df[x] <= xlim]

    # Plot historical data
    g = sns.relplot(kind="line", data=df, x=x, y=y, hue=hue, aspect=1.75).set(
        title=title
    )

    # Set x-axis limit if specified
    if xlim:
        g.set(xlim=(df[x].min(), xlim))

    # Set y-axis limit if specified
    if ylim:
        g.set(ylim=(df[y].min(), ylim))

    g.set_xticklabels(rotation=30)
    plt.show()


def train_model(X, y):
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=["Continent"])

    # Scale features using RobustScaler (robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Train a Random Forest regressor with cross-validation
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fine-tune hyperparameters
    param_grid = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_scaled, y)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Print best hyperparameters
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Evaluate the best model using cross-validated MSE scores
    mse_scores = -cross_val_score(
        best_model, X_scaled, y, scoring="neg_mean_squared_error", cv=5
    )
    print(f"Cross-validated MSE scores: {mse_scores}")
    print(f"Mean MSE: {mse_scores.mean()}")

    # Train the best model on the entire dataset
    best_model.fit(X_scaled, y)

    return best_model


def predict_future(model, future_data):
    # One-hot encode categorical features
    future_data_encoded = pd.get_dummies(future_data, columns=["Continent"])

    # Make predictions for future years
    predictions = model.predict(future_data_encoded)

    return pd.DataFrame(
        {
            "Continent": future_data["Continent"],
            "Year": future_data["Year"],
            "Predicted": predictions,
        }
    )


def main():
    # Read and clean data into a dataframe
    df = read_data(file_path)
    df = clean_data(df)
    print(df)

    # Summary and dimensions of dataframe
    print(df.describe())
    print(df.shape)

    # Create a table for statistics
    df_filtered = df.drop(columns=["Year"])
    df_filtered = df_filtered[df_filtered != 0].dropna()
    summary_stats = df_filtered.describe()
    styled_table = summary_stats.style.set_precision(2)
    display(styled_table)

    # Group and aggregate data by Continent
    cont_avg_df = df.groupby(["Continent", "Year"]).mean().reset_index()
    print(cont_avg_df)

    # Summary and dimensions of dataframe
    print(cont_avg_df.describe())
    print(cont_avg_df.shape)

    # Plot line graphs by Continent
    plot_data(
        cont_avg_df,
        "Year",
        "Access to Electricity (% of population)",
        "Continent",
        "Average Access to Electricity from 2000-2018 by Continent",
        xlim=2018,
    )
    plot_data(
        cont_avg_df,
        "Year",
        "Renewable Electricity Output (% of total electricity output)",
        "Continent",
        "Average Renewable Electricity Output from 2000-2018 by Continent",
        xlim=2015,
        ylim=100,
    )
    plot_data(
        cont_avg_df,
        "Year",
        "Renewable Energy Use (%)",
        "Continent",
        "Average Renewable Energy Use from 2000-2018 by Continent",
        xlim=2018,
        ylim=100,
    )
    plot_data(
        cont_avg_df,
        "Year",
        "Annual CO2 Emissions (Mt)",
        "Continent",
        "Average Annual CO2 Emissions from 2000-2018 by Continent",
        xlim=2018,
        ylim=500,
    )

    # Group and aggregate data by Income Classification
    inc_avg_df = df.groupby(["Income Classification", "Year"]).mean().reset_index()
    print(inc_avg_df)

    # Summary and dimensions of dataframe
    print(inc_avg_df.describe())
    print(inc_avg_df.shape)

    # Plot line graphs by Income Classification
    plot_data(
        inc_avg_df,
        "Year",
        "Access to Electricity (% of population)",
        "Income Classification",
        "Average Access to Electricity from 2000-2018 by Income Classification",
        xlim=2018,
    )
    plot_data(
        inc_avg_df,
        "Year",
        "Renewable Electricity Output (% of total electricity output)",
        "Income Classification",
        "Average Renewable Electricity Output from 2000-2018 by Income Classification",
        xlim=2015,
        ylim=100,
    )
    plot_data(
        inc_avg_df,
        "Year",
        "Renewable Energy Use (%)",
        "Income Classification",
        "Average Renewable Energy Use from 2000-2018 by Income Classification",
        xlim=2018,
        ylim=100,
    )
    plot_data(
        inc_avg_df,
        "Year",
        "Annual CO2 Emissions (Mt)",
        "Income Classification",
        "Average Annual CO2 Emissions from 2000-2018 by Income Classification",
        xlim=2018,
        ylim=500,
    )

    # Split the data into features (X) and target variable (y)
    X = df[["Continent", "Year"]]
    y = df["Annual CO2 Emissions (Mt)"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    co2_emissions_model = train_model(X_train, y_train)

    # Predict future values until 2050
    future_years = list(range(2019, 2050))
    unique_continents = df["Continent"].unique()
    future_data_co2 = pd.DataFrame(
        list(product(unique_continents, future_years)), columns=["Continent", "Year"]
    )
    co2_emissions_predictions = predict_future(co2_emissions_model, future_data_co2)

    # Plot graphs for historical and predicted data
    plot_data(
        pd.concat([cont_avg_df, co2_emissions_predictions]),
        "Year",
        "Annual CO2 Emissions (Mt)",
        "Continent",
        "Annual CO2 Emissions from 2000-2050 by Continent",
    )

    # Save as CSV
    save_to_csv(df, "df")
    save_to_csv(df.describe(), "df_stats")
    save_to_csv(cont_avg_df, "cont_avg")
    save_to_csv(inc_avg_df, "inc_avg")

if __name__ == "__main__":
    file_path = (
        "/workspaces/World-Sustainability-Dataset/WorldSustainabilityDataset.csv"
    )
    main()
