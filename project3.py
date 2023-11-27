"""World Sustainability Dataset"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

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

    # Convert "Year" to str
    df["Year"] = df["Year"].astype("str")

    return df

def plot_data(data, x, y, hue, title):
    # Plot line graph for historical data
    g = sns.relplot(kind="line", data=data, x=x, y=y, hue=hue, aspect=1.75).set(title=title)
    g.set_xticklabels(rotation=30)
    plt.show()

def train_model(X, y):
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=['Continent'])

    # Scale features using RobustScaler (robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Train a Random Forest regressor with cross-validation
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fine-tune hyperparameters (example: adjust n_estimators and max_depth)
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Print best hyperparameters
    print(f'Best Hyperparameters: {grid_search.best_params_}')

    # Evaluate the best model using cross-validated MSE scores
    mse_scores = -cross_val_score(best_model, X_scaled, y, scoring='neg_mean_squared_error', cv=5)
    print(f'Cross-validated MSE scores: {mse_scores}')
    print(f'Mean MSE: {mse_scores.mean()}')

    # Train the best model on the entire dataset
    best_model.fit(X_scaled, y)

    return best_model

def predict_future(model, future_data):
    # One-hot encode categorical features
    future_data_encoded = pd.get_dummies(future_data, columns=['Continent'])

    # Make predictions for future years
    predictions = model.predict(future_data_encoded)

    return pd.DataFrame({'Continent': future_data['Continent'], 'Year': future_data['Year'], 'Predicted': predictions})

def main():
    file_path = "/workspaces/World-Sustainability-Dataset/WorldSustainabilityDataset.csv"
    df = read_data(file_path)
    df = clean_data(df)

    # Convert "Year" to int for machine learning
    df["Year"] = df["Year"].astype(int)

    # Group and aggregate data
    avg_df = df.groupby(["Continent", "Year"]).agg({
        "Renewable Energy Use (%)": "mean",
        "Annual CO2 Emissions (Mt)": "mean"
    }).reset_index()

    # Plot graphs for historical data
    plot_data(avg_df, "Year", "Renewable Energy Use (%)", "Continent", "Average World Renewable Energy Use from 2000-2018 by Continent")
    plot_data(avg_df, "Year", "Annual CO2 Emissions (Mt)", "Continent", "Average Annual World CO2 Emissions from 2000-2018 by Continent")

    # Train models
    renewable_energy_model = train_model(df[['Continent', 'Year']], df['Renewable Energy Use (%)'])
    co2_emissions_model = train_model(df[['Continent', 'Year']], df['Annual CO2 Emissions (Mt)'])

    # Predict future values until 2050
    future_years = list(range(2019, 2050))
    unique_continents = df['Continent'].unique()

    future_data_renewable = pd.DataFrame(list(product(unique_continents, future_years)), columns=['Continent', 'Year'])
    future_data_co2 = pd.DataFrame(list(product(unique_continents, future_years)), columns=['Continent', 'Year'])

    renewable_energy_predictions = predict_future(renewable_energy_model, future_data_renewable)
    co2_emissions_predictions = predict_future(co2_emissions_model, future_data_co2)

    # Plot graphs for historical and predicted data
    plot_data(pd.concat([avg_df, renewable_energy_predictions]), "Year", "Renewable Energy Use (%)", "Continent", "Renewable Energy Use from 2000-2050 by Continent")
    plot_data(pd.concat([avg_df, co2_emissions_predictions]), "Year", "Annual CO2 Emissions (Mt)", "Continent", "Annual CO2 Emissions from 2000-2050 by Continent")

if __name__ == "__main__":
    main()