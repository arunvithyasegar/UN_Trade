# Trade Data Analysis Dashboard

An interactive dashboard for analyzing global trade export data, built with Streamlit.

## Features

- Export growth trend analysis
- Country classification based on growth and volatility
- Statistical analysis and distribution visualization
- Interactive filtering and comparison tools

## How to Run Locally

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Data Requirements

The dashboard expects a CSV file with the following columns:
- `refYear`: Year of the trade data
- `reporterDesc`: Country name
- `primaryValue` and/or `fobvalue`: Export values
- `partnerDesc`: Trade partner (filtered for "World")
- `flowDesc`: Trade flow direction (filtered for "Export")

## Live Demo

The dashboard is deployed at: [your-streamlit-cloud-url]