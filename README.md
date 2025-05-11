# 📦 UN Comtrade Trade Data Analysis: Electrical Machinery Exports (2016–2024)

> **Note**: This project was developed as part of the **Guidance Tamil Nadu BIU Team Assignment Task 1**, focusing on trade data analysis and visualization.

---

## 📊 Project Overview

This project analyzes global export trends of electrical machinery and equipment (HS Code 85) using annual data from the UN Comtrade Database spanning 2016 to 2024. It encompasses data preparation, growth trend analysis, volatility assessment, statistical evaluation, and forecasting to identify high-growth and volatile exporters.

---

## 📁 Dataset Details

* **Source**: [UN Comtrade Database](https://comtradeplus.un.org/)
* **Commodity**: HS Code 85 – Electrical machinery and equipment
* **Trade Flow**: Exports
* **Reporter**: All countries
* **Partner**: World
* **Frequency**: Annual
* **Period**: 2016–2024
* **Measure**: USD([Investing in Tamil Nadu][1])

---

## 🧰 Project Structure

```
UN_Trade/
├── data/
│   └── raw/
│       └── UN_Comtrade_HS85_Exports_2016_2024.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_growth_trend_analysis.ipynb
│   ├── 03_volatility_classification.ipynb
│   └── 04_statistical_forecasting_analysis.ipynb
├── outputs/
│   ├── cleaned_data.csv
│   ├── visualizations/
│   └── reports/
├── README.md
└── requirements.txt
```

---

## ✅ Tasks & Methodology

### Task 1: Data Preparation

* **Import**: Load the dataset in CSV format using Python.
* **Clean**: Handle missing values, standardize country names, and ensure data consistency.
* **Filter**: Retain countries with export values exceeding \$500 million in 2024.
* **Export**: Save the cleaned dataset for subsequent analyses.

### Task 2: Growth Trend Analysis

* **Compute**: Calculate year-on-year (YoY) growth rates for each country from 2016 to 2024.
* **Rank**: Determine average annual growth rates and rank countries accordingly.
* **Visualize**: Generate line plots showcasing export trends for the top 3 countries.

### Task 3: Volatility & Classification

* **Analyze**: Compute the standard deviation of export values to assess volatility.
* **Identify**: Highlight the top 10 most volatile exporters.
* **Compare**: Contrast volatility with average growth rates.
* **Classify**: Categorize countries into:

  * Stable High-Growth
  * Volatile High-Growth
  * Stable Low-Growth
  * Volatile Low-Growth


### Task 4: Statistical & Forecasting Analysis

* **Distribution**: Plot a histogram of average annual growth rates with an overlaid normal distribution curve.
* **Performance Segments**: Use percentiles to identify top 10% (high-growth) and bottom 10% (underperformers).
* **Visualize**: Highlight performance segments on the histogram and create boxplots to display distributions and outliers.

---

## 📊 Sample Visualizations

* **Line Plots**: Export trends over time for leading countries.
* **Bar Charts**: Volatility comparisons among top exporters.
* **Quadrant Charts**: Country classifications based on growth and volatility.
* **Histograms & Boxplots**: Distribution analyses of growth rates.

---

## 🛠️ Technologies Used

* **Programming Language**: Python 3.9+
* **Libraries**:

  * Data Manipulation: `pandas`, `numpy`
  * Visualization: `matplotlib`, `seaborn`, `plotly`
  * Statistical Analysis: `scipy`, `statsmodels`
  * Notebook Environment: `Jupyter Notebook`

---

## 🚀 Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/arunvithyasegar/UN_Trade.git
   cd UN_Trade
   ```

2. **Install Dependencies**:

   Ensure you have Python 3.9+ installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Notebooks**:

   Navigate to the `notebooks/` directory and open the Jupyter notebooks in sequence to execute the analyses.

---

## 📅 Live Dashboard

Explore the interactive Streamlit dashboard showcasing key insights from the analysis:

🔗 [Electronic Trade Dashboard](https://electronictradedashboard.streamlit.app/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

For questions or feedback, please reach out to [Arun Vithyasegar](mailto:arunvithyasegar@example.com).

---

## 🌐 References

* [UN Comtrade Database](https://comtradeplus.un.org/)
* [World Integrated Trade Solution (WITS)](https://wits.worldbank.org/)
* [HS Nomenclature 2022 Edition](https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2022-edition/hs-nomenclature-2022-edition.aspx)

---

\*Note: This project utilizes publicly available data from the UN Comtrade Database, which stores over 1 billion trade data records from 1962 onwards. Over 140 reporter countries provide the United Nations Statistics Division with their annual international trade statistics detailed by commodities and partner countries. These data are subsequently transformed into the United Nations Statistics Division standard format with consistent coding and valuation using the UN/OECD CoprA internal processing system.\* ([data.un.org](https://data.un.org/Data.aspx?d=ComTrade&f=_l1Code%3A85%3BcmdCode%3A852810&q=television&utm_source=chatgpt.com))

---

[1]: https://investingintamilnadu.com/DIGIGOV/StaticAttachment?AttachmentFileName=%2Fpdf%2Fpoli_noti%2FRFP2.pdf&utm_source=chatgpt.com "[PDF] GUIDANCE - Invest Tamil Nadu"
[2]: https://www.scribd.com/document/599079021/1-Assignment-1-Guidance-2?utm_source=chatgpt.com "1 - Assignment 1 Guidance | PDF | Business | Computers - Scribd"
