# UN Comtrade Trade Data Analysis: Electrical Machinery Exports (2016–2024)

This project analyzes global export trends of electrical machinery and equipment (HS Code 85) using annual data from the UN Comtrade Database spanning 2016 to 2024. It encompasses data preparation, growth trend analysis, volatility assessment, statistical evaluation, and forecasting to identify high-growth and volatile exporters.

---

## 📁 Dataset Overview

* **Source**: [UN Comtrade Database](https://comtradeplus.un.org/)
* **Commodity**: HS Code 85 – Electrical machinery and equipment
* **Trade Flow**: Exports
* **Reporter**: All countries
* **Partner**: World
* **Frequency**: Annual
* **Period**: 2016–2024
* **Measure**: USD([UN Comtrade][1], [OECD][2], [GOV.UK][3])

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
* **Visualize**: Present classifications using quadrant charts or labeled tables.([World Integrated Trade Solution][4], [GOV.UK][3])

### Task 4: Statistical & Forecasting Analysis

* **Distribution**: Plot a histogram of average annual growth rates with an overlaid normal distribution curve.
* **Performance Segments**: Use percentiles to identify top 10% (high-growth) and bottom 10% (underperformers).
* **Visualize**: Highlight performance segments on the histogram and create boxplots to display distributions and outliers.

---

## 📊 Sample Visualizations

* **Line Plots**: Export trends over time for leading countries.
* **Bar Charts**: Volatility comparisons among top exporters.
* **Quadrant Charts**: Country classifications based on growth and volatility.
* **Histograms & Boxplots**: Distribution analyses of growth rates.([OECD][2], [World Customs Organization][5])

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

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

For questions or feedback, please reach out to [Arun Vithyasegar](mailto:arunvithyasegar@example.com).

---

## 🌐 References

* [UN Comtrade Database](https://comtradeplus.un.org/)
* [World Integrated Trade Solution (WITS)](https://wits.worldbank.org/)
* [HS Nomenclature 2022 Edition](https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2022-edition/hs-nomenclature-2022-edition.aspx)([UN Comtrade][6], [World Integrated Trade Solution][7], [World Customs Organization][5])

---

*Note: This project utilizes publicly available data from the UN Comtrade Database, which stores over 1 billion trade data records from 1962 onwards. Over 140 reporter countries provide the United Nations Statistics Division with their annual international trade statistics detailed by commodities and partner countries. These data are subsequently transformed into the United Nations Statistics Division standard format with consistent coding and valuation using the UN/OECD CoprA internal processing system.* ([UNdata][8])

---

[1]: https://comtradeplus.un.org/TradeFlow?AggregateBy=none&BreakdownMode=plus&CommodityCodes=TOTAL&Flows=X&Frequency=A&Partners=0&Reporters=842&period=all&utm_source=chatgpt.com "Trade Data - UN Comtrade"
[2]: https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/12/risks-and-resilience-in-global-trade_c8a001ff/1c66c439-en.pdf?utm_source=chatgpt.com "[PDF] Risks and Resilience in Global Trade - OECD"
[3]: https://assets.publishing.service.gov.uk/media/68137af670b095d0d7011854/japan-trade-and-investment-factsheet-2025-05-02.pdf?utm_source=chatgpt.com "[PDF] Trade and Investment Factsheet - Japan - GOV.UK"
[4]: https://wits.worldbank.org/trade/country-byhs6product.aspx?lang=en&utm_source=chatgpt.com "Trade Statistics by Product (HS 6-digit)"
[5]: https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2022-edition/hs-nomenclature-2022-edition.aspx?utm_source=chatgpt.com "HS Nomenclature 2022 edition - World Customs Organization"
[6]: https://comtradeplus.un.org/TradeFlow?utm_source=chatgpt.com "Trade Data - UN Comtrade"
[7]: https://wits.worldbank.org/?utm_source=chatgpt.com "World Integrated Trade Solution (WITS) | Data on Export, Import ..."
[8]: https://data.un.org/Data.aspx?d=ComTrade&f=_l1Code%3A85%3BcmdCode%3A852810&q=television&utm_source=chatgpt.com "record view | Trade of goods, US$, HS, 85 Electrical machinery and ..."
