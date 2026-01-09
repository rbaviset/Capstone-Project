# Capstone-Project : AI driven Sourcing Strategy

Repository for Capstone Project: https://github.com/rbaviset/Capstone-Project
  
# 1. Business Understanding
The increasing complexity and volatility of global supply chains subject the sourcing function to constant, high-impact risks, including sudden supplier financial failures, geopolitical trade disruptions, and chronic supply shortages.
Traditional sourcing strategy process is constrained by its inability to synthesize millions of dynamic, heterogeneous data points across global supplier networks. Sourcing managers are overwhelmed trying to manually integrate internal operational performance data with fragmented external intelligence on technology roadmaps, financial health, geopolitical instability, and ESG goals. This results in delayed strategy updates and decisions based on incomplete, outdated information.
AI transforms the sourcing strategy process from a fragmented, data-intensive effort into a holistic, real-time data-driven approach. By leveraging Natural Language Processing (NLP) and Machine Learning (ML), AI synthesizes all internal and external data into a dynamic Knowledge Graph and real-time Supplier Health Score. This shift allows leaders to focus on  strategic discussions with Business Units for enabling supply base resilience and optimized Total Cost of Ownership.

# 2. Data Understanding

## 2.1 Data Understanding
The historical supplier dataset integrates multiple internal and external dimensions critical for sourcing strategy. The data structure consists of a) 10 years of memory sourcing data from 2015 to 2024 b) 3 memory suppliers : Micron, Samsung, SK Hynix c) ~20 core features per supplier-year d) Mixed numeric, text, % formats, and categorical fields

### Supplier Feature Categories & Definitions

| Category            | Description                                                                                          | Features Included                                                                 |
|--------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Operational**    | Cost, Quality, and Delivery performance from ERP & internal systems                                   | Cost Savings (%), PPV, QP (Quality Performance), QR (Quality Rating), Lead Time Attainment |
| **Financial**      | Supplier financial health metrics from balance sheet, income statement & cash flow statements        | Revenue, COGS, Cash Flow, Gross Margin %, Debt-to-Equity Ratio                   |
| **Geopolitical**   | Exposure to political instability, trade restrictions, natural disaster risks                         | Geo Risk Index, Tariff Risk, Chip Shortage Impact                                |
| **Compliance (ESG)** | Environmental, Social & Governance performance from Supplier Sustainability reports                   | Renewable Energy Usage, Plastic Recycling %, Human Rights Compliance Score        |
| **Technology Roadmap** | Technology alignment with DRAM node requirements and future memory/packaging capabilities         | DDR Generation Support, Node Parity                                               |


## 2.2 Data Quality Findings
EDA and profiling uncovered several issues that required systematic cleanup:

| Data Quality Issue        | Examples                                                                 | Impact on Modeling                                      | Required Fix Applied                                     |
|---------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------|
| **Inconsistent Supplier Naming** | “Micron Technology, Inc.”, “Micron Tech”, “Micron” → Same supplier            | Encoder breaks → unseen labels during prediction         | Normalized names to {Micron, Samsung, SK Hynix}          |
| **Mixed Numeric Formats** | Percentages stored as “35%”, values with commas “1,234,000”    | Non-numeric data blocks scaling + model ingestion        | Removed % and commas      |
| **Financial Unit Variability** | Millions, thousands, KRW billions in SK Hynix PDF                               | Prevents cross-supplier comparison                       | Converted all financials → Thousands USD , unit normalization → float values |
| **Missing Values Across Years** | Micron Cash Flow missing early years; risk metrics missing in rows              | Model cannot learn complete trends                       | Imputed / validated missing values and aligned timelines |
| **Constant Features**     | Node Parity always = 1.0 for all suppliers                                           | Zero predictive value → adds noise                       | Retained for completeness but flagged as non-informative |
| **Highly Skewed Distributions** | Samsung Cash Flow magnitude >> others; Tariff Risk spikes only in specific years | Risk of model bias & instability                          | StandardScaler + flagged for feature weighting           |
| **Risk Metrics Baseline Uniformity** | Geo Risk, Tariff Risk sometimes identical across suppliers                  | Reduces discriminatory power                              | Retained but documented for model interpretation         |
| **Temporal Gaps**         | Some metrics flat for long periods (e.g., Chip Shortage Impact)                      | Misleading time-series trends                            | Smoothing applied in visualization only                  |


## 2.3 EDA Insights

| Feature                  | Micron                           | Samsung                       | SK Hynix                     | Strategic Interpretation                            |
| ------------------------ | ---------------------------------| ------------------------------| -----------------------------| --------------------------------------------------- |
| **Cash Flow**            | Very low                         | Extremely high                | Very low                     | Samsung has strongest liquidity                     |
| **Gross Margin %**       | Highly volatile (18%–48%)        | Stable (37%–48%)              | Volatile (17%–59%)           | Samsung is financially most stable; SKH is cyclical |
| **DDR Gen Support**      | Gradual improvement              | Strong early adoption         | Fastest technology ramp      | SK Hynix leads technological advancement            |
| **Geo Risk**             | Highest                          | Lowest                        | Medium                       | Micron most sensitive to geopolitical concentration |
| **Tariff Risk**          | Highest spikes during trade war  | Lowest spikes                 | Medium                       | Micron most exposed to US–China trade cycles        |
| **Chip Shortage Impact** | High                             | Moderate                      | Very high                    | All suppliers impacted; SKH most vulnerable         |
| **Node Parity**          | Always at parity                 | Always at pari                | Always at pari               | No supplier lags                                    |
| **ESG**                  | Moderate                         | Moderate                      | Moderate                     | ESG does not differ drastically among suppliers     |

# 3. Data Preperation
These transformations were implemented during final EDA and will be reused in model training.

| Data Preparation Task | Description | Artifacts (Input → Output) |
|-----------------------|-------------|-----------------------------|
| **Step 1 — Load Raw Historical Dataset** | Load the consolidated 10-year dataset containing operational, financial, geopolitical, and ESG variables. | `historical_supplier_raw_data.csv` → in-memory |
| **Step 2 — Supplier Name Normalization** | Standardize inconsistent naming (e.g., “Micron Technology, Inc.” → “Micron”) for grouping, merging, and encoding. | `historical_supplier_raw_data.csv` → `historical_features_clean.csv` |
| **Step 3 — Numeric Cleaning & Type Conversion** | Strip `%`, commas, text units; convert all numeric columns into floats. Ensures clean continuous variables for modeling. | `historical_features_clean.csv` → `historical_features_clean.csv` |
| **Step 4 — Financial Unit Standardization** | Convert financial values (e.g., KRW billions, millions) into a consistent unit: **Thousands USD**. | `historical_features_clean.csv` → `historical_features_clean.csv` |
| **Step 5 — Handling Missing Values** | Impute missing operational, financial, risk, and ESG data using interpolation or business rules. | `historical_features_clean.csv` → `historical_features_clean.csv` |
| **Step 6 — Remove Constant / Irrelevant Features** | Drop zero-variance fields (e.g., Node Parity), redundant categorical columns, and unused metadata. | `historical_features_clean.csv` → `historical_features_clean.csv` |
| **Step 7 — Final Schema Validation** | Validate datatypes, fiscal-year ordering, supplier completeness, and feature consistency. | `historical_features_clean.csv` → `historical_features_clean.csv` |
| **Step 8 — Model-1A: Generate PSL Target (Unsupervised)** | Use KMeans clustering on strategic + financial signals to derive PSL category labels: **Preferred / Developing / Limited**. | `historical_features_clean.csv` → `historical_features_with_psl.csv` |
| **Step 9 — Model-2A: Generate Allocation Target (Unsupervised)** | Compute composite score using operational + ESG + risk measures → normalize supplier allocation to **sum = 100%**. | `historical_features_with_psl.csv` → `historical_features_with_allocation.csv` |
| **Step 10 — Build Master Historical Dataset** | Combine all cleaned features + PSL labels + Allocation targets into a single master modeling dataset. | `historical_features_with_allocation.csv` → `historical_features_master.csv` |


# 4. Modeling
## 4.1 Supervised Supplier Categorization (Model-1B)
This model is a multiclass classification model predicting PSL categories: 1)Preferred 2)Developing 3)Limited. Train the supervised classification model that learns historical supplier classification from engineered Financial, Risk, and Technology roadmap features.

| Component           | Details                                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Algorithm**       | XGBoost Classifier (multi-class)                                                                                                      |
| **Input Features**  | Financial(Gross Margin%, Cash Flow, Debt Equity Ratio), Risk(Geo, Tariff, Chip Shortage), Technology (DDR Gen, Node Parity) |

| **Target Variable** | `PSL_code` (from PSL_status produced by Model-1A unsupervised logic)                                                                             |
| **# of Classes**    | 3 (Preferred, Developing, Limited)                                                                                                    |
| **Data Source**     | `historical_features_with_allocation.csv`                                                                                             |

## 4.2 Supervised Spend Allocation Prediction (Model-2B)
This model is a regression model predicting continuous values: Supplier allocation percentage(%). Train the supervised regression model to learn optimal supplier allocation percentages based on historical Performance, ESG, and PSL classification.

| Component           | Details                                                 |
| ------------------- | ------------------------------------------------------- |
| **Algorithm**       | XGBoost Regressor                                       |
| **Input Features**  | PSL_predicted, Operational(Cost Savings, PPV, QP, QR, Lead Time Attainment), ESG(Carbon Emission Intensity, Renewable Energy Usage, Plastic Recycle, Human Rights Compliance)   |
| **Target Variable** | `allocation_percent` (from Model-2A unsupervised logic) |
| **Normalization**   | Final predictions normalized per fiscal year → 100%     |
| **Data Source**     | `supplier_categorization_predictions.csv`               |


# 5. Evaluation
## 5.1 Supervised Supplier Categorization (Model-1B)

### Performance Summary
| Metric                      | Result                                 |
| --------------------------- | -------------------------------------- |
| **Accuracy**                | **100%**                               |
| **Precision / Recall / F1** | **1.00 / 1.00 / 1.00** for all classes |
| **Confusion Matrix**        | All samples correctly classified       |

### Evaluation Interpretation
**1.** The perfect score is expected because the dataset is small (6×3 = 18 samples). Classes are strongly separable due to engineered target clusters from Model-1A. PSL categories follow consistent financial + operational patterns.

**2.** In summary, the model generalizes well because PSL categories are stable over time and strongly tied to distinguishable features (Samsung = strong cash flow + GM%, Micron & SK Hynix mid-range). PSL predictions for future years are expected to remain consistent since feature distributions follow the same ranges.

## 5.2 Supervised Spend Allocation Prediction (Model-2B)

### Performance Summary
| Metric       | Result                                         |
| ------------ | ---------------------------------------------- |
| **R² Score** | **0.848**                                      |
| **MAE**      | **0.763**                                      |
| **RMSE**     | **1.808**                                      |
| **Bias**     | None observed (predictions centered correctly) |

### Evaluation Interpretation
**1.** The model explains ~85% of the variance in optimal allocation. Errors are small: < 2% deviation from target allocation. Final allocations always sum to 100% due to normalization logic.

**2.** Insummary, The model correctly learned allocation patterns across suppliers and fiscal years. Allocation behavior strongly follows PSL categories, operational excellence, and risk stability. Due to the consistent patterns in historical data (Samsung = Preferred = higher allocation), the model reproduces allocations consistently and accurately.

# 6. Deployment

## FY25 Supplier Feature Extraction Pipeline
This pipeline prepares FY25 supplier financial and operational features for inference using the trained PSL (Model-1B) and Spend Allocation (Model-2B) models. It is part of the Deployment phase and separate from the historical data preparation used for model training.

| **Step**   | **Task**                                | **Description**                                                                                    | **Input → Output**                                                                                                        |
| ---------- | --------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Load Raw Financial Statements (PDF)** | Extract FY25 P&L data using Gemini LLM + pdfplumber.                                               | `Samsung 2025_con_quarter01_all.pdf`<br>`Micron Technology Inc_Fiscal2025.pdf`<br>`SK Hynix 3Q2025.pdf` → Structured JSON |
| **2** | **LLM Financial Parsing**               | Use Pydantic schema to extract revenue, COGS, gross margin %, cash flow, debt/equity, scale units. | Raw text → Parsed financial rows                                                                                          |
| **3** | **Supplier Name Normalization**         | Fuzzy-map LLM supplier names to official names (`Samsung`, `Micron`, `SK Hynix`).                  | Raw supplier names → Normalized supplier                                                                                  |
| **4** | **Scale & Currency Standardization**    | Convert KRW billions / millions into **Thousands USD** using mapping rules.                        | Revenue, COGS, CashFlow → `Thousands USD`                                                                                 |
| **5** | **Load FY25 Non-Financial Data**        | Import operational, ESG, and risk indicators from CSV.                                             | `FY25_ServerDRAM_NonFinancial.csv`                                                                                        |
| **6** | **Supplier Name Cleanup (CSV)**         | Normalize CSV supplier names with fuzzy matching.                                                  | Raw CSV names → Clean names                                                                                               |
| **7** | **Merge Financial + Non-Financial**     | Join datasets on supplier, ensuring a **single fiscal_year column**.                               | PDFs + CSV → `FY25_SDRAM_feature_table.csv`                                                                               |
| **8** | **Export FY25 Feature Table**           | Final dataset used as input to PSL & Allocation prediction models.                                 | Output: `FY25_SDRAM_feature_table.csv`                                                                                    |

## FY25 Sourcing Strategy Inference Pipeline
The trained PSL classifier (Model-1B) and spend allocation regressor (Model-2B) from the historical modeling stage are applied to the FY25 dataset to generate the final PSL category and allocation recommendations for category managers.

# 7. Future Scope & Next Steps

1. **Real-Time Intelligence Ecosystem :** Transition from static document analysis to live ingestion of market signals. Integrate dynamic APIs (Financial health, Geopolitical risk feeds, and ESG disclosures) to predict disruptions before they impact the supply chain.
2. **Autonomous Reasoning Engine (LangGraph) :** Upgrade to a Multi-Agent Architecture where specialized AI agents (e.g., Financial Analyst, Technical Sourcing Manager) collaborate to solve complex queries. This improves reasoning depth and ensures policy compliance with minimal manual oversight.
3. **Strategic Category Expansion :** Leverage the existing DRAM framework to rapidly scale coverage into critical silicon categories, specifically GPUs and ASICs.
4. **Defensible Sourcing Intelligence :** Deploy Advanced RAG to synthesize insights and generate supplier white papers for Strategic Decision Support. Enable stakeholders to generate data-backed narratives that justify multi-million dollar spend allocations in natural language.


