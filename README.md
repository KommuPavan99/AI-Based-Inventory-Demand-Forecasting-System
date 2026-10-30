# 📦 AI-Based Inventory Demand Forecasting System

An intelligent Machine Learning-based inventory management and demand forecasting system designed to predict future product demand, monitor stock levels, identify stockout risks, and generate smart replenishment recommendations.

## 🌟 Project Overview

The **AI-Based Inventory Demand Forecasting System** uses historical inventory data and Machine Learning techniques to estimate future product demand and support efficient inventory management.

The system provides an interactive dashboard where users can explore inventory datasets, analyse historical demand, view future demand forecasts, monitor inventory conditions, and obtain replenishment suggestions.

It combines a Python-based Machine Learning model with a FastAPI backend and a web-based frontend to provide an integrated inventory intelligence solution.

## 🎯 Objectives

- Predict future product demand using Machine Learning.
- Analyse historical inventory and demand patterns.
- Monitor stock availability and inventory levels.
- Identify products with potential stockout risks.
- Calculate reorder points and suggested order quantities.
- Provide visual insights through interactive graphs.
- Simplify inventory analysis through a user-friendly dashboard.
- Support better inventory planning and decision-making.

## ✨ Key Features

### 1. Inventory Dashboard
- Displays important inventory summary information.
- Shows total demand and product information.
- Provides an overview of inventory conditions.
- Presents key inventory indicators through visual elements.

### 2. Dataset Explorer
- Displays historical inventory records in tabular format.
- Supports searching, filtering, and sorting.
- Provides pagination for navigating records.
- Displays dataset information and missing-value statistics.
- Supports CSV and Excel data exports.

### 3. Graphs and Analysis
- Visualizes historical demand patterns.
- Displays forecast demand.
- Compares demand across products.
- Helps identify demand trends and inventory patterns.
- Presents information through interactive charts.

### 4. Demand Forecasting
- Uses a trained Random Forest Machine Learning model.
- Predicts future product demand.
- Supports 7-day, 14-day, and 30-day forecasting.
- Uses historical demand patterns and engineered features.
- Helps organizations plan inventory requirements.

### 5. Replenishment Engine
- Calculates net available stock.
- Determines reorder points.
- Calculates safety stock requirements.
- Generates suggested order quantities.
- Considers lead time and review periods.
- Helps reduce stockout risks and improve stock planning.

### 6. Product Daily Life Analysis
- Provides product-level demand analysis.
- Displays total demand and average daily demand.
- Helps understand individual product demand behaviour.
- Supports product-specific inventory decisions.

### 7. Voice Assistant
- Provides a conversational interface for inventory-related queries.
- Supports voice and text input.
- Allows users to ask questions about inventory information.
- Helps users access demand forecasts and replenishment information.

## 🧠 Machine Learning Methodology

The system uses a **Random Forest Regression** model to predict future product demand.

### Workflow

1. Dataset Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Saving
7. Demand Prediction

### Features Used

- Date-based features
- Day of the week
- Weekend information
- Trend information
- Previous demand (`lag_1`)
- Previous seven-day demand (`lag_7`)
- Seven-day rolling average
- Twenty-eight-day rolling average

### Forecasting Periods

| Forecast Period | Purpose |
|---|---|
| 7 Days | Short-term demand planning |
| 14 Days | Medium-term inventory planning |
| 30 Days | Monthly inventory planning |

The trained model is saved using Joblib and loaded by the backend for generating predictions.

## 🏗️ System Architecture

```text
             INVENTORY DATASET
                (Excel / CSV)
                     |
                     v
              DATA PREPROCESSING
                     |
                     v
             FEATURE ENGINEERING
                     |
                     v
              MACHINE LEARNING
               (Random Forest)
                     |
                     v
               TRAINED MODEL
                 (Joblib)
                     |
                     v
              FASTAPI BACKEND
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Forecast   Replenishment  Inventory
       API        Engine         Analysis
          |          |          |
          +----------+----------+
                     |
                     v
               WEB FRONTEND
              (HTML / CSS / JS)
                     |
                     v
             INVENTORY DASHBOARD
```

## 🛠️ Technologies Used

### Programming Language
- Python
- JavaScript

### Machine Learning and Data Processing

| Technology | Purpose |
|---|---|
| Pandas | Data processing and analysis |
| NumPy | Numerical calculations |
| Scikit-learn | Machine Learning model training and evaluation |
| OpenPyXL | Reading and working with Excel files |
| Joblib | Saving and loading the trained model |
| Matplotlib | Data visualization |

### Backend
- FastAPI – Backend API development
- Uvicorn – Running the FastAPI application

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

### Database and Storage
- SQLite
- Excel Dataset

### Deployment and Version Control
- Git
- GitHub
- Render – Backend deployment
- InfinityFree – Frontend hosting

## 📁 Project Structure

```text
AI-Based-Inventory-Demand-Forecasting-System/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── index.html
│
├── ml/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_demand.py
│   ├── demand_model.joblib
│   └── evaluation_metrics.json
│
├── data/
│   ├── dataset.xlsx
│   └── inventory.db
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

*Note: The structure above represents the main project components. Some generated files, datasets, or local database files may not be included in the GitHub repository.*

## 📊 Dataset Information

The project uses an Excel dataset containing historical inventory and demand information.

**Dataset Details:**

- Total Records: 1,000
- Total Columns: 17
- Number of Products: 5
- Data Format: Excel (`.xlsx`)
- Sheet Name: `Inventory_Data`

### Dataset Attributes

| Column | Description |
|---|---|
| Record_ID | Unique inventory record identifier |
| Date | Date of inventory record |
| Product_ID | Product identification |
| Product_Name | Product name |
| Category | Product category |
| Unit_Price | Price per unit |
| Promotion_Active | Promotion status |
| Stock_On_Hand | Available stock quantity |
| Demand_Units | Historical demand |
| Estimated_Revenue | Estimated revenue |
| Supplier | Supplier information |
| Region | Business region |
| Lead_Time_Days | Supplier lead time |
| Reorder_Level | Reorder threshold |
| Projected_Closing_Stock | Expected closing stock |
| Stock_Status | Current stock condition |
| Stockout_Risk | Potential stockout risk |

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or later
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/KommuPavan99/AI-Based-Inventory-Demand-Forecasting-System.git
```

### Step 2: Open the Project Folder

```bash
cd AI-Based-Inventory-Demand-Forecasting-System
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Backend

```bash
uvicorn backend.app:app --reload
```

If running the application directly from the backend directory, use the corresponding module path configured for your project.

### Step 6: Open the Application

Visit:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## 🔗 API Endpoints

The backend provides API endpoints for accessing inventory information and forecasting functionality.

| Endpoint | Purpose |
|---|---|
| `/api/health` | Checks backend service status |
| `/api/summary` | Provides inventory summary |
| `/api/products` | Retrieves product information |
| `/api/history` | Retrieves historical demand records |
| `/api/metrics` | Provides model evaluation metrics |
| `/api/predict` | Generates demand predictions and replenishment information |
| `/api/reports/replenishment.csv` | Downloads replenishment report |

Additional endpoints may be available depending on the current backend implementation.

## 🌐 Live Deployment

**GitHub Repository:**  
https://github.com/KommuPavan99/AI-Based-Inventory-Demand-Forecasting-System

**Live Frontend:**  
https://finalyearaimini.infinityfree.me/

**Backend API:**  
https://ai-based-inventory-demand-forecasting.onrender.com

**API Documentation:**  
https://ai-based-inventory-demand-forecasting.onrender.com/docs

## 🔮 Future Enhancements

- Integration with real-time sales and purchase transactions.
- Integration with live business databases.
- Improved forecasting using advanced Machine Learning and Deep Learning models.
- Seasonal and holiday demand analysis.
- Promotion-based demand prediction.
- Automated purchase order generation.
- Email and notification-based stock alerts.
- Multi-warehouse inventory management.
- User authentication and role-based access control.
- Improved voice interaction capabilities.

## 🎯 Applications

- Retail stores
- Supermarkets
- E-commerce businesses
- Warehouses
- Manufacturing industries
- Supply chain management
- Distribution centres

## 🏁 Conclusion

The AI-Based Inventory Demand Forecasting System demonstrates how Machine Learning and web technologies can improve traditional inventory management. By combining demand forecasting, historical data analysis, stockout risk identification, and intelligent replenishment recommendations, the system helps users make informed inventory decisions.

The project provides an integrated platform for understanding demand patterns, predicting future requirements, and supporting efficient inventory planning.

## 👨‍💻 Author

**Pavan Kommu**

GitHub: [@KommuPavan99](https://github.com/KommuPavan99)

---

⭐ If you find this project useful, consider giving the repository a star!
