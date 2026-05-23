Cinema Audience Forecasting Challenge
Project Overview
This project addresses the critical business challenge of forecasting cinema audience counts to optimize operational efficiency and enhance customer experience. Accurate predictions enable theater managers to make informed decisions regarding staffing, inventory management (e.g., concessions), and targeted marketing campaigns. The forecasting model leverages diverse data sources from two primary booking platforms: BookNow (online bookings) and CinePOS (point-of-sale systems).

Data Description
The dataset comprises time-series data related to cinema operations, audience visits, and booking behavior. It is sourced from two distinct platforms, necessitating careful integration.

Key Data Files:

booknow_visits.csv: Contains historical daily audience_count for theaters registered with the BookNow platform. This is the primary target variable for our forecasting model.
booknow_booking.csv: Transaction-level data for tickets booked through the BookNow online platform, including show_datetime and booking_datetime.
cinePOS_booking.csv: Transaction-level data for tickets sold via the CinePOS point-of-sale system, also including show_datetime and booking_datetime.
booknow_theaters.csv: Metadata (e.g., theater_type, theater_area, approximate latitude, longitude) for theaters on the BookNow platform.
cinePOS_theaters.csv: Metadata for theaters utilizing the CinePOS system.
movie_theater_id_relation.csv: A crucial mapping file that links book_theater_id (from BookNow) to cine_theater_id (from CinePOS), facilitating data integration across platforms.
date_info.csv: Supplemental calendar information for each show_date, such as day_of_week, and holiday indicators.
sample_submission.csv: Defines the required format for the final predictions.
Important Notes:

Audience counts are influenced by factors like holidays, weekends, theater type, and booking trends.
Latitude and longitude values are anonymized.
The model integrates both POS and online booking patterns.
Days with zero audiences (due to theater closures or no shows) are included but are handled appropriately during processing.
Methodology
The project follows a structured machine learning pipeline:

1. Data Loading and Initial Exploration
All provided CSV files are loaded into pandas DataFrames. An initial assessment of data shapes, column names, data types, and date ranges is performed. Critical date columns across all datasets are converted to datetime objects to ensure consistency and enable time-series operations.

2. Exploratory Data Analysis (EDA)
EDA is conducted to uncover patterns, distributions, and relationships relevant to audience forecasting. Key analyses include:

Global Audience Trends: Visualizing overall daily audience counts to identify general trends.
Audience Count Distribution: Examining the distribution of audience_count, noting any sparsity or zero-inflated characteristics.
Data Overlap and Coverage: Assessing how much of the target visits data is covered by booknow_theaters metadata and relation mappings.
Booking System Volume & Lead Time: Comparing ticket sales volume between CinePOS (physical) and BookNow (online) systems. Analyzing lead times (time between booking and show) to understand customer planning horizons. Insights reveal that CinePOS handles higher volume, often for last-minute purchases, while BookNow captures more planned, advance bookings.
Impact of Theater Type and Day of Week: Analyzing how theater_type and day_of_week influence audience size.
Autocorrelation (The Echo): Identifying periodic patterns (e.g., weekly seasonality) in audience counts using autocorrelation plots, crucial for lagged feature engineering.
Identifying Peak Days: Detecting unusually high audience days (e.g., 2 standard deviations above mean) to pinpoint potential holidays or blockbuster events.
3. Data Preprocessing and Feature Engineering
This phase transforms raw data into a machine learning-ready format, integrating information from both booking platforms into a master_df.

Steps include:

Aggregating Booking Data: Consolidating cinePOS_booking and booknow_booking to daily pos_tickets and online_tickets for each theater-date pair. The movie_theater_id_relation.csv is used to link CinePOS bookings to the BookNow theater IDs.
Assembling Master Table: Merging visits with date_info and booknow_theaters to create a comprehensive dataset for each book_theater_id and show_date.
Handling Missing Values: Filling NaN in online_tickets and pos_tickets with 0 (implying no sales for that channel). Filling missing theater_type and theater_area with 'Unknown' to retain information and allow the model to learn from this category. Dropping latitude and longitude as they are anonymized and not used as features.
Feature Engineering:
Date Features: Extracting day_of_week_num, is_weekend, month to capture seasonality.
Cyclical Date Features: Applying sine and cosine transformations to month and day_of_week_num to represent their cyclical nature more effectively.
Lagged Features: Creating lag_7 and lag_14 (audience counts from 7 and 14 days prior) to capture weekly and bi-weekly patterns.
Rolling Mean Features: Calculating rolling_7_mean and rolling_28_mean (average audience over past 7 and 28 days) to capture short-term trends and longer-term smoothed patterns. A shift(1) is applied to all lagged and rolling features to prevent data leakage.
Exponential Moving Average (EWM): Incorporating ewm_7 to give more weight to recent observations, capturing recent trends more responsively.
Label Encoding: Converting categorical features (theater_type, theater_area, day_of_week) into numerical representations for model compatibility.
4. Model Training: XGBoost with Advanced Features
A robust XGBoost Regressor model is trained, selected for its strong performance with tabular data and ability to handle complex interactions.

Time-Based Data Split: The data is split chronologically into training (85%) and validation (15%) sets to simulate real-world forecasting scenarios and prevent data leakage.
XGBoost Configuration: Parameters like n_estimators, learning_rate, max_depth, subsample, colsample_bytree, and early_stopping_rounds are tuned to optimize performance and prevent overfitting.
Evaluation: The model's performance is primarily evaluated using RMSE (Root Mean Squared Error) and R2 Score on the validation set. A significant improvement in RMSE is observed after incorporating advanced cyclical, lagged, and rolling features.
Feature Importance: A feature importance plot is generated to identify the most influential predictors, providing business insights into what drives audience attendance (e.g., rolling_7_mean, ewm_7, lag_7 often show high importance).
5. Recursive Forecasting Loop
Forecasting future audience counts requires a recursive approach because time-dependent features (e.g., rolling_7_mean, lag_7, ewm_7) for future dates depend on prior audience_count values, which are unknown. The loop proceeds day-by-day:

Initialize History: The master_df (historical data) is used as the starting point.
Iterate Daily: For each day in the future prediction period:
Prepare Data: Rows for the current prediction date are extracted from the sample_submission template.
Generate Features: All necessary features (date-based, static, and time-dependent) are generated. Crucially, pos_tickets and online_tickets are treated as unknown (or zero if no future booking data is available), simulating real-world conditions. Time-dependent features are calculated by combining recent historical data (which now includes previous predictions) with the current day's data.
Encode Categoricals: Label encoding is applied consistently.
Predict: The trained XGBoost model predicts audience_count for the current day's theaters.
Update History: The new predictions are appended to the historical dataset (history_df). This step is vital, as today's prediction becomes part of the history used to generate features for tomorrow's prediction.
Aggregate & Save: All daily predictions are collected and formatted into submission.csv as per the sample_submission.csv structure.
Results and Conclusion
The final model, utilizing XGBoost with an extensive set of engineered features (including cyclical date features, various lags, and rolling/exponential moving averages) and employing a recursive forecasting strategy, demonstrates robust performance.

Validation RMSE: (e.g., 20.0675 in the notebook's run) indicates the average magnitude of the prediction errors.
Validation R2 Score: (e.g., 0.5381 in the notebook's run) shows the proportion of variance in audience counts explained by the model.
These metrics highlight the model's ability to capture complex time-series patterns and provide reasonably accurate forecasts. The iterative approach to feature engineering and the use of both BookNow and CinePOS data streams significantly contribute to the model's predictive power, offering valuable insights for business decision-making in the cinema industry.
