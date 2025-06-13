import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("apple_products.csv")

# Data Cleaning
df['Sale Price'] = df['Sale Price'].replace('[₹,]', '', regex=True).astype(float)
df['Mrp'] = df['Mrp'].replace('[₹,]', '', regex=True).astype(float)
df['Number Of Ratings'] = pd.to_numeric(df['Number Of Ratings'], errors='coerce')
df['Number Of Reviews'] = pd.to_numeric(df['Number Of Reviews'], errors='coerce')
df['Discount Percentage'] = pd.to_numeric(df['Discount Percentage'], errors='coerce')
df['Star Rating'] = pd.to_numeric(df['Star Rating'], errors='coerce')
df['Ram'] = df['Ram'].apply(lambda x: float(str(x).split()[0]) if pd.notnull(x) else 0)
df.dropna(inplace=True)

# Feature Engineering
df['Computed Discount %'] = round(((df['Mrp'] - df['Sale Price']) / df['Mrp']) * 100, 2)
df['Engagement Score'] = df['Number Of Ratings'] + df['Number Of Reviews']

# Sidebar Inputs
st.sidebar.header("📱 Price Range")
min_price = st.sidebar.number_input("Minimum Price", min_value=0, value=30000)
max_price = st.sidebar.number_input("Maximum Price", min_value=0, value=100000)

# Train Model
features = ['Mrp', 'Discount Percentage', 'Number Of Ratings', 'Number Of Reviews', 'Star Rating', 'Ram']
X = df[features]
y = df['Sale Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict prices for all phones
df['Predicted Price'] = xgb_model.predict(X)

# Filter based on user price range
filtered_df = df[(df['Predicted Price'] >= min_price) & (df['Predicted Price'] <= max_price)]

# Recommend top 5
top_5 = filtered_df.sort_values(by=['Star Rating', 'Engagement Score'], ascending=False).head(5)

st.title("📊 iPhone Insights & Recommendations")

st.subheader("🔝 Top 5 Recommended iPhones")
st.dataframe(top_5[['Product Name', 'Predicted Price', 'Star Rating', 'Number Of Ratings']])

# 📊 Graphs
st.subheader("💰 Price Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['Sale Price'], kde=True, ax=ax1)
ax1.set_title("Distribution of iPhone Sale Prices")
st.pyplot(fig1)

st.subheader("📉 Discount vs Sale Price")
fig2, ax2 = plt.subplots()
sns.regplot(data=df, x='Sale Price', y='Computed Discount %', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax2)
ax2.set_title("Discount % vs Sale Price")
st.pyplot(fig2)

st.subheader("⭐ Star Ratings Count")
fig3, ax3 = plt.subplots()
sns.countplot(x='Star Rating', data=df, ax=ax3)
ax3.set_title("Star Ratings Count")
st.pyplot(fig3)

st.subheader("📈 Correlation Heatmap")
fig4, ax4 = plt.subplots()
sns.heatmap(df[['Sale Price', 'Mrp', 'Number Of Ratings', 'Number Of Reviews', 'Star Rating']].corr(), annot=True, cmap='Blues', ax=ax4)
st.pyplot(fig4)

st.subheader("📊 Plotly: Ratings vs Sale Price")
fig5 = px.scatter(df, x="Number Of Ratings", y="Sale Price", size="Discount Percentage", trendline="ols",
                  title="Sale Price vs Number of Ratings")
st.plotly_chart(fig5)

st.subheader("📊 Plotly: Ratings vs Discount")
fig6 = px.scatter(df, x="Number Of Ratings", y="Discount Percentage", size="Sale Price", trendline="ols",
                  title="Discount % vs Number of Ratings")
st.plotly_chart(fig6)
