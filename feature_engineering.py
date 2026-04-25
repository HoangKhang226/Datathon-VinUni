import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from pathlib import Path

# read csv files
df_customers = pd.read_csv("Data/customers.csv")
df_geography = pd.read_csv("Data/geography.csv")
df_inventory = pd.read_csv("Data/inventory.csv")
df_order_items = pd.read_csv("Data/order_items.csv")
df_orders = pd.read_csv("Data/orders.csv")
df_products = pd.read_csv("Data/products.csv")
df_payments = pd.read_csv("Data/payments.csv")
df_promotions = pd.read_csv("Data/promotions.csv")
df_reviews = pd.read_csv("Data/reviews.csv")
df_returns = pd.read_csv("Data/returns.csv")
df_shipments = pd.read_csv("Data/shipments.csv")
df_web_traffic = pd.read_csv("Data/web_traffic.csv")

