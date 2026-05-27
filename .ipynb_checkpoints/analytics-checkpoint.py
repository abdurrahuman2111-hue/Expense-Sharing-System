import pandas as pd
import matplotlib.pyplot as plt

def generate_analytics():

    df = pd.read_csv("sample_data.csv")

    print("\nExpense Dataset")
    print(df)

    # Total spending by payer
    spending = df.groupby("payer")["amount"].sum()

    print("\nTotal Spending")
    print(spending)

    # Pie Chart
    plt.figure(figsize=(6,6))
    spending.plot(
        kind="pie",
        autopct="%1.1f%%"
    )

    plt.title("Expense Distribution")
    plt.ylabel("")

    plt.savefig("charts/spending_chart.png")

    # Bar Chart
    plt.figure(figsize=(8,5))

    spending.plot(kind="bar")

    plt.title("Total Spending Per User")
    plt.xlabel("Users")
    plt.ylabel("Amount")

    plt.savefig("charts/balance_chart.png")

    print("\nCharts Generated Successfully")