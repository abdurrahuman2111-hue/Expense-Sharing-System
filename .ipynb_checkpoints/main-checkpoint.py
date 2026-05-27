from expense_manager import ExpenseSharing
from analytics import generate_analytics

friends = ["Aman", "Rahul", "Priya", "Sara"]

app = ExpenseSharing(friends)

app.add_expense(
    payer="Aman",
    amount=1200,
    participants=["Aman", "Rahul", "Priya"]
)

app.add_expense(
    payer="Sara",
    amount=800,
    participants=["Rahul", "Sara"]
)

app.add_expense(
    payer="Priya",
    amount=500,
    participants=["Aman", "Sara", "Priya"]
)

app.show_balances()

app.settle_expenses()

app.export_data()

generate_analytics()