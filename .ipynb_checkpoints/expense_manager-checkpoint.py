import pandas as pd

class ExpenseSharing:

    def __init__(self, friends):
        self.friends = friends
        self.balances = {friend: 0.0 for friend in friends}
        self.transactions = []

    def add_expense(self, payer, amount, participants):

        split_amount = amount / len(participants)

        for person in participants:
            self.balances[person] -= split_amount

        self.balances[payer] += amount

        self.transactions.append({
            "payer": payer,
            "amount": amount,
            "participants": ",".join(participants),
            "split": split_amount
        })

    def show_balances(self):

        print("\nCurrent Balances")

        for person, balance in self.balances.items():
            print(f"{person}: ₹{balance:.2f}")

    def settle_expenses(self):

        creditors = []
        debtors = []

        for person, balance in self.balances.items():

            if balance > 0:
                creditors.append([person, balance])

            elif balance < 0:
                debtors.append([person, -balance])

        print("\nFinal Settlement")

        while debtors and creditors:

            debtor = debtors.pop()
            creditor = creditors.pop()

            debtor_name, debt = debtor
            creditor_name, credit = creditor

            payment = min(debt, credit)

            print(
                f"{debtor_name} pays "
                f"{creditor_name} ₹{payment:.2f}"
            )

            debt -= payment
            credit -= payment

            if debt > 0:
                debtors.append([debtor_name, debt])

            if credit > 0:
                creditors.append([creditor_name, credit])

    def export_data(self):

        df = pd.DataFrame(self.transactions)
        df.to_csv("sample_data.csv", index=False)

        print("\nTransaction data exported.")