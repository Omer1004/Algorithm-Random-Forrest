from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from algo1.1 import best_model_predictions, stock_list

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def create_stock_contracts(self, stock_list):
        contracts = []
        for stock in stock_list:
            contract = Contract()
            contract.symbol = stock
            contract.secType = 'STK'
            contract.exchange = 'SMART'
            contract.currency = 'USD'
            contract.primaryExchange = 'NASDAQ'
            contracts.append(contract)
        return contracts

    def create_order(self, action, quantity, order_type, price=None):
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        if order_type == 'LMT':
            order.lmtPrice = price
        return order

    def place_orders(self, contracts, orders):
        for i, (contract, order) in enumerate(zip(contracts, orders)):
            self.placeOrder(i, contract, order)

    def get_last_price_and_cash(self, contract, cash_percentage):
        last_price = None
        account_cash = None

        self.reqMarketDataType(3)  # Request real-time market data
        self.reqMktData(0, contract, "", False, False, [])  # Request market data for the contract

        while last_price is None or account_cash is None:
            self.run()
            # Wait until both last_price and account_cash are retrieved

            # Retrieve account cash
            if hasattr(self, "accountSummary"):
                for item in self.accountSummary:
                    if item.tag == "TotalCashValue":
                        account_cash = float(item.value)

            # Retrieve last price
            if hasattr(self, "tickPrice"):
                if self.tickPrice.field == 4 and self.tickPrice.tickType == 2:
                    last_price = self.tickPrice.price

        self.cancelMktData(0)  # Cancel market data request

        # Calculate amount of money to enter the trade
        trade_amount = max(0.01 * cash_percentage * account_cash, last_price)

        # Calculate number of stocks to buy
        shares_to_buy = int(trade_amount // last_price)

        return last_price, account_cash, trade_amount, shares_to_buy


def main():
    app = IBapi()
    app.connect('127.0.0.1', 7497, 123)
    app.run()

    stock_list = ['QQQ', 'AAPL', 'TSLA', 'GOOGL', 'NVDA', 'META', 'MSFT', 'AMZN', 'AMD', 'NFLX', 'BABA']

    # Create contracts for stocks
    contracts = app.create_stock_contracts(stock_list)
    print("Contracts:", contracts)

    # Create orders for each stock
    orders = []
    for contract in contracts:
        order = app.create_order(action='BUY', quantity=1, order_type='LMT', price=200)
        orders.append(order)

    # Place orders
    app.place_orders(contracts, orders)

    # Get last price, cash, trade amount, and shares to buy for a specific stock
    example_stock = contracts[0]
    cash_percentage = 5
    last_price, account_cash, trade_amount, shares_to_buy = app.get_last_price_and_cash(example_stock, cash_percentage)

    print(f"Last Price: {last_price}")
    print(f"Account Cash: {account_cash}")
    print(f"Trade Amount: {trade_amount}")
    print(f"Shares to Buy: {shares_to_buy}")


if __name__ == "__main__":
    main()
