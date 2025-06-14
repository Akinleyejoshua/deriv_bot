from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
import logging
app = FastAPI()
warnings.filterwarnings('ignore')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*', ''],
    allow_methods=["*"],
    allow_headers=["*"]
)

class DerivTradingBot:
    def __init__(self):
        # Connection attributes
        self.app_id = None
        self.token = None
        self.websocket = None

        # Trading parameters
        self.symbol = None
        self.duration = None
        self.duration_unit = None  # 't' for ticks, 's' for seconds
        self.stake_amount = None
        self.max_trades = None
        self.trades_per_signal = None

        # Market condition thresholds
        self.volatile_threshold = None
        self.confidence_threshold = None
        self.min_confidence = None

        # Data management
        self.historical_data = pd.DataFrame()
        self.scaler = StandardScaler()
        self.model = None

        # Trading state
        self.active_trades = []
        self.trade_history = []
        self.balance = 0.0
        self.current_tick = None
        self.trade_count = 0
        self.is_trading = False

        # Statistics
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0

    def set_connection_params(self, app_id: str, token: str):
        """Set API connection parameters"""
        self.app_id = app_id
        self.token = token

    def set_trading_params(self, symbol: str, duration: int, duration_unit: str,
                          stake_amount: float, max_trades: int, trades_per_signal: int):
        """Set trading parameters"""
        self.symbol = symbol
        self.duration = duration
        self.duration_unit = duration_unit
        self.stake_amount = stake_amount
        self.max_trades = max_trades
        self.trades_per_signal = trades_per_signal

    def set_market_conditions(self, volatile_threshold: float, confidence_threshold: float,
                            min_confidence: float):
        """Set market condition thresholds"""
        self.volatile_threshold = volatile_threshold
        self.confidence_threshold = confidence_threshold
        self.min_confidence = min_confidence

    def log(self, message: str, icon: str = "📊"):
        """Enhanced logging with icons and timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{icon} [{timestamp}] {message}")

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(
                f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
            )
            self.log("Connected to Deriv WebSocket", "🔗")

            # Authorize connection
            auth_msg = {"authorize": self.token}
            await self.websocket.send(json.dumps(auth_msg))
            response = await self.websocket.recv()
            auth_data = json.loads(response)

            if "error" in auth_data:
                self.log(f"Authorization failed: {auth_data['error']['message']}", "❌")
                return False

            self.balance = auth_data.get("authorize", {}).get("balance", 0)
            self.log(f"Authorized successfully. Balance: ${self.balance:.2f}", "✅")
            return True

        except Exception as e:
            self.log(f"Connection error: {str(e)}", "❌")
            return False

    async def get_historical_data(self, count: int = 100):
        """Fetch historical tick data"""
        try:
            ticks_history_msg = {
                "ticks_history": self.symbol,
                "count": count,
                "end": "latest",
                "style": "ticks"
            }

            await self.websocket.send(json.dumps(ticks_history_msg))
            response = await self.websocket.recv()
            data = json.loads(response)

            if "error" in data:
                self.log(f"Error fetching historical data: {data['error']['message']}", "❌")
                return False

            history = data["history"]
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(history["times"], unit='s'),
                'price': history["prices"]
            })
            df.set_index('timestamp', inplace=True)
            df['returns'] = df['price'].pct_change()
            df['volatility'] = df['returns'].rolling(10).std()

            self.historical_data = df.dropna()
            self.log(f"Loaded {len(self.historical_data)} historical ticks", "📈")
            return True

        except Exception as e:
            self.log(f"Error loading historical data: {str(e)}", "❌")
            return False

    def prepare_features(self, data: pd.DataFrame):
        """Prepare features for SARIMAX model"""
        features = pd.DataFrame(index=data.index)
        features['price'] = data['price']
        features['returns'] = data['returns']
        features['volatility'] = data['volatility']
        features['price_lag1'] = data['price'].shift(1)
        features['price_lag2'] = data['price'].shift(2)
        features['returns_lag1'] = data['returns'].shift(1)
        features['ma_5'] = data['price'].rolling(5).mean()
        features['ma_10'] = data['price'].rolling(10).mean()

        return features.dropna()

    def train_model(self):
        """Train SARIMAX model"""
        try:
            if len(self.historical_data) < 30:
                self.log("Insufficient data for training", "⚠️")
                return False

            features = self.prepare_features(self.historical_data)

            # Prepare exogenous variables
            exog_vars = ['returns', 'volatility', 'price_lag1', 'returns_lag1']
            exog_data = features[exog_vars].fillna(method='bfill')

            # Fit SARIMAX model
            self.model = SARIMAX(
                features['price'],
                exog=exog_data,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            self.model = self.model.fit(disp=False)
            self.log("SARIMAX model trained successfully", "🤖")
            return True

        except Exception as e:
            self.log(f"Model training error: {str(e)}", "❌")
            return False

    def predict_next_tick(self):
        """Make prediction for next tick"""
        try:
            if self.model is None:
                return None, 0.0

            # Get latest data for exogenous variables
            latest_data = self.historical_data.tail(1)
            features = self.prepare_features(self.historical_data.tail(20))

            if len(features) == 0:
                return None, 0.0

            exog_vars = ['returns', 'volatility', 'price_lag1', 'returns_lag1']
            exog_latest = features[exog_vars].tail(1).fillna(method='bfill')

            # Make prediction
            forecast = self.model.forecast(steps=1, exog=exog_latest)
            prediction = forecast.iloc[0]

            # Calculate confidence based on model's standard error
            confidence_interval = self.model.get_forecast(steps=1, exog=exog_latest).conf_int()
            confidence = 1.0 - (confidence_interval.iloc[0, 1] - confidence_interval.iloc[0, 0]) / prediction
            confidence = max(0.0, min(1.0, abs(confidence)))

            return prediction, confidence

        except Exception as e:
            self.log(f"Prediction error: {str(e)}", "❌")
            return None, 0.0

    def analyze_market_condition(self):
        """Analyze current market volatility"""
        if len(self.historical_data) < 10:
            return "unknown"

        current_volatility = self.historical_data['volatility'].tail(1).iloc[0]
        avg_volatility = self.historical_data['volatility'].tail(20).mean()

        if current_volatility > avg_volatility * self.volatile_threshold:
            return "volatile"
        elif current_volatility < avg_volatility * 0.5:
            return "calm"
        else:
            return "normal"

    def should_trade(self, prediction: float, confidence: float, current_price: float):
        """Determine if we should place a trade"""
        if confidence < self.min_confidence:
            return None, "Low confidence"

        market_condition = self.analyze_market_condition()

        # Adjust confidence threshold based on market condition
        required_confidence = self.confidence_threshold
        if market_condition == "volatile":
            required_confidence += 0.1
        elif market_condition == "calm":
            required_confidence -= 0.05

        if confidence < required_confidence:
            return None, f"Confidence {confidence:.3f} below threshold for {market_condition} market"

        # Determine trade direction
        price_diff = (prediction - current_price) / current_price
        print(price_diff)
        if abs(price_diff) < 0.009:  # Too small movement
            return None, "Predicted movement too small"

        direction = "CALL" if price_diff > 0 else "PUT"
        return direction, f"{market_condition} market, confidence: {confidence:.3f}"

    async def place_trade(self, direction: str, reason: str):
        """Place a binary options trade"""
        try:
            trade_msg = {
                "buy": 1,
                "price": self.stake_amount,
                "parameters": {
                    "contract_type": direction,
                    "symbol": self.symbol,
                    "duration": self.duration,
                    "duration_unit": self.duration_unit,
"currency":"USD",
"basis":"stake",
"amount":self.stake_amount
                }
            }

            await self.websocket.send(json.dumps(trade_msg))
            response = await self.websocket.recv()
            trade_data = json.loads(response)

            if "error" in trade_data:
                self.log(f"Trade placement failed: {trade_data['error']['message']}", "❌")
                return False

            contract_id = trade_data["buy"]["contract_id"]
            buy_price = trade_data["buy"]["buy_price"]

            trade_info = {
                'contract_id': contract_id,
                'direction': direction,
                'stake': self.stake_amount,
                'buy_price': buy_price,
                'entry_price': self.current_tick,
                'timestamp': datetime.now(),
                'reason': reason,
                'status': 'active'
            }

            self.active_trades.append(trade_info)
            self.trade_count += 1

            self.log(f"Trade #{self.trade_count} placed: {direction} ${self.stake_amount} | {reason}", "🚀")
            return True

        except Exception as e:
            self.log(f"Trade placement error: {str(e)}", "❌")
            return False

    async def check_active_trades(self):
        """Check status of active trades"""
        for trade in self.active_trades[:]:
            try:
                proposal_msg = {
                    "proposal_open_contract": 1,
                    "contract_id": trade['contract_id']
                }

                await self.websocket.send(json.dumps(proposal_msg))
                response = await self.websocket.recv()
                contract_data = json.loads(response)

                if "error" in contract_data:
                    continue

                contract = contract_data.get("proposal_open_contract", {})

                if contract.get("is_sold"):
                    # Trade completed
                    profit = contract.get("profit", 0)
                    sell_price = contract.get("sell_price", 0)

                    trade['profit'] = profit
                    trade['sell_price'] = sell_price
                    trade['exit_price'] = self.current_tick
                    trade['status'] = 'completed'

                    self.active_trades.remove(trade)
                    self.trade_history.append(trade)

                    self.total_profit += profit
                    if profit > 0:
                        self.wins += 1
                        result_icon = "✅"
                    else:
                        self.losses += 1
                        result_icon = "❌"

                    # Calculate prediction accuracy
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                    predicted_direction = trade['direction']
                    actual_direction = "CALL" if exit_price > entry_price else "PUT"

                    accuracy = "✓" if predicted_direction == actual_direction else "✗"

                    self.log(f"Trade completed {result_icon} | Profit: ${profit:.2f} | "
                           f"Prediction: {accuracy} | Entry: {entry_price:.5f} → Exit: {exit_price:.5f}", "📊")

                    await self.update_balance()

            except Exception as e:
                self.log(f"Error checking trade {trade['contract_id']}: {str(e)}", "❌")

    async def update_balance(self):
        """Update current balance"""
        try:
            balance_msg = {"balance": 1}
            await self.websocket.send(json.dumps(balance_msg))
            response = await self.websocket.recv()
            balance_data = json.loads(response)

            if "balance" in balance_data:
                self.balance = balance_data["balance"]["balance"]

        except Exception as e:
            self.log(f"Balance update error: {str(e)}", "❌")

    def print_statistics(self):
        """Print comprehensive trading statistics"""
        win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
        avg_profit = self.total_profit / max(1, len(self.trade_history))

        print("\n" + "="*60)
        print("📊 TRADING STATISTICS")
        print("="*60)
        print(f"💰 Current Balance: ${self.balance:.2f}")
        print(f"📈 Total Profit: ${self.total_profit:.2f}")
        print(f"🎯 Trades: {len(self.trade_history)} (Active: {len(self.active_trades)})")
        print(f"✅ Wins: {self.wins} | ❌ Losses: {self.losses}")
        print(f"📊 Win Rate: {win_rate:.1f}%")
        print(f"💵 Avg Profit per Trade: ${avg_profit:.2f}")
        print(f"🔄 Current Tick: {self.current_tick:.5f}" if self.current_tick else "")
        print("="*60 + "\n")

    async def subscribe_to_ticks(self):
        """Subscribe to real-time tick data"""
        try:
            tick_msg = {"ticks": self.symbol, "subscribe": 1}
            await self.websocket.send(json.dumps(tick_msg))
            self.log(f"Subscribed to {self.symbol} ticks", "📡")

        except Exception as e:
            self.log(f"Tick subscription error: {str(e)}", "❌")

    async def trading_loop(self):
        """Main trading loop"""
        self.is_trading = True
        self.log("Starting trading loop", "🚀")

        while self.is_trading and self.trade_count < self.max_trades:
            try:
                # Receive tick data
                response = await self.websocket.recv()
                data = json.loads(response)

                if "tick" in data:
                    tick_data = data["tick"]
                    self.current_tick = tick_data["quote"]
                    timestamp = pd.to_datetime(tick_data["epoch"], unit='s')

                    # Update historical data
                    new_row = pd.DataFrame({
                        'price': [self.current_tick],
                        'returns': [np.nan],
                        'volatility': [np.nan]
                    }, index=[timestamp])

                    if not self.historical_data.empty:
                        new_row['returns'] = (self.current_tick - self.historical_data['price'].iloc[-1]) / self.historical_data['price'].iloc[-1]

                    self.historical_data = pd.concat([self.historical_data, new_row])
                    self.historical_data = self.historical_data.tail(200)  # Keep last 200 ticks

                    # Recalculate volatility
                    self.historical_data['volatility'] = self.historical_data['returns'].rolling(10).std()

                    # Check active trades
                    await self.check_active_trades()

                    # Only trade if no active trades (prevent overlapping)
                    if len(self.active_trades) == 0 and len(self.historical_data) > 30:
                        # Retrain model periodically
                        if len(self.historical_data) % 50 == 0:
                            self.train_model()

                        # Make prediction
                        prediction, confidence = self.predict_next_tick()

                        if prediction is not None:
                            # Decide on trade
                            direction, reason = self.should_trade(prediction, confidence, self.current_tick)

                            if direction == "PUT":
                                # Place multiple trades per signal if configured
                                for i in range(self.trades_per_signal):
                                    if self.trade_count < self.max_trades:
                                        await self.place_trade(direction, reason)
                                        await asyncio.sleep(0.1)  # Small delay between trades

                            self.log(f"Prediction: {prediction:.5f} | Current: {self.current_tick:.5f} | "
                                   f"Confidence: {confidence:.3f} | Action: {direction or 'HOLD'}", "🔮")

                    # Print statistics every 10 ticks
                    if len(self.historical_data) % 10 == 0:
                        self.print_statistics()

                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming

            except websockets.exceptions.ConnectionClosed:
                self.log("WebSocket connection closed", "⚠️")
                break
            except Exception as e:
                self.log(f"Trading loop error: {str(e)}", "❌")
                await asyncio.sleep(1)

    async def run(self):
        """Main bot execution"""
        self.log("Initializing Deriv Trading Bot", "🤖")

        # Connect to API
        if not await self.connect():
            return

        # Load historical data and train model
        if not await self.get_historical_data():
            return

        if not self.train_model():
            return

        # Subscribe to real-time ticks
        await self.subscribe_to_ticks()

        # Start trading
        await self.trading_loop()

        # Final statistics
        self.print_statistics()
        self.log("Trading bot stopped", "🛑")

# Usage Example
async def main():
    bot = DerivTradingBot()

    # Set connection parameters
    bot.set_connection_params("40003", "1C1Qo1372JjoJ1w")

    # Set trading parameters
    bot.set_trading_params(
        symbol="R_50",  # Synthetic index
        duration=1,  # 5 ticks
        duration_unit="t",  # ticks
        stake_amount=1000.0,  # $1 per trade
        max_trades=100000000000,  # Maximum number of trades
        trades_per_signal=50  # Trades per signal
    )

    # Set market condition parameters
    bot.set_market_conditions(
        volatile_threshold=1.5,  # Volatility threshold multiplier
        confidence_threshold=0.6,  # Minimum confidence for normal market
        min_confidence=0.5  # Absolute minimum confidence
    )

    # Run the bot
    await bot.run()

@app.get("/")
async def root():
    asyncio.run(main())
    return "started"
    
