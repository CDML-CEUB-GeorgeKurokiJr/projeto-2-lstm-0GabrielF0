import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

def add_indicators(df, col, window=14):

    df[f'SMA_{window}'] = df[col].rolling(window).mean()
    df[f'EMA_{window}'] = df[col].ewm(span=window, adjust=False).mean()

    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df[f'RSI_{window}'] = 100 - (100/(1+rs))

    return df

tickers = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO"]

target_stock = "NVDA"   ##### <<<< MUDE AQUI A AÇÃO <<<<
print("Target stock:", target_stock)

dfs = []

for t in tickers:

    df = yf.download(t, start="2015-01-01", end="2025-01-01")
    #PANDEMIA 
    df = df[(df.index.year != 2020) & (df.index.year != 2021)]

    df = df[['Close']].rename(columns={'Close':f'{t}_Close'})
    df = add_indicators(df, col=f'{t}_Close')
    dfs.append(df)

data = pd.concat(dfs, axis=1).dropna()
target_column = f"{target_stock}_Close"

# Separa X e y
y = data[[target_column]].values
X = data.drop(columns=[target_column]).values

# Normalização
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# sequências para entrada na LSTM
def create_sequences(X, y, seq_len):
    Xs, ys = [], []

    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

seq_len = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)

# treino/teste
split = int(0.8 * len(X_seq))

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
test_ds = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# modelo LSTM + GRU
class LSTM_GRU_Model(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        #LSTM aprende padrões de longo prazo
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        #GRU refina as representações aprendidas pela LSTM
        self.gru = nn.GRU(64, 32, batch_first=True)
        #Camada final transforma GRU em uma predição
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out_lstm,_ = self.lstm(x)
        out_gru,_ = self.gru(out_lstm)
        out = self.fc(out_gru[:,-1,:])

        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_GRU_Model(X_train.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# treinamento
best_val_loss = float('inf')
best_model_state = None

train_losses=[]
val_losses=[]

for epoch in range(1,51):
    model.train()
    batch_losses=[]
    for xb,yb in train_loader:
        xb,yb = xb.to(device),yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds,yb.squeeze())
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)

    model.eval()

    val_batch_losses=[]

    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb = xb.to(device),yb.to(device)
            preds = model(xb).squeeze()
            val_loss = criterion(preds,yb.squeeze())
            val_batch_losses.append(val_loss.item())
    val_loss = np.mean(val_batch_losses)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch}: Train={train_loss:.6f} Val={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = deepcopy(model.state_dict())

model.load_state_dict(best_model_state)

#predição teste
preds=[]

model.eval()

with torch.no_grad():
    for xb,yb in test_loader:
        xb = xb.to(device)
        out = model(xb)
        preds.extend(out.cpu().numpy())
preds = np.array(preds)
pred_prices = scaler_y.inverse_transform(preds)
real_prices = scaler_y.inverse_transform(y_test)

# plot previsão
plt.figure(figsize=(10,5))
plt.plot(real_prices,label="Real")
plt.plot(pred_prices,label="Predicted")
plt.legend()
plt.title(f"Stock Price Prediction - {target_stock}")
plt.show()

# plot o quão o modelo errou
errors = real_prices.flatten() - pred_prices.flatten()
plt.figure(figsize=(10,4))
plt.plot(errors)
plt.title("Prediction Error Over Time")
plt.xlabel("Time")
plt.ylabel("Error")
plt.grid(True)
plt.show()

# plot loss
plt.figure(figsize=(8,4))
plt.plot(train_losses,label="Train Loss")
plt.plot(val_losses,label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# previsão futura
model.eval()
n_future = 7
last_sequence = X_scaled[-seq_len:]
future_preds=[]
current_input = torch.tensor(last_sequence,dtype=torch.float32).unsqueeze(0).to(device)
for _ in range(n_future):
    with torch.no_grad():
        next_pred = model(current_input)
        pred_val = next_pred.cpu().numpy()[0,0]
        future_preds.append(pred_val)
    next_step = current_input[0,-1,:].cpu().numpy()
    next_sequence = np.vstack([current_input[0,1:].cpu(),next_step])
    current_input = torch.tensor(next_sequence,dtype=torch.float32).unsqueeze(0).to(device)
future_prices = scaler_y.inverse_transform(np.array(future_preds).reshape(-1,1))

# plot previsão futura
plt.figure(figsize=(10,5))
plt.plot(range(1,n_future+1), future_prices, marker='o')
plt.title(f"7-Day Forecast for {target_stock}")
plt.xlabel("Days Ahead")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.show()

print("\nPrevisão futura:")
for i,p in enumerate(future_prices):
    print(f"Dia +{i+1}: ${p[0]:.2f}")