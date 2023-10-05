import datetime
import numpy as np
import yfinance as yf  # Import yfinance for fetching stock data
from matplotlib import pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn.hmm import GaussianHMM

# Define the start and end date for fetching stock data
start_date = datetime.date(1995, 10, 10)
end_date = datetime.date(2015, 4, 25)

# Fetch historical stock data using yfinance
symbol = 'INTC'
data = yf.download(symbol, start=start_date, end=end_date)

# Extract closing prices and volumes
closing_quotes = data['Adj Close'].values
volumes = data['Volume'].values[1:]  # Remove the first row

# Calculate difference percentages
diff_percentages = 100.0 * np.diff(closing_quotes) / closing_quotes[:-1]
dates = data.index.values[1:]  # Use the datetime index

# Combine data for training
training_data = np.column_stack([diff_percentages, volumes])

# Create and fit the HMM model
hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000, random_state=42)
hmm.fit(training_data)

# Generate and plot samples
num_samples = 300
samples, _ = hmm.sample(num_samples)

plt.figure()
plt.title('Difference percentages')
plt.plot(np.arange(num_samples), samples[:, 0], c='black')

plt.figure()
plt.title('Volume of shares')
plt.plot(np.arange(num_samples), samples[:, 1], c='black')
plt.ylim(ymin=0)
plt.show()
