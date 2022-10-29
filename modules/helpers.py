import numpy as np

class ECDF():
    def __init__(self, data):
        self.x = np.sort(data)
        self.y = np.arange(1, len(self.x)+1) / len(self.x) # transforms in fractions based on the position of each datapoint

    # cumulative distribution function
    def cdf(self, value):
        return self.y[np.argmax(self.x >= value)]

    # percent point function (inverse of cdf)
    def ppf(self, percentile):
        return self.x[np.argmax(self.y >= percentile)]


if __name__ == "__main__":
    dt = np.random.normal(50, 5, size=1000)
    ecdf = ECDF(dt)
    print(ecdf.cdf(49))
    print(ecdf.ppf(0.5))
  
  