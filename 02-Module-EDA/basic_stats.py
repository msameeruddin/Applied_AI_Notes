import math
from collections import Counter

class StatsBasics():
    def compute_mean(self, data):
        mean_val = sum(data)/len(data)
        return mean_val

    def compute_median(self, data):
        data = sorted(data)
        n = len(data)
        mid_idx = n // 2

        if (n % 2 != 0):
            return data[mid_idx]
        return (data[mid_idx - 1] + data[mid_idx]) / 2

    def compute_mode(self, data):
        datac = Counter(data)
        max_freq = max(list(datac.values()))

        if (max_freq == 1):
            return "Mode doesn't exist"

        modals = [i for (i, j) in datac.items() if (j == max_freq)]
        return min(modals)

    def compute_stddev(self, data):
        mean_val = self.compute_mean(data=data)
        dispersions = [(i - mean_val)**2 for i in data]
        dispersion_mean = self.compute_mean(data=dispersions)
        return math.sqrt(dispersion_mean)

    def compute_variance(self, data):
        stddev = self.compute_stddev(data=data)
        return stddev**2

    def compute_percentile(self, p, data):
        data = sorted(data)
        
        if (p == 100):
            return data[-1]
        
        l_p = (len(data) - 1) * (p / 100) + 1
        int_l_p = int(l_p)
        fl_l_p = l_p - int_l_p
        val1 = data[int_l_p - 1]
        val2 = data[int_l_p]    
        pval = val1 + (fl_l_p * (val2 - val1))
        
        return round(pval, 2)

    def compute_mad(self, data, c=0.6745):
        median_val = self.compute_median(data=data)
        abs_std = [abs(i - median_val) for i in data]
        mad = self.compute_median(data=abs_std) / c
        return round(mad, 2)

    def compute_covariance(self, X, Y):
        if (len(X) != len(Y)):
            return None

        mean_x = self.compute_mean(data=X)
        mean_y = self.compute_mean(data=Y)

        covals = [(x - mean_x)*(y - mean_y) for (x, y) in zip(X, Y)]
        covar_val = self.compute_mean(data=covals)

        return covar_val

    def compute_correlation(self, X, Y):
        covar_val = self.compute_covariance(X=X, Y=Y)
        std_X = self.compute_stddev(data=X)
        std_Y = self.compute_stddev(data=Y)
        corr_val = covar_val / (std_X * std_Y)
        return corr_val


if __name__ == '__main__':
    import random

    X = [random.randint(10, 50) for i in range(30)]
    Y = [random.randint(10, 80) for i in range(30)]
    X = sorted(X)
    Y = sorted(Y)

    stats = StatsBasics()

    print("Mean →\t", stats.compute_mean(data=X))
    print("Median →\t", stats.compute_median(data=X))
    print("Mode →\t", stats.compute_mode(data=X))
    print("Stddev →\t", stats.compute_stddev(data=X))
    print("Variance →\t", stats.compute_variance(data=X))
    print("MAD →\t", stats.compute_mad(data=X))
    print("Covar →\t", stats.compute_covariance(X=X, Y=Y))
    print("Corr →\t", stats.compute_correlation(X=X, Y=Y))