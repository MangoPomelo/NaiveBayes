import math
import pickle
import numpy as np


class GaussianNB(object):
    """GaussianNB"""
    def __init__(self, priors=None):
        self.priors = priors

    def fit(self, x, y):
        unique_y = sorted(np.unique(y))

        n_classes = len(unique_y)
        n_features = len(x[0])
        n_samples = len(x)

        self.mu_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))

        # The flag indicates whether priors need to be constructed or not 
        p_flg = False
        if self.priors is None:
            p_flg = True
            self.priors = np.zeros((n_classes,))

        # Update theta and sigma for each class
        for class_idx in unique_y:
            # Select coresponding subset
            x_in_this_class = x[y == class_idx]
            # Fill in the mu and sigma
            self.mu_[class_idx, :] = np.mean(x_in_this_class, axis=0)
            self.sigma_[class_idx, :] = np.var(x_in_this_class, axis=0)

            if p_flg:
                # Construct priors
                self.priors[class_idx] = len(x_in_this_class) / n_samples

    def predict(self, x):
        n_classes = len(self.priors)
        n_features = len(x[0])
        n_samples = len(x)

        # Latex: \ln{\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}} = -\ln{\sqrt{2\pi}}-\ln{\sigma}-\frac{1}{2}(\frac{x-\mu}{\sigma})^2
        log_gauss = lambda x, s, m: \
            -np.log(np.power(2*np.pi,0.5))-np.log(s)-np.power((x-m)/s,2)/2

        # ln(p(x1|y)) + ln(p(x2|y)) + ...
        # Shape = (n_samples, n_classes)
        log_likelihood = np.sum([
                log_gauss(x, self.sigma_[class_idx, :], self.mu_[class_idx, :]) 
                for class_idx in range(n_classes)], axis=2
            ).transpose()
        
        log_priors = np.log(self.priors)
        # ln(p(prior)) + ln(p(likelihood))
        log_numerator = np.asarray([log_likelihood[sample_idx, :] + log_priors for sample_idx in range(n_samples)])

        return np.argmax(log_numerator, axis=1)
                
    def score(self, x, y):
        prediction = self.predict(x)
        return np.average(prediction == y)

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        return self

    @classmethod
    def load(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)


def generate_data(seed = None, n_samples = 100):
    if seed: np.random.seed(seed)

    n_males = n_samples // 2
    n_females = n_samples - n_males

    male_heights = np.random.normal(loc=5.855, scale=(3.5033e-2)**0.5, size=(n_males,))
    male_weights = np.random.normal(loc=176.25, scale=(1.2292e-2)**0.5, size=(n_males,))
    male_footsizes = np.random.normal(loc=11.25, scale=(9.1667e-1)**0.5, size=(n_males,))
    male_categories = np.zeros(n_males,)
    # Combine into male dataset
    male_data = np.column_stack((male_heights, male_weights, male_footsizes, male_categories)).astype(float)

    female_heights = np.random.normal(loc=5.4175, scale=(9.7225e-2)**0.5, size=(n_females,))
    female_weights = np.random.normal(loc=132.5, scale=(5.5833e-2)**0.5, size=(n_females,))
    female_footsizes = np.random.normal(loc=7.5, scale=(1.6667)**0.5, size=(n_females,))
    female_categories = np.ones(n_females,)
    # Combine into female dataset
    female_data = np.column_stack((female_heights, female_weights, female_footsizes, female_categories)).astype(float)

    # Merge data
    merged_data = np.row_stack((male_data, female_data))
    
    # Split into x, y
    splitted = np.hsplit(merged_data, [3])
    x, y = splitted[0], splitted[1].reshape(n_samples,).astype(int)

    # Make noise
    x += np.random.normal(size=(n_samples, 3))

    return x, y


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    x, y = generate_data(seed=42, n_samples=100) # shapes = (1000, 3) and (1000,)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    # Save'n'load
    clf.save("./gnb.pkl")
    clf = GaussianNB.load("./gnb.pkl")

    pred = clf.predict(x_test)
    acc = clf.score(x_test, y_test)

    print(f"Score:{acc:.2f}")
    print("Predictions:", pred[:5])
    print("True values:", y_test[:5])