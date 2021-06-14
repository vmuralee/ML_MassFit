from load_data import load_data
from tqdm import tqdm

def train(self, epochs, batch_size=128, save_interval=50):
    # Load the dataset
    (X_train, y_train), (X_test, y_true) = load_data('W2LNu10000Events_13Tev.root')

    half_batch = int(batch_size / 2)
    batch_count = X_train.shape[0]/batch_size

    for epoch in range(epochs):
        print('-'*15,f' Epoch {epoch}','-'*15)
        for _ in tqdm(range(batch_count)):
            

