import numpy as np


def gen_fake_data():
    fake_data = np.random.randint(1, 30522, size=(4, 64)).astype(np.int64)
    fake_label = np.array([0, 1, 1, 0]).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()