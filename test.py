
def get_random_on_shpere(d=10):
    x_rand = np.random.normal(size=d)
    x_rand = x_rand / np.linalg.norm(x_rand)
    return x_rand

w_lists = [[] for _ in range(5)]
for _ in range(10):
    print(_)
