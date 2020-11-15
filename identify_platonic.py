import torchvision
import torch

torch.set_printoptions(precision=0, linewidth=160)

# Approach 1: Platonic ideals
# Create 'perfect' digits from simplified matrices and compare to those


def to_tensor(img):
    """Converts a PIL image into a pytorch tensor"""
    return torchvision.transforms.ToTensor()(img)


small_platonic_matrices = [
    # 0
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    # 1
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    # 2
    [
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    # 3
    [
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    # 4
    [
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ],
    # 5
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    # 6
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    # 7
    [
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ],
    # 8
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    # 9
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ],
]


def mk_platonic_matrix(n):
    """
    Takes an n ∈ ℕ s.t. 0 <= n <= 9 and returns a 28x28 matrix representing the 'perfect'
    version of that digit
    """
    m = small_platonic_matrices[n]
    return make_bigger(m)


flatten = lambda t: [item for sublist in t for item in sublist]


def make_bigger(matrix):
    """Takes a 7x7 matrix and makes it into a 28x28 version"""
    return flatten(
        list(
            map(
                lambda row: [row, row, row, row],
                list(
                    map(
                        lambda row: flatten(
                            list(map(lambda cell: [cell, cell, cell, cell], row))
                        ),
                        matrix,
                    )
                ),
            )
        )
    )


def compare(t1, t2):
    """
    Returns a score representing the diff between tensors t1 and t2.
    Lower score is better.
    Assumes t1 and t2 are of rank 2.
    """
    score = 0

    for i in range(len(t1)):
        for j in range(len(t1[0])):
            a = t1[i][j]
            b = t2[i][j]
            # square to avoid -ve

            cell_score = abs(a - b)
            score += cell_score
    return score


def predict(img):
    """Predicts which digit the image represents"""
    t = to_tensor(img)
    l = t.tolist()  # convert to python list
    scores = {}
    for n in range(10):
        scores[n] = compare(mk_platonic_matrix(n), t[0].tolist())

    lowest = 0
    lowest_score = float("inf")
    for s in scores.items():
        if s[1] < lowest_score:
            lowest = s[0]
            lowest_score = s[1]
    return lowest


# get dataset
mnist = torchvision.datasets.MNIST("data", download=True)


def get_results():
    correct = 0
    for i in range(1000):
        item = mnist.__getitem__(i)
        img = item[0]
        result = item[1]
        prediction = predict(img)

        if prediction == result:
            correct += 1

        print(f"Prediction: {prediction} | Result: {result}")

    return correct


print(get_results())
