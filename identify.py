import torchvision
import torch

# TODO Platonic ideals
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
    ]
    # TODO add more numbers
    # 1...
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
            cell_score = (a - b) ** 2
            score += cell_score
    return score


def predict(img):
    """Predicts which digit the image represents"""
    t = to_tensor(img)
    # TODO loop through possible images and get a compare score for each platonic
    # TODO return the int with the best score


# get dataset
mnist = torchvision.datasets.MNIST("data", download=True)

# get an arbitrary digit at index 0
item = mnist.__getitem__(0)
img = item[0]
result = item[1]

torch.set_printoptions(precision=0, linewidth=160)

# print(predict(img))
