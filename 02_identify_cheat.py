from PIL import Image
import torchvision
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=0, linewidth=160)

# Approach 2: Cheat
# Copying the methodology from the fast.ai book:
# Averaging over the training dataset to get the Platonic digits, instead of hand-defining them


# get dataset
mnist = torchvision.datasets.MNIST("data", download=True)


def to_img(t):
    """Converts a pytorch tensor into an image"""
    return torchvision.transforms.ToPILImage()(t)


# get tensors from dataset for a particular digit #
def get_n_tensors(n):
    return [mnist.data[i] for i in range(len(mnist.data)) if mnist.targets[i] == torch.tensor(n)]


def make_average(t): return (torch.stack(t).float() / 255).mean(0)


platonic_digits = [make_average(get_n_tensors(n)) for n in range(10)]

# How to show an img:
# to_img(t).show()

# stacked_twos = make_average(get_n_tensors(2))
# to_img(stacked_twos).show()

def predict(img_tensor):
    """Predicts which digit the image represents"""
    scores = [F.l1_loss(digit, img_tensor) for digit in platonic_digits]

    lowest = 0
    lowest_score = float("inf")
    for i, s in enumerate(scores):
        if s < lowest_score:
            lowest = i
            lowest_score = s
    return lowest


def get_results():
    correct = 0
    dataset_length = len(mnist.data)
    for i in range(dataset_length):
        img = mnist.data[i]
        result = mnist.targets[i]

        # img = mnist.data[i]
        # result = mnist.targets[i]
        prediction = predict(img)

        if prediction == result:
            correct += 1

    return correct / dataset_length


print(get_results())
