# MNIST

This repo is an experiment in non-ML approaches to creating a digit classifier on the MNIST dataset. It aims to provide a learning experience to better understand the advantages of Stochastic Gradient Descent and also to practice pytorch and tensor manipulation.

The inspiration for this project was taken from the fast.ai course.

## Challenge

Using the MNIST dataset, use a non-ML approach to classify digits as accurately as possible.

I started out with the clue that images can be represented as light/dark matrices.

## Tackling the Problem

### Approach 1: Platonic

_This approach can be found in `01_identify_platonic.py`._

My first approach was to try and 'draw' some numbers in matrix form and then do a simple comparison between my ideal (Platonic) digits and the images. The digits I drew ressembled those from an LCD screen on a digital alarm clock. e.g. here is 3:

```python
[
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
],
```

### Approach 2: Averaging (cheating a bit)

_This approach can be found in `02_identify_cheat.py`._

When Approach 1 did not work very well, I peaked at the high-level solution in the fast.ai book and attempted to implement that. This approach involved averaging over the training data to create the ideal digits, rather than trying to hand-draw them.

It was interesting to me that, although this is certainly not ML, it is still statistical in nature.

## Results

- Approach 1: 0.1477% accuracy
- Approach 2: 0.7118% accuracy

Approach 1 did not go very well. Upon inspecting the training data further, it seems that the hand-drawn characters did simply not match the LCD-like digits I had created. The real data was often clumped in the centre and curvey. My data was often around the edge and at right angles. The two excpetions to this were '1' which drew a straight line down the middle and '0' which had more of a curve (and also more 'mass'), and these two digits were the ones exclusively predicted by my script. I experimented with modifying the digits to match but did not manage to obtain significant improvements.

## Things I learned doing this

- Statistical models probably work better, even if not ML per se. Strict comparisons are unlikely to perform well.
- How to use and manipulate pytorch tensors.
- `map` is awkard in Python. List comprehensions are probably more idiomatic.
- Working in vim without autocomplete or IDE support of any kind is a little tough when working with a new library. I probably just need to tweak my Python setup! VSCode was okay but not as useful as a notebook.
