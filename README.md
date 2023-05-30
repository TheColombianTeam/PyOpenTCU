# Tensor Core

Tensor Core is a software model architecture based on [1](??) and [2](??)

## Installation

Install the dependencies using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requeriments.txt
```

## Usage

You can see a basic example inside the sim.py file. However, the general use of this model is as follows:

```python
from Tensor import Tensor
import numpy as np

# create the object
tensor = Tensor()

# create A, B and C matrix
# matrix should have a 16x16 shape
a = np.random.rand(16, 16)
b = np.random.rand(16, 16)
c = np.random.rand(16, 16)

# return the matrix result
# d = (A * B) + C
d = tensor.mul(a, b, c)

print(d)
```

## Example of tail

Check the custom_mul.py file. This file has a simple tail multiply algorithm. The algorithm implemented is just a first approach; **it must be improved it.**

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.