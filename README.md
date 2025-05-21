PyOpenTCU - Architectural Simulator for GPU Tensor Core Units (TCUs)
=====================================================================

Overview
--------
PyOpenTCU is an open-source Python-based simulator that models the structural logic and internal organization of Tensor Core Units (TCUs) in GPUs. The tool provides a highly flexible framework for studying permanent hardware faults, evaluating CNN performance on faulty hardware, and designing fault-tolerant architectures.

Inspired from:
- Raihan et al., 2019: https://doi.org/10.1109/ISPASS.2019.00016
- Boswell et al., 2019: https://patents.google.com/patent/US10338919B2/en

PyOpenTCU supports the development of accurate error models in safety-critical systems such as autonomous robotics and automotive applications.

What is a TCU?
--------------
Modern GPUs integrate in-chip accelerators to boost machine learning performance. These accelerators, known as matrix core processing units or TCUs, perform matrix-matrix operations using a 4x4 array of dot-product units (DPUs). Each TCU can compute 16 multiply-and-add (MaA) operations per cycle on 4x4 matrix segments (A, B, and C matrices), enabling efficient MxM matrix computations.

TCUs exploit spatial and temporal locality through special registers known as buffers (or near-registers) and through optimized scheduling policies like thread grouping and octets. An octet comprises pairs of thread groups that process data segments efficiently to hide memory latency and improve performance.

GPUs use dedicated assembly instructions (e.g., HMMA) to support larger matrices (e.g., 16x16), by chaining multiple 4x4 segment operations and using intermediate buffer storage.

General Architecture of a TCU
------------------------------------

![TCU Architecture](Docs/images/architecture.png)

Description:
- Shows two TCUs inside an SM.
- Each TCU includes:
  * A 4x4 array of Dot-Product Units (DPUs)
  * Buffers for matrix segments A, B, and C
  * A scheduler
- Octets (thread group pairs) and their color-coded matrix segments are visualized.
- Buffers store input and intermediate segments (A, B, C) for reuse and reduced latency.

TCUs also support multi-precision arithmetic with multiple number formats.

Number Formats Supported
------------------------
1. Floating-Point (FP):
   - IEEE-754 standard
   - Supports 16-bit and 32-bit precision
   - Uses a sign bit, exponent, and fraction

2. Posit Format:
   - Uses sign, regime, exponent, and fraction
   - Scales with useed^k where useed = 2^(2^es)
   - Offers high accuracy near 1.0 and avoids overflow/underflow
   - Ideal for deep learning tasks like CNN inference

These formats are implemented using:
- SoftFloat and SoftPosit Python libraries
- https://pypi.org/project/sfpy/ (accessed on 30 January 2024)

Key Features
------------
- Models TCU components: DPU arrays, buffers, controllers
- Supports real GPU-like structures (e.g., NVIDIA Volta with 2 TCUs per SM)
- Configurable number formats and precision
- Fault injection capabilities for reliability studies
- Simulates thread group behavior and matrix segment reuse
- Allows exploration of FP and Posit at 16- and 32-bit widths
- Suitable for architectural design, CNN testing, and safety analysis

Example Use Cases
-----------------
- Analyze CNN inference under hardware faults
- Design and simulate new TCU configurations
- Study the effects of precision on performance and accuracy
- Explore fault-tolerant computing techniques in GPU-based systems

Installation
------------
Clone the repository:

    git clone https://github.com/TheColombianTeam/pyOpenTCU.git
    cd pyOpenTCU

Install dependencies:

    pip install -r requirements.txt

If you encounter issues when installing sfpy, please read the BUILDING guide:
    https://github.com/billzorn/sfpy/blob/master/BUILDING.md


Usage
----------------------

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

Publication References
----------------------

The following research works have used or cited PyOpenTCU:

1. Robert Limas Sierra, Juan-David Guerrero-Balaguera, Josie E. Rodriguez Condia, Matteo Sonza Reorda,  
   "Exploring Hardware Fault Impacts on Different Real Number Representations of the Structural Resilience of TCUs in GPUs",  
   *Electronics*, Vol. 13, No. 3, Article 578, 2024.  
   URL: https://www.mdpi.com/2079-9292/13/3/578

2. Robert Limas Sierra, Juan-David Guerrero-Balaguera, Josie E. Rodriguez Condia, Matteo Sonza Reorda,  
   *Analyzing the Impact of Different Real Number Formats on the Structural Reliability of TCUs in GPUs*,  
   2023 IFIP/IEEE 31st International Conference on Very Large Scale Integration (VLSI-SoC),  
   Pages 1–6, DOI: https://doi.org/10.1109/VLSI-SoC57769.2023.10321881

3. Robert Limas Sierra, Juan-David Guerrero-Balaguera, Francesco Pessia, Josie E. Rodriguez Condia, Matteo Sonza Reorda,  
   *Analyzing the Impact of Scheduling Policies on the Reliability of GPUs Running CNN Operations*,  
   2024 IEEE 42nd VLSI Test Symposium (VTS),  
   Pages 1–7, DOI: https://doi.org/10.1109/VTS60656.2024.10538940

4. **VLSI-SoC 2023: Innovations for Trustworthy Artificial Intelligence**  
   eBook ISBN: 978-3-031-70947-0  
   Print ISBN: 978-3-031-70946-3  
 

To include your work in this list, please submit a pull request with the reference.

To include your work in this list, please submit a pull request with the reference.

Contact
-------
For questions, contributions, or collaboration:
Open an issue on GitHub or reach out via institutional contact.