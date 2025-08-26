# Solving the Traveling Salesperson Problem (TSP) with a Genetic Algorithm

This project is a simple and educational implementation of a Genetic Algorithm (GA) to solve the classic Traveling Salesperson Problem (TSP) in Python. The algorithm attempts to find the shortest possible route to visit a set of cities and return to the starting city.



---

## 📜 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [License](#-license)

---

## 🎯 About the Project

The Traveling Salesperson Problem (TSP) is one of the most famous optimization problems in computer science. Due to its high complexity (it's NP-Hard), finding a perfect, optimal solution for a large number of cities is practically impossible.

A Genetic Algorithm, inspired by the process of natural selection, provides a smart approach to search the solution space and find optimal or near-optimal solutions. This project serves as an educational tool to demonstrate the core steps of a GA applied to TSP.

---

## ✨ Features

* Implemented in Python with the NumPy library.
* Chromosome representation using **Permutation Encoding**.
* **Fitness Function** based on the inverse of the total tour distance.
* **Roulette Wheel Selection** mechanism.
* A specialized crossover operator for TSP called **Ordered Crossover (OX1)**.
* **Swap Mutation** operator.
* Implementation of **Elitism** to preserve the best solution in each generation.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You will need Python 3.6 or higher installed.

* First, install the NumPy library.
    ```bash
    pip install numpy
    ```

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/alipgm/TSP-Genetic-Algorithm.git](https://github.com/alipgm/TSP-Genetic-Algorithm.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd TSP-Genetic-Algorithm
    ```

---

## 💻 Usage

To run the algorithm, simply execute the Python script:

```bash
python core.py
```

The program will start running and will print the best distance found in each generation. At the end, it will display the best overall tour and its total distance.

```
Starting Genetic Algorithm to solve TSP...
Generation 1: Best Distance = 21.90
Generation 2: Best Distance = 21.90
...
Generation 100: Best Distance = 18.21

-----------------------------------------
Evolution complete.
Best tour found: 0 -> 1 -> 2 -> 4 -> 3 -> 0
Total distance: 18.21
```

---

## 🧠 How It Works

The algorithm uses an evolutionary cycle to progressively improve a population of tours.

1.  **Initial Population:** A collection of completely random tours is created.
2.  **Evaluation:** The total distance of each tour is calculated, and its fitness score (1 / Distance) is determined.
3.  **Selection:** Better tours (with higher fitness scores) are selected as parents for the next generation using the Roulette Wheel method.
4.  **Crossover:** New offspring are generated from the selected parents using Ordered Crossover, which combines features from both parents.
5.  **Mutation:** With a small probability, two cities in an offspring's tour are randomly swapped to prevent premature convergence and explore new solutions.
6.  **New Population:** The new population replaces the old one, and the cycle repeats.

---

## ⚙️ Configuration

You can adjust the algorithm's hyperparameters at the beginning of the `core.py` file to control its behavior:

* `POPULATION_SIZE`: The number of tours in each generation. (Higher value = more diversity, slower execution).
* `NUM_GENERATIONS`: The total number of generations the algorithm will run.
* `MUTATION_RATE`: The probability that an offspring will undergo mutation.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.