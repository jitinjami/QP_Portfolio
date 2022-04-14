# Quadratic Programming for Investment Portfolio Optimization

## Installation
### Using Conda
Execute
`$ conda env create -f env.yml`

to create environment. With this environment all scripts should be executable.
Follow instructions on 'https://scaron.info/doc/qpsolvers/installation.html' for installing qpsolvers and respective solvers included in that.

Please note the qpOases only works on Linux systems, we were unable to port that to MacOS.

## Structure

### Portfolio
The file [portfolio.py](./portfolio.py) contains the definition of a portfolio class. 
This class generates a random portfolio with n assets and a Timeseries of length T.
This class contains the covariance matrix, the asset weiths the asset returns and other helper objects.

### Constaints

The constrains class is defined in [constraints.py](./constraints.py). The constraints object is initialized once, 
setting some basic variables but no actual constraints. To add constraints the add_<constraints>_constraings() methods 
have to be called. They generate the Matrices A,G and vectors h,b for Ax = b and Gx <= h according to the semantic 
constraints which are added. Helperfunctions ensure that if other constraints have already been added, the new 
constraints are concatenated with the old ones row-wise.

The constraints object holds the matrices as well as some other variables like shapes and interval bounderies if needed 
for these constraints.

### Problem formulation

The file [problem_formulations.py](./problem_formulations.py) defines the system Matrices for the Problem in standart 
formulation min {-t cT x + 1/2 xT C x | G x <= h, A x = b}. The matrix C and the vector c is calculated directly, the 
constraints are added to the constraints object as described above. The return value of the problem fomulation contains 
everythin to solve the problem.

### Optimizers

The file [optimizers.py](./optimizers.py) contains the optimizers. If contains two wrappers for a solver provided by 
the cvxopt library. One solver uses sparse matrices and the other full matrices. We tried to implement an optimized 
version however did not finish it. This method is thus depricated.

### Comparison

The file [comparison.py](./comparison.py) generates data from the solution of various open source qp solvers like cvxopt, quadprog, ecos, qpOases on our problem formulations. The comparison is done between all methods for the first 2 problem formulations but only done between cvxopt and quadprog for the 2 complex models. It also generates the plotting images as seen on the report (Figure 3).
It uses an opensource library qpsolvers that enables us to use many different algorithms apart from just cvxopt.

### Main

The file [main.py](./main.py) contains a script to perform runtime measurements. In the first section of the main method, you can define specific parameters. Behind all parameters a short description of the variable can be found. If the runtime is too long, decreasing m or the upper bound of n might help.

## Usage

The Portfolio class helps us generate a random portfolio to begin our analysis with. In the Constraint class, we have different kinds of constraints that are used for formulating our problems. The problem formulation function file uses the Constraints class in the format we can use to solve. Let it be open source solvers or kerna optmizer.

One can use the Portfolio class to define a portfolio of their own and the problem formulation function file to define their own unique problem formulations or use existing formulations where one can change different aspects (like changing the risk factor etc)

To run the time measurements just run `$ python3 main.py`. Parameters have to be adjusted in the soure code. The results will be written into the folder ./data

## About
### Built With

- Python
- YAML
- cvxopt

### Authors

**Valentin Bacher**

- [Email](mailto:valentin.bacher@fau.de?subject=pq_portfolio "pq_portfolio")

**Jitin Jami**

- [Email](mailto:jitin.jami@usi.ch?subject=pq_portfolio "pq_portfolio")
