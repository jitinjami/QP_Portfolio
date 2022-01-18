# QP_Portfolio

## Installation
### Using Conda
Execute
`$ conda env create -f env.yml`

to create environment. With this environment all scripts should be executable.

## Structure

### Portfolio
The file [portfolio.py](./portfolio.py) contains the definition of a portfolio class. 
This class generates a random portfolio with n assets and a Timeseries of length T.
This class contains the covariance matrix, the asset weiths the asset returns and other helper objects.

### Constaints

The constrains class is defined in [constraints.py](./comparison.py). The constraints object is initialized once, 
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

the file [optimizers.py](./optimizers.py) contains the optimizers. If contains two wrappers for a solver provided by 
the cvxopt library. One solver uses sparse matrices and the other full matrices. We tried to implement an optimized 
version however did not finish it. This method is thus depricated.

### Others

@Jitin: feel free to describe some of your files...

## Usage

@Jitin

## About
### Built With

- Python
- YAML
- MS Excel

### Authors

**Valentin Bacher**

- [Email](mailto:valentin.bacher@fau.de?subject=pq_portfolio "pq_portfolio")

**Jitin Jami**

- [Email](mailto:jitin.jami@usi.ch?subject=pq_portfolio "pq_portfolio")