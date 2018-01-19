from genalg import *

if __name__ == '__main__':

    ## define operator and objective functions

    def add(x, y):
        return x + y

    def sub(x, y):
        return x - y

    def mul(x, y):
        return x * y

    def div(x, y):
        return x / y

    def objective(target, value):
        diff = abs(target - value)
        if diff > 0:
            return 1 / diff
        return float("inf")

    # create operators

    plus = Op('+', add)  

    minus = Op('-', sub)

    multiply = Op('*', mul)

    divide = Op('/', div)

    ## create genome

    genome = Genome([1, 2, 3, 4, 5, 6, 7, 8, 9, plus, minus, multiply, divide])

    ## create and run environment

    env = Environment(
        genome=genome,
        chrom_length=300,
        cross_rate=0.7,
        max_iters=400,
        mut_rate=0.01,
        objective=objective,
        pop_size=100,
        target=50
    )

    (soln, iters) = env.run()

    ## print the solution and the number of iterations
    ## if solution wasn't found, iters should equal max_iters

    print('Solution (iters=%d): %s' % (iters, soln))

    ## example output:
    ## Solution (iters=4): 2-2-9+8-7+3+6-1+4+4*7-7+1=50
    ## Note: doesn't follow order of operations; just go left to right