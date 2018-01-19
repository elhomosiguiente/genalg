from bitarray import bitarray
from copy import deepcopy
from functools import wraps
import math
import random
import unittest

length_prefix = 2

def bitarray_to_bytes(bitarr):
    return bitarr.length().to_bytes(length_prefix, byteorder='big') + bitarr.tobytes()

def bitarray_from_bytes(bites):
    length = int.from_bytes(bites[:length_prefix], byteorder='big')
    bitarr = bitarray()
    bitarr.frombytes(bites[length_prefix:])
    return bitarr[:length]

def unexpected_type(name, exp, val):
    raise TypeError('expected "%s" to be %s, got %s' % (name, exp, type(val)))

## from https://stackoverflow.com/a/15577293 ##

def argtypes(**decls):

    def decorator(f):
        code = f.__code__
        names = code.co_varnames[:code.co_argcount]

        @wraps(f)
        def decorated(*args, **kwargs):
            for argname, argtypes in decls.items():
                try:
                    val = args[names.index(argname)]
                except ValueError:
                    val = kwargs.get(argname)
                if argtypes == callable:
                    if not callable(val):
                        unexpected_type(argname, 'function', val)
                elif not isinstance(val, argtypes):
                    unexpected_type(argname, argtypes, val)
            return f(*args, **kwargs)
        return decorated
    return decorator

################################################

class Op():

    @argtypes(key=str, f=callable)
    def __init__(self, key, f):
        self.key = key
        self.f = f

    def __str__(self):
        return self.key

def revise(tokens):
    return match_num(tokens, [])

def match_num(tokens, acc):
    if not tokens:
        return acc
    elif isinstance(tokens[0], int):
        acc.append(tokens[0])
        return match_op(tokens[1:], acc)
    return match_num(tokens[1:], acc)

def match_op(tokens, acc):
    if not tokens:
        return acc
    elif isinstance(tokens[0], Op):
        acc.append(tokens[0])
        return match_num(tokens[1:], acc)
    return match_op(tokens[1:], acc)

def eval(tokens):
    if not tokens:
        return 0
    else:
        return do_eval(tokens[1:], tokens[0])

def do_eval(tokens, acc):
    if len(tokens) < 2:
        return acc
    op = tokens[0]
    num = tokens[1]
    acc = op.f(acc, num)
    return do_eval(tokens[2:], acc)

class Genome():

    @argtypes(tokens=list)
    def __init__(self, tokens):
        self.enc_by_key = {}
        self.token_by_key = {}
        self.token_by_enc = {}
        self.tokens = []
        self.gene_length = math.floor(math.log2(len(tokens))) + 1
        fmt = '0%db' % self.gene_length
        for i, token in enumerate(tokens):
            if isinstance(token, int):
                key = token
            elif isinstance(token, Op):
                key = token.key
            else:
                unexpected_type('token', (int,Op,), token)
            bitarr = bitarray(format(i, fmt))
            enc = bitarray_to_bytes(bitarr)
            self.enc_by_key[key] = enc
            self.token_by_key[key] = token
            self.token_by_enc[enc] = token

    @argtypes(keys=list)
    def encode(self, keys):
        bitarr = bitarray()
        for key in keys:
            enc = self.enc_by_key[key]
            bitarr.extend(bitarray_from_bytes(enc))
        return bitarr

    def get_token(self, bitarr, i):
        enc = bitarray_to_bytes(bitarr[i:i+self.gene_length])
        try:
            return self.token_by_enc[enc]
        except KeyError:
            return None

    def decode(self, input):
        if isinstance(input, bitarray):
            bitarr = input
        elif isinstance(input, bytes):
            bitarr = bitarray_from_bytes(input)
        else:
            unexpected_type('input', (bitarray,bytes,), input)
        tokens = [self.get_token(bitarr, i) for i in range(0, bitarr.length(), self.gene_length)]
        return revise(tokens)

    def new_chrom(self, input):
        if isinstance(input, bitarray):
            bitarr = input 
            tokens = self.decode(input)
        elif isinstance(input, bytes):
            bitarr = bitarray_from_bytes(input)
            tokens = self.decode(input)
        elif isinstance(input, list):
            bitarr = self.encode(input)
            tokens = input
        else:
            unexpected_type('input', (bitarray,bytes,list,), input)
        return Chromosome(bitarr, tokens)

class Chromosome():

    @argtypes(bitarr=bitarray, tokens=list)
    def __init__(self, bitarr, tokens):
        self.bitarr = bitarr
        self.tokens = tokens
        self.value = eval(tokens)

    def bytes(self):
        return bitarray_to_bytes(self.bitarr)

    def fitness(self, objective, target):
        return objective(target, self.value)

    def copy_bitarray(self):
        return self.bitarr.copy()

    def __str__(self):
        return ''.join(['%s' % token for token in self.tokens]) + '=%s' % self.value

def rand(nums, last):
    x = random.randint(0, last-1)
    for i, num in enumerate(nums):
        if num > x:
            return i
    raise ValueError('expected num < %d, got %d' % (last, x))

class Environment():

    @argtypes(genome=Genome, chrom_length=int, cross_rate=float, max_iters=int, mut_rate=float, objective=callable, pop_size=int, target=int)
    def __init__(self, **kwargs):
        self.genome = kwargs.get('genome')
        self.chrom_length = kwargs.get('chrom_length')
        self.cross_rate = kwargs.get('cross_rate')
        self.max_iters = kwargs.get('max_iters')
        self.mut_rate = kwargs.get('mut_rate')
        self.objective = kwargs.get('objective')
        self.pop_size = kwargs.get('pop_size')
        self.target = kwargs.get('target')
        self.pop = []
        for _ in range(self.pop_size):
            bitarr = bitarray([random.choice([False, True]) for _ in range(self.chrom_length)])
            chrom = self.genome.new_chrom(bitarr)
            self.pop.append(chrom)

    def set_target(self, target):
        self.target = target

    def copy_chrom(self, i):
        return deepcopy(self.pop[i])

    def copy_pop(self):
        return deepcopy(self.pop)

    def total_fitness(self):
        fitness = 0 
        for chrom in self.pop:
            fitness += chrom.fitness(self.objective, self.target)
        return fitness

    def chrom_fitness(self, chrom):
        return chrom.fitness(self.objective, self.target)

    def try_crossover(self, bitarr1, bitarr2):
        if self.cross_rate < random.random():
            return False
        end = min(bitarr1.length(), bitarr2.length()) 
        start = random.randint(0, end-1)
        temp = bitarr1[start:end]
        bitarr1[start:end] = bitarr2[start:end]
        bitarr2[start:end] = temp
        return (start, end)

    def try_mutate(self, bitarr1, bitarr2):
        xs = []
        ys = []
        for x, b in enumerate(bitarr1):
            if self.mut_rate >= random.random():
                bitarr1[x] = not b
                xs.append(x)
        for y, b in enumerate(bitarr2):
            if self.mut_rate >= random.random():
                bitarr2[y] = not b
                ys.append(y)
        return (xs, ys)

    def iter(self):
        last = 0
        nums = []
        for i, chrom in enumerate(self.pop):
            fitness = self.chrom_fitness(chrom)
            if fitness == float("inf"):
                return chrom
            nums.append(round(fitness * 1000) + last)
            last = nums[i]
        new_pop = []
        for _ in range(0, self.pop_size, 2):
            i = rand(nums, last)
            j = rand(nums, last)
            while i == j:
                j = rand(nums, last)
            bitarr1 = self.pop[i].copy_bitarray()
            bitarr2 = self.pop[j].copy_bitarray()
            self.try_crossover(bitarr1, bitarr2)
            self.try_mutate(bitarr1, bitarr2)
            chrom1 = self.genome.new_chrom(bitarr1)
            chrom2 = self.genome.new_chrom(bitarr2)
            new_pop.append(chrom1)
            new_pop.append(chrom2)
        self.pop = new_pop
        return False

    def run(self):
        iters = 0
        chrom = False 
        while not chrom and iters < self.max_iters:
            chrom = self.iter()
            iters += 1
        return (chrom, iters)

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

def div(x, y):
    if not y:
        return x
    return x / y

plus = Op('+', add)
minus = Op('-', sub)
multiply = Op('*', mul)
divide = Op('/', div)

class TestGenome(unittest.TestCase):

    def setUp(self):
        self.genome = Genome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, plus, minus, multiply, divide])

    def test_encode(self):
        bits = self.genome.encode([1, '+', 2, '-', 3])
        assert bits == bitarray('00011010001010110011')

    def test_decode(self):
        tokens = self.genome.decode(bitarray('00011010001010110011'))
        assert tokens == [1, plus, 2, minus, 3]

    def test_decode_invalid(self):
        tokens = self.genome.decode(bitarray('0010001010101110101101110010'))
        assert tokens == [2, plus, 7]

    def test_eval(self):
        value = eval([1, plus, 2, multiply, 3, minus, 4]) 
        assert value == 5

    def test_chromosome(self):
        chrom = self.genome.new_chrom(bitarray('011010100101110001001101001010100001'))
        fitness = chrom.fitness(objective, 42)
        assert chrom.value == 23
        assert fitness == 1/19

def objective(target, value):
    diff = abs(target - value)
    if diff > 0:
        return 1 / diff
    return float("inf")


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        genome = Genome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, plus, minus, multiply, divide])
        self.env = Environment(
            genome=genome, 
            objective=objective,
            chrom_length=300,
            cross_rate=0.7,
            max_iters=400,
            mut_rate=0.01,
            pop_size=100,
            target=50
        )

    def test_crossover(self):
        bitarr1 = self.env.pop[0].copy_bitarray()
        bitarr2 = self.env.pop[1].copy_bitarray()
        copy1 = bitarr1.copy()
        copy2 = bitarr2.copy()
        res = self.env.try_crossover(bitarr1, bitarr2)
        if res is False:
            assert bitarr1 == copy1 
            assert bitarr2 == copy2
        else:
            (start, end) = res
            assert bitarr1 == copy1[:start] + copy2[start:end] + copy1[end:]
            assert bitarr2 == copy2[:start] + copy1[start:end] + copy2[end:]

    def test_mutate(self):
        chrom1 = self.env.copy_chrom(0)
        chrom2 = self.env.copy_chrom(1)
        bitarr1 = chrom1.copy_bitarray()
        bitarr2 = chrom2.copy_bitarray()
        (xs, ys) = self.env.try_mutate(chrom1.bitarr, chrom2.bitarr)
        if xs:
            for x in xs:
                bitarr1[x] = not bitarr1[x]
        if ys:
            for y in ys:
                bitarr2[y] = not bitarr2[y]
        assert bitarr1 == chrom1.bitarr
        assert bitarr2 == chrom2.bitarr

    def test_iter(self):
        fitness_before = self.env.total_fitness()
        pop_before = self.env.copy_pop()
        chrom = self.env.iter()
        pop_after = self.env.copy_pop()
        if not chrom:
            for chrom in pop_before:
                assert float("inf") != self.env.chrom_fitness(chrom)
            fitness_after = self.env.total_fitness()
            assert fitness_before < fitness_after
        else:
            assert float("inf") == self.env.chrom_fitness(chrom)

    def test_run(self):
        (chrom, iters) = self.env.run()
        if chrom:
            print(chrom)
            assert float("inf") == self.env.chrom_fitness(chrom)
            assert iters < self.env.max_iters
        else:
            assert iters == self.env.max_iters

if __name__ == '__main__':
    unittest.main()
