"""
Script to test parallel processing in python.
"""
import os
import time
import multiprocessing as mp

def sum_um_to(number):
    time.sleep(0.5)
    result = sum(range(1, number + 1))
    return result

def sum_um_to_2(number, arg1, arg2):
    time.sleep(0.5)
    result = sum(range(1, number + 1))
    result = result + arg1
    result = result*arg2
    return result

def main2():
    numbers = range(50)

    # Compute execution time
    start_time = time.time()

    # Init Pool Class
    pool = mp.Pool(mp.cpu_count())
    result = [pool.apply(sum_um_to_2, args=(nb, 0, 1)) for nb in numbers]
    print(result)
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    print('\n')
    
    # Compute execution time
    start_time = time.time()

    # Loop
    result = []
    for nb in numbers:
        result.append(sum_um_to_2(nb, 0, 1))

    print(result)
    print('Execution time: %.4f seconds' % (time.time() - start_time))

def main():
    numbers = range(50)

    # Compute execution time
    start_time = time.time()

    # Pool
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(sum_um_to, numbers)

    print(result)
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    print('\n')

    # Compute execution time
    start_time = time.time()

    # Loop
    result = []
    for nb in numbers:
        result.append(sum_um_to(nb))
    
    print(result)
    print('Execution time: %.4f seconds' % (time.time() - start_time))

if __name__ == '__main__':
    # main()
    main2()
