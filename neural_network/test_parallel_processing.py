"""
Script to test parallel processing in python.
"""
import os
from multiprocessing import Process, current_process

def square(number):
    result = number * number

    # We can use the os module in Python to print out the Process ID
    # assigned to the call of this function
    #process_id = os.getpid()
    #print("Process ID: " + str(process_id))

    # We can also use the current_process function to get the name
    # of the current process
    process_name = current_process().name
    print("Process name: ", process_name)
    print("The number " + str(number) + " squares to " + str(result))

def main():
    """
    """
    print('Number of processors: ', mp.cpu_count())

if __name__ == '__main__':

    processes = []
    numbers = [1, 2, 3, 4]

    for number in numbers:
        process = Process(target=square, args=(number,))
        processes.append(process)

        # Processes are spawned by creating a Process object and
        # then calling its start() method
        process.start()
