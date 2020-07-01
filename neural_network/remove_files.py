"""
Script to remove certain files extension.
"""

import os

def main():
    """
    Remove certain files.
    """

    # Current directory
    cwd = os.getcwd()

    # Itens in the directory (files, folders)
    dirs = os.listdir(cwd)

    # Extensions to remove
    exts = ('.npy')

    # Max size of a file (bytes)
    max_size = 50000000 # 50 MB

    # Choose an option
    option = input('Do you want remove the files with the ' + str(exts) + ' extensions? [y/n]\n')

    # Counter
    cnt = 0

    if option == 'y':
        # Loop throught the files
        for root, dirs, files in os.walk(cwd):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.stat(file_path).st_size
                if file.endswith(exts) or file_size >= max_size:
                    os.remove(file_path)
                    print('File: ' + str(file) + ' removed')
                    # print('File path: ' + str(file_path))
                    # print('File size: ' + str(file_size) + '\n')
                    cnt = cnt + 1
    elif option == 'n':
        print('Script stopped!')
    else:
        print('Invalid entry!')

    print('Done! ' + str(cnt) + ' files removed!')

if __name__ == '__main__':
    main()
