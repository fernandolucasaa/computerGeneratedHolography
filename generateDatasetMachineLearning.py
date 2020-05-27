import numpy as np

def sobel(image):
    w = len(image)
    kernel_x = np.array([ [ 1, 0,-1],
                          [ 2, 0,-2],
                          [ 1, 0,-1] ])

    kernel_y = np.array([ [ 1, 2, 1],
                          [ 0, 0, 0],
                          [-1,-2,-1] ])
    
    grad_x = np.zeros([w - 2, w - 2])
    grad_y = np.zeros([w - 2, w - 2])
    
    for i in range(w - 2):
        for j in range(w - 2):
            grad_x[i, j] = sum(sum(image[i : i + 3, j : j + 3] * kernel_x))
            grad_y[i, j] = sum(sum(image[i : i + 3, j : j + 3] * kernel_y))
            if grad_x[i, j] == 0:
                grad_x[i, j] = 0.000001 
    
    mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
    ang = np.arctan(grad_y / (grad_x + np.finfo(float).eps))
  
    # Gradient computation
    return [mag,ang]

def pixel_count(image):
    pc_x = np.zeros(len(image))
    pc_y = np.zeros(len(image))
  
    # Pixel count computation
    for i in range(len(image)):
        pc_x[i] = np.count_nonzero(image[i, :])
        pc_y[i] = np.count_nonzero(image[:, i])

    return [pc_x, pc_y]

class Element:
    def __init__(self, data, target):
        self.target = target
        self.features = {'var' : 0,
                         'std' : 0,
                         'mean_grad_M' : 0,
                         'std_grad_M'  : 0,
                         'mean_grad_D' : 0,
                         'std_grad_D'  : 0,
                         'mean_PC_X'   : 0,
                         'std_PC_X'    : 0,
                         'active_PC_X' : 0,
                         'mean_PC_Y'   : 0,
                         'std_PC_Y'    : 0,
                         'active_PC_Y' : 0}
        self.computeFeatures(data)
        
    def computeFeatures(self, data):
        """
        Compute the 12 features of each image (8 images in total) and returns the average of each feature.
        """
        # Auxiliar variable
        matrix = np.zeros((8,12))
        pos = 0 # 0..7
        
        # Feature computation for each image
        for image in data:
            # Feature computation
            mag, ang = sobel(image)
            pcx, pcy = pixel_count(image)
            matrix[pos][0] = (np.var(image))
            matrix[pos][1] = (np.std(image))
            matrix[pos][2] = (np.mean(mag))
            matrix[pos][3] = (np.std(mag))
            matrix[pos][4] = (np.mean(ang))
            matrix[pos][5] = (np.std(ang))
            matrix[pos][6] = (np.mean(pcx))
            matrix[pos][7] = (np.std(pcx))
            matrix[pos][8] = (np.count_nonzero(pcx))
            matrix[pos][9] = (np.mean(pcy))
            matrix[pos][10] = (np.std(pcy))
            matrix[pos][11] = (np.count_nonzero(pcy)) 
            # Update variable
            pos = pos + 1
        
        # Features' average
        keys = ['var', 'std', 'mean_grad_M', 'std_grad_M', 'mean_grad_D', 'std_grad_D', 'mean_PC_X', 'std_PC_X', 'active_PC_X', 'mean_PC_Y', 'std_PC_Y', 'active_PC_Y']
        for i in range(12):
            self.features[keys[i]] = sum(matrix[:,i]/8)
    
    def __print__(self):
        print("Element target: " + str(self.target))
        print("Element features:")
        print(self.features)

def createClassElement(data, target):
    return Element(data, target)

class Dataset:
    def __init__(self, array, length, y_list):
        self.array = array   # (nb_examples, 8, 200, 200)
        self.length = length # nb_examples
        self.wd_list = []
        self.wd_list = self.createElements(y_list)
        self.features = [[float(f) for f in elem.features.values()] for elem in self.wd_list]
        self.raw_targets  = [[self.wd_list[i].target] for i in range(self.length)]
    
    def createElements(self, y_list):
        elements = []
        pos = 0
        for row in self.array:
            # row : (8, 200, 200)
            elements.append(Element(row, y_list[pos]))
            pos = pos + 1
        return elements

def createClassDataset(array, length, y_list):
    return Dataset(array, length, y_list)

def computeTargets(array):
    targets = []
    y = 1
    for i in range(len(array)):
        targets.append(y)
        if((i+1)%5 == 0):
            y = y + 1
    return targets
    
def generate_dataset(array):
    """
    bla bla bla
    """
    target_list = computeTargets(array)
    dataset = Dataset(array, len(array), target_list)
    
    return dataset

def cvt_obj_nparray(dataset):
    '''
    This function converts an object as input in 2 numpy arrays as output.
    X: 2D matrix (nb of examples, nb of features)
    Y: vector with targets
    '''
    X = np.zeros((dataset.length, 12))
    Y = np.zeros((dataset.length,))
    for i, elem in enumerate(dataset.wd_list):
        Y[i] = elem.target
        for j, feature in enumerate(elem.features):
            X[i, j] = elem.features[feature]
    return X, Y

def create_data_file(filename):
    '''
    bla bla bla
    '''
    # Load the database (.npy) files 
    wd_results_array = np.load(filename) 

    print("Creating dataset...")
    data_set = generate_dataset(wd_results_array)
    print ("\nFinished creating dataset\n")

    X_array, Y_array = cvt_obj_nparray(data_set)

    return X_array, Y_array

def normalize(arr):
    '''
    Function to normalize the features
    '''
    max_line = np.max(arr, axis=0)
    min_line = np.min(arr, axis=0)
    
    arr = (arr - min_line) / (max_line - min_line)
    
    return arr