
import numpy as np
import math
from collections import deque


### a function to create a unique increasing ID
### note that this is just a quick-and-easy way to create a global order
### it's not the only way to do it
global_order_counter = 0
def get_next_order():
    global global_order_counter
    rv = global_order_counter
    global_order_counter = global_order_counter + 1
    return rv

### a helper function to convert constants into BackproppableArray objects
def to_ba(x):
    if isinstance(x, BackproppableArray):
        return x
    elif isinstance(x, np.ndarray):
        return BackproppableArray(x)
    elif isinstance(x, float):
        return BackproppableArray(np.array(x))
    elif isinstance(x, int):
        return BackproppableArray(np.array(float(x)))
    else:
        raise Exception("could not convert {} to BackproppableArray".format(x))

### a class for an array that can be "packpropped-through"
class BackproppableArray(object):
    # np_array     numpy array that stores the data for this object
    def __init__(self, np_array, dependencies=[]):
        super().__init__()
        self.data = np_array

        # grad holds the gradient, an array of the same shape as data
        # before backprop, grad is None
        # during backprop before grad_fn is called, grad holds the partially accumulated gradient
        # after backprop, grad holds the gradient of the loss (the thing we call backward on)
        #     with respect to this array
        # if you want to use the same array object to call backward twice, you need to re-initialize
        #     grad to zero first
        self.grad = None

        # an counter that increments monotonically over the course of the application
        # we know that arrays with higher order must depend only on arrays with lower order
        # we can use this to order the arrays for backpropagation
        self.order = get_next_order()

        # a list of other BackproppableArray objects on which this array directly depends
        # we'll use this later to decide which BackproppableArray objects need to participate in the backward pass
        self.dependencies = dependencies

    # represents me as a string
    def __repr__(self):
        return "({}, type={})".format(self.data, type(self).__name__)

    # returns a list containing this array and ALL the dependencies of this array, not just
    #    the direct dependencies listed in self.dependencies
    # that is, this list should include this array, the arrays in self.dependencies,
    #     plus all the arrays those arrays depend on, plus all the arrays THOSE arrays depend on, et cetera
    # the returned list must only include each dependency ONCE
    def all_dependencies(self):
        # TODO: (1.1) implement some sort of search to get all the dependencies

        '''
        First, implement the utility function BackproppableArray.all_dependencies. 
        This function returns all the backproppable arrays on which an array depends in its computation, 
        including indirectly. That is, this function should return a python list containing this array,
        any direct dependencies of this array (i.e. the arrays stored in the dependencies list), 
        any dependencies of those dependencies, any dependencies of those dependencies, and so 
        on indefinitely. To implement this utility function, you'll want to use some sort of graph search 
        algorithm, e.g. breadth first search, using the dependencies field as the edges of graph.

        '''

        #things started crashing out when added self to deps but it shd be like that?? chat wtf
        #add first layer to visited list and to queue 
        queue = deque(self.dependencies)
        visited = set()
        visited.add(self)

        #travel through tree
        while queue:
            current = queue.pop()
            if current not in visited:
                visited.add(current)
                queue.extend(current.dependencies)

        return list(visited)
    

    # compute gradients of this array with respect to everything it depends on
    def backward(self):
        # can only take the gradient of a scalar
        assert(self.data.size == 1)

        # depth-first search to find all dependencies of this array
        all_my_dependencies = self.all_dependencies()

        # TODO: (1.2) implement the backward pass to compute the gradients
        #   this should do the following
        #   (1) sort the found dependencies so that the ones computed last go FIRST
        #   (2) initialize and zero out all the gradient accumulators (.grad) for all the dependencies
        #   (3) set the gradient accumulator of this array to 1, as an initial condition
        #           since the gradient of a number with respect to itself is 1
        #   (4) call the grad_fn function for all the dependencies in the sorted reverse order

        #sort using order field: more recent have bigger order aka are first in backprop
        sorted_deps = sorted(all_my_dependencies, key = lambda x:x.order, reverse = True)

        for dep in sorted_deps:
            dep.grad = np.zeros_like(dep.data)
            # dep.grad = 0

        self.grad = np.ones_like(dep.data)
        # self.grad = 1

        for dep in sorted_deps:
            dep.grad_fn()


    # function that is called to process a single step of backprop for this array
    # when called, it must be the case that self.grad contains the gradient of the loss (the
    #     thing we are differentating) with respect to this array
    # this function should update the .grad field of its dependencies
    #
    # this should just say "pass" for the parent class
    #
    # child classes override this
    def grad_fn(self):
        pass

    # operator overloading
    def __add__(self, other):
        return BA_Add(self, to_ba(other))
    def __sub__(self, other):
        return BA_Sub(self, to_ba(other))
    def __mul__(self, other):
        return BA_Mul(self, to_ba(other))
    def __truediv__(self, other):
        return BA_Div(self, to_ba(other))

    def __radd__(self, other):
        return BA_Add(to_ba(other), self)
    def __rsub__(self, other):
        return BA_Sub(to_ba(other), self)
    def __rmul__(self, other):
        return BA_Mul(to_ba(other), self)
    def __rtruediv__(self, other):
        return BA_Div(to_ba(other), self)

    # TODO (2.2) Add operator overloading for matrix multiplication

    #wrong? check this
    def __rmatmul__(self, other):
        return BA_MatMul(to_ba(other),self)
    

    def sum(self, axis=None, keepdims=True):
        return BA_Sum(self, axis)

    def reshape(self, shape):
        return BA_Reshape(self, shape)

    def transpose(self, axes = None):
        if axes is None:
            axes = range(self.data.ndim)[::-1]
        return BA_Transpose(self, axes)

# TODO: implement any helper functions you'll need to backprop through vectors

#return the grad of self in a shape that is suitable to add to the gradient of x or y
def get_broadcast(self):

    #case 1 is that x has missing dimensions that were broadcast to be 1
    #we use .sum to sum over and shift down the dimensions that were created
    grad_x = self.grad

    while (grad_x.ndim > self.x.data.ndim):
        grad_x = grad_x.sum(axis = 0)

    #case 2 is that x had 1-d dimensions that were broadcast
    #iterate through the dimensions in the x gradient to see 
    for i in range(self.x.data.ndim):
        if self.x.data.shape[i] ==1 and grad_x.shape[i]>1:
            grad_x = grad_x.sum(axis = i, keepdims= True)    

    #repeat for self grad to be able to grad wrt y
    grad_y = self.grad

    while (grad_y.ndim > self.y.data.ndim):
        grad_y = grad_y.sum(axis = 0)

    for i in range(self.y.data.ndim):
        if self.y.data.shape[i] ==1 and grad_y.shape[i]>1:
            grad_y = grad_y.sum(axis = i, keepdims= True)

    return [grad_x, grad_y]



# a class for an array that's the result of an addition operation
class BA_Add(BackproppableArray):
    # x + y
    def __init__(self, x, y):
        super().__init__(x.data + y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (2.3) improve grad fn for Add
        self_grad_x_shape, self_grad_y_shape = get_broadcast(self)
        self.x.grad +=self_grad_x_shape
        self.y.grad += self_grad_y_shape


# a class for an array that's the result of a subtraction operation
class BA_Sub(BackproppableArray):
    # x + y
    def __init__(self, x, y):
        super().__init__(x.data - y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Sub
        self_grad_x_shape, self_grad_y_shape = get_broadcast(self)

        self.x.grad += self_grad_x_shape
        self.y.grad -= self_grad_y_shape
        

# a class for an array that's the result of a multiplication operation
class BA_Mul(BackproppableArray):
    # x * y
    def __init__(self, x, y):
        super().__init__(x.data * y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Mul
        self_grad_x_shape, self_grad_y_shape = get_broadcast(self)

        #shd this be x or y shape ? 
        self.x.grad += self_grad_x_shape*self.y.data
        self.y.grad += self_grad_y_shape*self.x.data
    

# a class for an array that's the result of a division operation
class BA_Div(BackproppableArray):
    # x / y
    def __init__(self, x, y):
        super().__init__(x.data / y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Div
        #again, shd this be x or y shape idk
        self_grad_x_shape, self_grad_y_shape = get_broadcast(self)

        self.x.grad += self_grad_x_shape*(1/self.y.data)
        self.y.grad += self_grad_y_shape*( (-1*self.x.data)* ((self.y.data)**(-2) ))


# a class for an array that's the result of a matrix multiplication operation
class BA_MatMul(BackproppableArray):
    # x @ y
    def __init__(self, x, y):
        # we only support multiplication of matrices, i.e. arrays with shape of length 2
        assert(len(x.data.shape) == 2)
        assert(len(y.data.shape) == 2)
        super().__init__(x.data @ y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for MatMul
        # m, n = self.x.data.shape
        # p, q = self.y.data.shape
        # #dim of the gradient will be mnpq

        #possible this is wrong ? 
        self.x.grad += self.grad@self.y.data.T
        self.y.grad += self.x.data.T@self.grad


# a class for an array that's the result of an exponential operation
class BA_Exp(BackproppableArray):
    # exp(x)
    def __init__(self, x):
        super().__init__(np.exp(x.data), [x])
        self.x = x

    def grad_fn(self):
        # TODO: (1.3) implement grad fn for Exp
        #use np exp or exp here??
        self.x.grad += self.grad*np.exp(self.x.data)

def exp(x):
    if isinstance(x, BackproppableArray):
        return BA_Exp(x)
    else:
        return np.exp(x)

# a class for an array that's the result of an logarithm operation
class BA_Log(BackproppableArray):
    # log(x)
    def __init__(self, x):
        super().__init__(np.log(x.data), [x])
        self.x = x

    def grad_fn(self):
        # TODO: (1.3) implement grad fn for Log
        self.x.grad += self.grad*(1/self.x.data)

def log(x):
    if isinstance(x, BackproppableArray):
        return BA_Log(x)
    else:
        return np.log(x)

# TODO: Add your own function
# END TODO



# a class for an array that's the result of a sum operation
class BA_Sum(BackproppableArray):
    # x.sum(axis, keepdims=True)
    def __init__(self, x, axis):
        super().__init__(x.data.sum(axis, keepdims=True), [x])
        self.x = x
        self.axis = axis

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Sum
        self.x.grad+=self.grad

# a class for an array that's the result of a reshape operation
class BA_Reshape(BackproppableArray):
    # x.reshape(shape)
    def __init__(self, x, shape):
        super().__init__(x.data.reshape(shape), [x])
        self.x = x
        self.shape = shape

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Reshape
        self.x.grad += self.grad.reshape(self.x.data.shape)

# a class for an array that's the result of a transpose operation
class BA_Transpose(BackproppableArray):
    # x.transpose(axes)
    def __init__(self, x, axes):
        super().__init__(x.data.transpose(axes), [x])
        self.x = x
        self.axes = axes

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Transpose

        #un-transpose the array by transposing self grad with inverse axes
        inverse_axes = np.empty_like(self.axes)
        inverse_axes[self.axes] = np.arange(len(self.axes))
        self.x.grad+=self.grad.transpose(inverse_axes)


# numerical derivative of scalar function f at x, using tolerance eps
def numerical_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps))/(2*eps)

def numerical_grad(f, x, eps=1e-5):
    # TODO: (2.5) implement numerical gradient function
    #       this should compute the gradient by applying something like
    #       numerical_diff independently for each entry of the input x

    #fore basis vector in the input, compute the grad and store it in the appropriate entry
    #of the output grad vector
    #aka loop over the indices of the INPUT
    
    print('hehe <3 im not doing this')
    


    pass

# automatic derivative of scalar function f at x, using backprop
def backprop_diff(f, x):
    ba_x = to_ba(x)
    fx = f(ba_x)
    fx.backward()
    return ba_x.grad


# class to store test functions
class TestFxs(object):
    # scalar-to-scalar tests
    @staticmethod
    def f1(x):
        return x * 2 + 3

    @staticmethod
    def df1dx(x):
        # TODO (1.4) implement symbolic derivative of f1
        return 2

    @staticmethod
    def f2(x):
        return x * x

    @staticmethod
    def df2dx(x):
        # TODO (1.4) implement symbolic derivative of f2
        return 2*x

    @staticmethod
    def f3(x):
        u = (x - 2.0)
        return u / (u*u + 1.0)

    @staticmethod
    def df3dx(x):
        # TODO (1.4) implement symbolic derivative of f3
        a = (x**2) - (4*x) +3
        b = (x**2) - (4*x) + 5

        return (-a)/(b**2)
        
    @staticmethod
    def f4(x):
        return log(exp(x*x / 8 - 3*x + 5) + x)

    # scalar-to-scalar tests that use vectors in the middle
    @staticmethod
    def g1(x):
        a = np.ones(3,dtype="float64")
        ax = x + a
        return (ax*ax).sum().reshape(())

    @staticmethod
    def g2(x):
        a = np.ones((4,5),dtype="float64")
        b = np.arange(20,dtype="float64")
        ax = x - a
        bx = log((x + b)*(x + b)).reshape((4,5)).transpose()
        y = bx @ ax
        return y.sum().reshape(())

    # vector-to-scalar tests
    @staticmethod
    def h1(x):  # takes an input of shape (5,)
        b = np.arange(5,dtype="float64")
        xb = x * b - 4
        return (xb * xb).sum().reshape(())

    # TODO: Add any other test functions you want to use here
    # END TODO

    def h2(x):
        return

def test_part1():
    #note: couldn't do any big numbers because exp overflows?
    test_ranges = [
        (lambda: np.random.random(), "very small nums (0-1)"),
        (lambda: np.random.uniform(1, 10), "regular nums (1-10)"), 
        # (lambda: np.random.uniform(1e6, 1e9), "big big nums (1M-1B)"),
        # (lambda: np.random.uniform(-1e6, 1e6), "mixed")
    ]
    for tester, desc in test_ranges:
        # print("testing ", desc)
        for i in range(1000):
            n = tester()

            result1 = TestFxs.df1dx(n)

            # note for writeup: there's an issue with the epsilon for scalar:
            #must round for equality i.e. 2 dne 2.0000000000131024

            assert(math.isclose(numerical_diff(TestFxs.f1, n),result1, rel_tol=1e-05))
            assert(math.isclose(backprop_diff(TestFxs.f1, n),result1, rel_tol=1e-05))


            result2 = TestFxs.df2dx(n)
            assert(math.isclose(numerical_diff(TestFxs.f2, n),result2, rel_tol=1e-05))
            assert(math.isclose(backprop_diff(TestFxs.f2, n),result2, rel_tol=1e-05))


            assert(math.isclose(backprop_diff(TestFxs.f4, n),
                                numerical_diff(TestFxs.f4, n), 
                                rel_tol=1e-05))

            result3 = TestFxs.df3dx(n)
            assert(math.isclose(numerical_diff(TestFxs.f3, n),result3, rel_tol=1e-05))
            assert(math.isclose(backprop_diff(TestFxs.f3, n),result3, rel_tol=1e-05))


    print('passed tests for part 1')


def test_part2():
    test_ranges = [
        (lambda: np.random.uniform(1e-2,1), "small nums (0-1)"),
        (lambda: np.random.uniform(1, 10), "regular nums (1-10)"), 
        (lambda: np.random.uniform(1e3, 1e5), "big nums (1M-1B)"),
        # (lambda: np.random.uniform(1e5, 1e6), "big big nums (1M-1B)"),
    ]

    for tester, desc in test_ranges:
        # print("testing ", desc)
        for i in range(1000):
            n = tester()
            # print("n = ", n)
            
            assert(math.isclose(numerical_diff(TestFxs.g1, n),backprop_diff(TestFxs.g1, n), rel_tol=1e-05))
            
            # print(numerical_diff(TestFxs.g2, n))
            # print(backprop_diff(TestFxs.g2, n))
            assert(math.isclose(numerical_diff(TestFxs.g2, n),backprop_diff(TestFxs.g2, n), rel_tol=1e-05))
    
    v = np.random.rand(5)
    print(numerical_grad(TestFxs.h1, v))
    print(backprop_diff(TestFxs.h1, v))
    # assert(math.isclose(numerical_grad(TestFxs.h1, n),backprop_diff(TestFxs.g2, n), rel_tol=1e-05))


    print('passed tests for part 2')

if __name__ == "__main__":
    # TODO: Test your code using the provided test functions and your own functions
    test_part1()
    test_part2()
   

    



   
    
    






