
import numpy as np
import time

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
        dependencies = []
        visited = set()
        queue = []

        queue.append(self)
        visited.add(id(self))

        while queue:
            curr_array = queue.pop(0)
            dependencies.append(curr_array)

            for dependency in curr_array.dependencies:
                if id(dependency) not in visited:
                    queue.append(dependency)
                    visited.add(id(dependency))
    
        return dependencies


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
        
        # (1) reverse sorting
        sorted_dependencies = sorted(all_my_dependencies, key=lambda x: x.order, reverse=True)
        
        # (2) zero out grad
        for array in all_my_dependencies:
            array.grad = np.zeros_like(array.data)
        
        # (3) base differentiation
        self.grad = np.ones_like(self.data)
        
        # (4) compute grad
        for array in sorted_dependencies:
            array.grad_fn()

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
    def __matmul__(self, other):
        return BA_MatMul(self, to_ba(other))

    def __rmatmul__(self, other):
        return BA_MatMul(to_ba(other), self)

    def sum(self, axis=None, keepdims=True):
        return BA_Sum(self, axis)

    def reshape(self, shape):
        return BA_Reshape(self, shape)

    def transpose(self, axes = None):
        if axes is None:
            axes = range(self.data.ndim)[::-1]
        return BA_Transpose(self, axes)

# TODO: implement any helper functions you'll need to backprop through vectors
def unbroadcast(grad, target_shape):
    """
    Reduces grad to target_shape by summing over broadcasted dimensions.
    This handles the reverse of NumPy broadcasting.
    """
    # collapse extra dimensions from broadcasting by summing
    ndims_added = len(grad.shape) - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    
    # reduce enlarged dimensions from broadcasting by summing
    for i in range(len(target_shape)):
        if target_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad

# a class for an array that's the result of an addition operation
class BA_Add(BackproppableArray):
    # x + y
    def __init__(self, x, y):
        super().__init__(x.data + y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (2.3) improve grad fn for Add
        # self.x.grad += self.grad
        # self.y.grad += self.grad

        self.x.grad += unbroadcast(self.grad, self.x.data.shape)
        self.y.grad += unbroadcast(self.grad, self.y.data.shape)


# a class for an array that's the result of a subtraction operation
class BA_Sub(BackproppableArray):
    # x - y
    def __init__(self, x, y):
        super().__init__(x.data - y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Sub
        # z = x - y, dz/dx & dz/dy
        # self.x.grad += self.grad
        # self.y.grad += -self.grad

        self.x.grad += unbroadcast(self.grad, self.x.data.shape)
        self.y.grad += unbroadcast(-self.grad, self.y.data.shape)

# a class for an array that's the result of a multiplication operation
class BA_Mul(BackproppableArray):
    # x * y
    def __init__(self, x, y):
        super().__init__(x.data * y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Mul
        # self.x.grad += self.grad * self.y.data
        # self.y.grad += self.grad * self.x.data

        grad_x = self.grad * self.y.data
        grad_y = self.grad * self.x.data
        self.x.grad += unbroadcast(grad_x, self.x.data.shape)
        self.y.grad += unbroadcast(grad_y, self.y.data.shape)

# a class for an array that's the result of a division operation
class BA_Div(BackproppableArray):
    # x / y
    def __init__(self, x, y):
        super().__init__(x.data / y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Div
        # z = x / y
        # self.x.grad += self.grad / self.y.data
        # self.y.grad += self.grad * (-self.x.data / (self.y.data ** 2))

        grad_x = self.grad / self.y.data
        grad_y = self.grad * (-self.x.data / (self.y.data ** 2))
        self.x.grad += unbroadcast(grad_x, self.x.data.shape)
        self.y.grad += unbroadcast(grad_y, self.y.data.shape)


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
        # z = x @ y, dz/dx = x' · y^T
        self.x.grad += self.grad @ self.y.data.T
        self.y.grad += self.x.data.T @ self.grad 


# a class for an array that's the result of an exponential operation
class BA_Exp(BackproppableArray):
    # exp(x)
    def __init__(self, x):
        super().__init__(np.exp(x.data), [x])
        self.x = x

    def grad_fn(self):
        # TODO: (1.3) implement grad fn for Exp
        self.x.grad += self.grad * self.data

def exp(x):
    if isinstance(x, BackproppableArray):
        return BA_Exp(x)
    else:
        return np.exp(x)

# a class for an array that's the result of an logarithm operation
class BA_Log(BackproppableArray):
    # log(x)
    def __init__(self, x):
        # super().__init__(np.log(x.data), [x])
        super().__init__(np.log(x.data + 1e-12), [x])
        self.x = x

    def grad_fn(self):
        # TODO: (1.3) implement grad fn for Log
        # self.x.grad += self.grad / self.x.data
        self.x.grad += self.grad / (self.x.data)

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
        # dz/dx = 1, so copy dz across the summed entries
        self.x.grad += np.broadcast_to(self.grad, self.x.data.shape)

# a class for an array that's the result of a reshape operation
class BA_Reshape(BackproppableArray):
    # x.reshape(shape)
    def __init__(self, x, shape):
        super().__init__(x.data.reshape(shape), [x])
        self.x = x
        self.shape = shape

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Reshape
        # reshape back
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
        # transpose to inverse axes
        self.x.grad += self.grad.transpose(np.argsort(self.axes))


# numerical derivative of scalar function f at x, using tolerance eps
def numerical_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps))/(2*eps)

def numerical_grad(f, x, eps=1e-5):
    # TODO: (2.5) implement numerical gradient function
    #       this should compute the gradient by applying something like
    #       numerical_diff independently for each entry of the input x

    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        # Create perturbation vector (e_i scaled by eps)
        perturbation = np.zeros_like(x)
        perturbation[i] = eps
        grad[i] = (f(x + perturbation) - f(x - perturbation)) / (2 * eps)
    
    return grad

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
        return 2 * x

    @staticmethod
    def f3(x):
        u = (x - 2.0)
        return u / (u*u + 1.0)

    @staticmethod
    def df3dx(x):
        # TODO (1.4) implement symbolic derivative of f3
        # = ((u² + 1)·1 - u·2u) / (u² + 1)²
        # = (u² + 1 - 2u²) / (u² + 1)²
        # = (1 - u²) / (u² + 1)²
        u = x - 2.0
        return (1 - u * u) / (u * u + 1) ** 2

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

    @staticmethod
    def h2(x):  
        m = x.reshape((40, 25))
        r = m @ m.transpose()
        return (r * r + 16.6).sum().reshape(())

    @staticmethod  
    def h3(x):  
        a = x.reshape((30, 50))
        b = np.arange(50, dtype='float64')
        b = b.reshape(50,1)
        mult = a@b
        r = log(mult+1.0)
        r = r.sum().reshape(())
        return r
    
    @staticmethod
    def h4(x): # takes (d, ) where d can be 1000
        # f(x) = sum(x_i^3 * exp(-0.01 * x_i^2)) + ||x||^2 / 1000
        x_squared = x * x        
        term1 = (x_squared * x * exp(x_squared * (-0.01))).sum()
        term2 = x_squared.sum() / 1000.0
        return term1 + term2
    

def random_test(function, dimension):
    print(f"\nTesting {function}")
    print("=" * 60)
    
    np.random.seed(42)
    num_tests = 100
    vector_dim = dimension
    
    test_cases = {
        "Negative [-100, 0]": [],
        "Small [0, 0.001]": [],
        "Large [1000, 100000]": []
    }
    
    # Generate random vectors
    for i in range(num_tests):
        negative_vec = np.random.uniform(-100, 0, vector_dim)
        test_cases["Negative [-100, 0]"].append(negative_vec)
        
        small_vec = np.random.uniform(0, 0.001, vector_dim)
        test_cases["Small [0, 0.001]"].append(small_vec)
        
        large_vec = np.random.uniform(1000, 100000, vector_dim)
        test_cases["Large [1000, 100000]"].append(large_vec)
    
    # convert to numpy for numerical gradient
    def f_numpy(x):
        ba_x = to_ba(x)
        result = function(ba_x)
        return result.data.item()
    
    # Run Test
    for case_name, vectors in test_cases.items():
        # print(f"\nTesting {case_name}:")
        # print("-" * 50)
        
        mean_errors = []
        max_errors = []
        numerical_times = []
        backprop_times = []
        
        for i, x_test in enumerate(vectors):
            # print(f"Processing test {i+1}/{num_tests}...", end='\r')

            # numerical
            start_time = time.time()
            numerical_grad_result = numerical_grad(f_numpy, x_test)
            numerical_time = time.time() - start_time
            numerical_times.append(numerical_time)
            
            # backprop
            start_time = time.time()
            backprop_grad_result = backprop_diff(function, x_test)
            backprop_time = time.time() - start_time
            backprop_times.append(backprop_time)
            
            # compute error
            error_diff = numerical_grad_result - backprop_grad_result
            mean_error = np.mean(error_diff)
            max_error = np.max(np.abs(error_diff))
            mean_errors.append(mean_error)
            max_errors.append(max_error)
                
        mean_errors = np.array(mean_errors)
        max_errors = np.array(max_errors)
        numerical_times = np.array(numerical_times)
        backprop_times = np.array(backprop_times)
        
        avg_mean_error = np.mean(mean_errors)
        avg_max_error = np.mean(max_errors)
        avg_numerical_time = np.mean(numerical_times)
        avg_backprop_time = np.mean(backprop_times)
        
        # print(f"Valid test cases: {len(mean_errors)}/{num_tests}")
        # print(f"Average Mean Error: {avg_mean_error:.8e}")
        # print(f"Average Max Error:  {avg_max_error:.8e}")
        # print(f"Average Numerical Time: {avg_numerical_time:.4f} seconds")
        # print(f"Average Backprop Time:  {avg_backprop_time:.4f} seconds")
        # print(f"Average Speedup: {avg_numerical_time/avg_backprop_time:.2f}x")
        # print("\n" + "=" * 70)

        print(f"Testing for {function} complete!")
        return {
            "valid_test_cases": f"{len(mean_errors)}/{num_tests}",
            "average_mean_error": f"{avg_mean_error:.8e}",
            "average_max_error": f"{avg_max_error:.8e}",
            "average_numerical_time": f"{avg_numerical_time:.4f} seconds",
            "average_backprop_time": f"{avg_backprop_time:.4f} seconds",
            "average_speedup": f"{avg_numerical_time/avg_backprop_time:.2f}x"
        }
        
if __name__ == "__main__":
    # TODO: Test your code using the provided test functions and your own functions

    results = []
    results.append(random_test(TestFxs.g1, 1))
    results.append(random_test(TestFxs.g2, 1))
    results.append(random_test(TestFxs.h1, 5))
    results.append(random_test(TestFxs.h2, 1000))
    results.append(random_test(TestFxs.h3, 1000))

    num_results = len(results)
    avg_mean_error = sum(float(r["average_mean_error"]) for r in results) / num_results
    avg_max_error = sum(float(r["average_max_error"]) for r in results) / num_results
    avg_numerical_time = sum(float(r["average_numerical_time"].split()[0]) for r in results) / num_results
    avg_backprop_time = sum(float(r["average_backprop_time"].split()[0]) for r in results) / num_results
    avg_speedup = sum(float(r["average_speedup"].rstrip('x')) for r in results) / num_results
    
    valid_totals = [r["valid_test_cases"].split('/') for r in results]
    total_valid = sum(int(vt[0]) for vt in valid_totals)
    total_tests = sum(int(vt[1]) for vt in valid_totals)
    
    print("=" * 50)
    print("AVG RESULTS ACROSS ALL TESTS:")
    print("=" * 50)
    print(f"Total Valid test cases: {total_valid}/{total_tests}")
    print(f"Average Mean Error: {avg_mean_error:.8e}")
    print(f"Average Max Error:  {avg_max_error:.8e}")
    print(f"Average Numerical Time: {avg_numerical_time:.4f} seconds")
    print(f"Average Backprop Time:  {avg_backprop_time:.4f} seconds")
    print(f"Average Speedup: {avg_speedup:.2f}x")
