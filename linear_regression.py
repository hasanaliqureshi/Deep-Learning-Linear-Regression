from numpy import *

def compute_error_for_line_given_point(b,m,points):
	#Here we are computing the error which is the difference in distance of values of y coordinate of our data set and the random line we create
	totalError = 0
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (y-(m*x +b)) **2 #Equation 1
	#We have summed all the distances and squared it and returning the average
	return totalError/ float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
	#This is where the MAGIC! (gradient descent) happens :P 
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		#computing partial derivative of error function
		#partial derivative gives us the tangent line from which we update values of b and m
		b_gradient += -(2/N) * (y- ((m_current * x) + b_current)) #Equation 2
		m_gradient += -(2/N) * x * (y- ((m_current * x) + b_current)) #Equation 3
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b,new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	#Here we are performing Gradient Descent to reach to the optimal values of b and m for the best fitting line
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
	#Collecting our data from dataset data.csv which is test scores of students and amount of hours they studied
	points = genfromtxt("data.csv", delimiter=',')
	#Defining our hyperparameters
	learning_rate = 0.0001 #Learning Rate : How fast we should learn
	#Initia b (y-intercept) and m (slope) value
	initial_b = 0
	initial_m = 0
	#Number of iterations (1000 is enough because database is small)
	num_iterations = 1000
	#Displaying Results
	print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_point(initial_b, initial_m, points)))
	print ("Running ...")
	[b,m] = gradient_descent_runner(points,initial_b, initial_m, learning_rate, num_iterations)
	print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_point(b,m,points)))


if __name__ == '__main__':
	run()