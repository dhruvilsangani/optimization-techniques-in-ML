import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# dataset = pd.read_csv("dataset.csv", index_col = 0)
# x = np.array(dataset['x'])
# y = np.array(dataset['y'])
def random_data():
    x = np.random.uniform(0,1,100)
    y = x+np.random.randn(1)
    return (x, y)

x,y = random_data()

def costFunction(m, t0, t1, x, y):
	return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])

def gen_mini_batches(x, y, batch_size): 
	mini_batches = [] 
	data = np.hstack((x, y)) 
	# print(x)
	# print(y)
	#print(data)
	np.random.shuffle(data) 
	#print(data)
	n_minibatches = data.shape[0]
	i = 0
  
	for i in range(n_minibatches + 1):
		mini_batch = data[i * batch_size:(i + 1)*batch_size :] 
		x_mini = mini_batch[: :-1] 
		Y_mini = mini_batch[: -1].reshape((-1, 1)) 
		mini_batches.append((x_mini, Y_mini)) 
	if data.shape[0] % batch_size != 0: 
		mini_batch = data[i * batch_size:data.shape[0]] 
		x_mini = mini_batch[: :-1] 
		Y_mini = mini_batch[: -1].reshape((-1, 1)) 
		mini_batches.append((x_mini, Y_mini)) 
	#print(f"Mini --------- {mini_batches}")
	#print("-----------------------------------------------------------------------------------")
	return mini_batches 


def mini_batch_gd(learning_rate, x, y, batch_size,iterations):
	theeta0 = 0
	theeta1 = 0

	m = x.shape[0]

	#total error
	J = costFunction(m, theeta0, theeta1, x, y)
	count = [i+1 for i in range(10000)]

	lossHistory = []
	theeta0=0
	theeat1=0

	for i in range(iterations):
		mini_batches = gen_mini_batches(x, y, batch_size) 
		for mini_batch in mini_batches: 
			grad0 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
			grad1 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])
			theeta0 = theeta0 - learning_rate * grad0
			theeta1 = theeta1 - learning_rate * grad1
			lossHistory.append(costFunction(m, theeta0, theeta1, x, y))
		# print('theta0 = ' + str(theeta0)+"    ",end="")
		# print('theta1 = ' + str(theeta1))

	return lossHistory, theeta0, theeta1

alpha = 0.01
num_iteration = 100
batch_size=40

lossHistory, theta0, theta1 = mini_batch_gd(alpha, x, y,batch_size,num_iteration)

print('Final  theta0 = ' + str(theta0)+"    ",end="")
print('Final  theta1 = ' + str(theta1))

plt.figure(0)
plt.scatter(x, y, c = 'red')
plt.plot(x, theta0 + theta1 * x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()