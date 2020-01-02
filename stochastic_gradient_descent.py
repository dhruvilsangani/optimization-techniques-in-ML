import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"dataset.csv",index_col=0)
x=np.array(dataset['x'])
y=np.array(dataset['y'])

def cost(m, t0, t1, x, y):
    return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])

t0 = 0
t1 = 1
iterations = 50000
lossHistory = np.empty(iterations)

count = [i for i in range(1,iterations+1)]  #x-axis for plotting lossHistory
lr = 0.001
m = x.shape[0]

for iteration in range(0,iterations):
    it = np.random.randint(0,100)
    grad0 = ((t0 + t1*x[it] - y[it]))
    grad1 = ((t0 + t1*x[it] - y[it])*x[it])
    t0 = t0 - lr*grad0
    t1 = t1 - lr*grad1
    lossHistory[iteration] = cost(m, t0, t1, x, y)

print("Theta 0 : ",t0,"Theta1 : ",t1)

plt.figure(0)
plt.scatter(x,y, c='red')
plt.plot(x,t0 + t1*x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()
plt.figure(1)
plt.plot(count,lossHistory)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss.png')
plt.show()