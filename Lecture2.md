### Lecture2 P19 Example:State-Value Function for Student MRP (2) gamma=0.9


```python
%%html
<img src='MRP.jpg'  width=600, height=400>
```


<img src='https://raw.githubusercontent.com/GarfieldF/RL-Course-by-David-Silver/master/MRP.jpg'  width=600, height=400>



### 1. direct solution by matrix calculation


```python
import numpy as np
P = np.matrix([
    [0, 0.5, 0, 0, 0, 0.5,0],
    [0, 0, 0.8, 0, 0, 0,0.2],
    [0, 0, 0, 0.6, 0.4, 0,0],
    [0, 0, 0, 0, 0, 0,1],
    [0.2, 0.4, 0.4, 0, 0, 0,0],
    [0.1, 0, 0, 0, 0, 0.9,0],
    [0,0,0,0,0,0,1]
], dtype=np.float32)
R = np.matrix([-2, -2, -2, 10, 1, -1,0], dtype=np.float32).T  
I = np.eye(7)
```


```python
def cal_v(gamma):
    inv = (I - gamma * P).I
    v =  inv.dot(R)
    return v
```


```python
cal_v(0.9)
```




    matrix([[-5.01272827],
            [ 0.9426556 ],
            [ 4.0870216 ],
            [10.        ],
            [ 1.90839272],
            [-7.6376073 ],
            [ 0.        ]])



    matrix([[-5.01272827],
            [ 0.9426556 ],
            [ 4.0870216 ],
            [10.        ],
            [ 1.90839272],
            [-7.6376073 ],
            [ 0.        ]])
This result is roughly equal to the result shown in the picture \[-5.0,0.9,4.1,10,1.9,-7.6,0\] 

### 2. The iterative method 


```python
def iteration(gamma,V):
    return R+gamma*np.dot(P,V)
V=np.zeros((7,1))#V=np.ones((7,1)) both initial vector is right
for i in range(50):
    V=iteration(0.9,V)
print(V)
```

    [[-5.01073231]
     [ 0.9428716 ]
     [ 4.08727958]
     [10.        ]
     [ 1.90900894]
     [-7.63400914]
     [ 0.        ]]
    

the result of iteration is :

    [[-5.01073231]
    [ 0.9428716 ]
    [ 4.08727958]
    [10.        ]
    [ 1.90900894]
    [-7.63400914]
    [ 0.        ]]


### Lecture2 P19 Example: State-Value Function for Student MDP


```python
%%html
<img src='MDP.jpg'  width=600, height=400 >
```


<img src='https://raw.githubusercontent.com/GarfieldF/RL-Course-by-David-Silver/master/MDP.jpg'  width=600, height=400 >



### 1. direct solution by matrix calculation


```python
P_ssa=np.ones((5,2,5), dtype=np.float32)
Pi=np.ones((5,2), dtype=np.float32)*0.5#the shape of matrix Pi is supposed to be (5，1，2) in theory
P_ssa[0]=[[1,0,0,0,0],[0,1,0,0,0]]
P_ssa[1]=[[1,0,0,0,0],[0,0,1,0,0]]
P_ssa[2]=[[0,0,0,1,0],[0,0,0,0,1]]
P_ssa[3]=[[0,0.2,0.4,0.4,0],[0,0,0,0,1]]
P_ssa[4]=[[0,0,0,0,1],[0,0,0,0,1]]
```


```python
R_sa = np.matrix([[-1,0],[-1,-2],[-2,0],[1,10],[0,0]], dtype=np.float32)#the shape of matrix R_sa is supposed to be (5，2，1) in theory
I2=np.eye(5)
```


```python
P_ss_pi=np.zeros((5,5))
for i in range(len(P_ssa)):
    P_ss_pi[i] = np.matmul(Pi[i],P_ssa[i])
print(P_ss_pi)
```

    [[0.5 0.5 0.  0.  0. ]
     [0.5 0.  0.5 0.  0. ]
     [0.  0.  0.  0.5 0.5]
     [0.  0.1 0.2 0.2 0.5]
     [0.  0.  0.  0.  1. ]]
    


```python
R_s_pi=np.zeros((5,1))
for i in range(len(Pi)):
    R_s_pi[i] = np.matmul(Pi[i],R_sa[i].T)
print(R_s_pi)
```

    [[-0.5]
     [-1.5]
     [-1. ]
     [ 5.5]
     [ 0. ]]
    


```python
def cal_v_pi(gamma):
    inv = (I2 - gamma * np.asmatrix(P_ss_pi)).I
    v =  inv.dot(R_s_pi)
    return v
```


```python
cal_v_pi(0.9999)#gamma=1 singular matrix can not calc its inverse matrix
```




    matrix([[-2.30758211],
            [-1.30794366],
            [ 2.6917332 ],
            [ 7.38420483],
            [ 0.        ]])




    matrix([[-2.30758211],
            [-1.30794366],
            [ 2.6917332 ],
            [ 7.38420483],
            [ 0.        ]])

This result is roughly equal to the result shown in the picture \[-2.3,-1.3,2.7,7.4,0\]

### 2. The iterative method 


```python
def iteration_MDP(gamma,V):
    return R_s_pi+gamma*np.dot(P_ss_pi,V2)
V2=np.zeros((5,1))#np.ones((5,1)) can not get the right answer
for i in range(50):
    V2=iteration_MDP(1.0,V2)
print(V2)
```

    [[-2.30774159]
     [-1.30772445]
     [ 2.69230386]
     [ 7.38460906]
     [ 0.        ]]
    

the result of iteration is :

    [[-2.30774159]
     [-1.30772445]
     [ 2.69230386]
     [ 7.38460906]
     [ 0.        ]]



```python

```
