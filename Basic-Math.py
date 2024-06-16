"""
with masters theorem for solving the recurrance relations:
func should be in the form of aT(n/b)+O(n^c)

case 1: O(n^loga base b) if c<log a base b

case 2: O(n^c log n) if c = log a base b

case 3: O(n^c) if c>log a base b

where a>=1, b>1, c>=0
 count the number of digits in python 
 time complexity == theta(d times) where d is number of the digits

x = 123
count=0

while x!=0:
        x = x//10
        count = count+1
    
print(count)

palin

x = 112111
xx=x
count = 0
while xx!=0:
    
    count=count*10+xx%10
    xx=xx//10
    print(count)

if x ==count: 
    print(x ==count)
else:
    print(x)
    print(count)
    print(x==count)

# time complexity = theta(D) where D is no of digits

factorial

n = 5
ans=1
while n!=0:
    ans = ans*n
    n=n-1
print(ans)


# using the recurssive approach 
def fac(n):
    if(n==0):
        return 1
    else:
        return n*fac(n-1)

print(fac(4))

Tc of rec sol = t(n-1)+theta(1) ==> complexity ==> n where the auxilary space is theta(n) as in every call stack it takes some space.



time complexity = Theta(n) where n is the size of the array and the auxilary space complexity is theta(1) because we are not using any additional data structure for storing

# No of trailing zero's:

n=25
ans =1 
anss=0
count=0
while n!=0:
    ans = ans*n
    n=n-1
print("ans:",ans)

while(ans%10==0):
    count = count+1
    ans=ans//10 
    # // operator return the exact integer
print("count",count)


# optimizing the code 
n=25
while n>=5:
    anss = anss+n//5
    n=n//5 
print("optimized count:", anss)



#another optimized code similar to for in java for(int i=5 ; i<=n; i=i*5)

i=5
n=25
count=0
while i<=n:
    count=count+n//i
    i=i*5
print("max optimized code count:",count)



"""
# gcd or hcf 
import math

x=9
y=24

d = min(x,y)

# or logical or and | shows bitwise or

while x%d != 0 or y%d !=0 :
    d=d-1

print('gcd',d)

# The optimized code for gcd is using the normal eucalidiean method

x=9
y=24

while x!=y:
    if x>y:
        x=x-y
    else:
        y=y-x

# when both the num becomes equal we can return/print any of it
print('using the eucalidiean algo gcd',x)

# using the optimized eucalidiean algo

def euc(x,y):
    
    
    # base case for the recurssion
    if(y==0):
        return x
    
    else:
        return euc(y,x%y)
print('the optimized eucalidien gcf code hcf',euc(9,24))







#LCM

x=4 
y=6

d = max(x,y)

while d%x!=0 or d%y!=0:

    d= d+1
print('lcm',d)

# efficient sol of gcd = multiplication of 2 num x,y / gcd of 2 num



def eucalgo(x,y):


    if(y==0):
        return x
    else:
        return eucalgo(y,x%y)
    
gcd = eucalgo(4,6)

lcm = 4*6//gcd

print("gcd of 4,6 is ",gcd,"and lcm is",lcm)


# prime number note:1 is neither prime nor composite
# naive solution
def isprime(x):

    for i in range(2,x):
        if(x%i == 0):
            return False
        else:
            return True
        
print(isprime(17))

# the time complexity will be O(n) 

# optimizing code checking till sqrt will be enough

def isprimeopt(x):

    for i in range(2,math.isqrt(x)):
        if(x%i == 0):
            return False
        else:
            return True
        
print(isprimeopt(17))

# here the time complexity is O(sqrt(n))

# optimizing more by checking the base conditons like n%2 so no need to check for 4 also and same with n%3==0 return false and if n==2 and n==3 is a prime return true


def moreopt(x):
    if(x==1):
        return False
    
    if(x==2 or x==3):
        return True
    
    if(x%2==0 or x%3==0):
        return False
    
    for i in range(5,math.isqrt(x)+1,6):
        if(x%i == 0 or x%(i+2) == 0):
            return False
    else:    
        return True
        
        # checks the case 1,2,3 initially , next 5,7 & 11,13 and so  no need checking the num in between like 4,6,8,9,10 as we are checking the 2 and 3 


print("the max optimized solution for prime",moreopt(11))

# // prime factors

def isprimee(x):
    if x == 1:
        return False
    if x==2 or x==3:
        return True
    
    if x%2==0 or x%3==0:
        return False
    
    for i in range(5,math.isqrt(x),6):

        if(x%i==0 or x%i+2==0):
            return False
    return True
    
# print("checkprimeforprimefactors",isprimee(12)), x=100 50 25 5 1 , 2*2*5*5

def primefactor(x):

    for i in range(1,x+1):
        while isprimee(i) and x%i==0:
            print(i," ")
            x=x//i
        if x==1:
            break

primefactor(23)
# checking for all divisors of n 

def divisorsnaive(x):

    for i in range (2,x+1):
        if(x%i==0):
            print("the div of x are", i, " ")
divisorsnaive(100)

def divisorsopt(x):
    i=1

    while(i<=math.isqrt(x)):
        if(x%i==0):
            print(i)
            if(i != x/i):
                print(x//i)
        i=i+1

divisorsopt(25)

def divisorsmoreopt(x):
    # 25 --> till sqrt -> 5.. 1,5 and backward dividing i with n 
    i=1
    while(i<math.isqrt(x)):
        if(x%i==0):
            print(i)
        i=i+1
    while(i>=1):
        if(x%i==0):
            print(x//i)
        i=i-1
divisorsmoreopt(25)

# Sieve of eratosthenes .. which is basically calculating the prime numbers till that number
# naive
def sieve(x):
    for i in range(2,x+1):
        if(isprimee(i)):
            print("sieve",i)

sieve(23)

# other version of sieve, isprime[]


def sieveopt(x):
    if(x)<=1:
        return 
    
    isprime = [True]*(x+1)

    i=2

    while i<=math.isqrt(x):
        if isprime[i]:
            for j in range(2*i,x+1,i):
                isprime[j]=False
        i=i+1
    
    for i in range(2,x+1):
        if isprime[i]:
            print("sieveopt",i)
sieveopt(23)

# computing power 2 5 


def pow(x,y):
    value=1
    while(y>=1):

        value=value*x
        y=y-1
    return value

print("calculating power using naive approach ",pow(2,5))

""" using the recurssive solution 
2,5 -->temp*x
2*2*2*2
2,2 --> temp
2*2
2,1 --> temp*x
1*1
2,0 --> return 1 
"""

def optpow(x,y):
    # handling the base case 
    if(y==0):
        return 1
    temp=optpow(x,y//2)
    temp=temp*temp

    if(y%2==0):
        return temp
    if(y%2!=0):
        return temp*x
    
print("recursive optimized approach",optpow(2,5))

"""

PART - 2 , BIT MAGIC 


using the iterative approach
2,5 
res=1
n=5
5%2!=0 --> res*x ==>2

x=x*x =>4
n=n//2

x=2, n=2
x=x*x ==>16

x=2, n=1 --> res*x => 16*2 

5 -> (2*2)*(2*2)*2

2 ->(2*2)*(2*2)

1 ->2*2

0 -> 1

4
2*2
2
2*2*2*2
1
1*(2*2*2*2)

def iterativebinary(x,n):

    res=1
    while(n>0):
        if n%2!=0:
            res=res*x
        x=x*x
        n=n//2
    return res

print("using the iterative approach the power is",iterativebinary(2,5))
"""

# binary representation of negative numbers is always stored in 2's complement form where it's ranging from [-2^(n-1) to 2^(n-1)-1]
# only negative numbers are represented in 2's complement form in most of the languages and positive numbers are represented in only decimal to binary form
# steps to get 2's complement: ex:0011
# 1.invert all bits 1100
# 2.add one to it   1101

# other options:
# we will represent 1st bit from the left side as signed bit , 100.......011 (-3 example for 32 bit)
# why only 2's complement:
# 1.in 2's complement there is only form of zero representation (000)
# 2.arthemetic operations are very easy to perform they are derived from the idea 0-x
# 3.the leading bit is always 1 for the negative numbers

# decimal number has 10 digits 0 to 9 and the binary system have 2 bits 0 and 1
# octal will only 8 digits 0 to 7
#hexa-decimal will have 16 digits 0-9,A,B,C,D,E,F..10-19,1A,1B,1C,1D,1E,1F,20-29,2A-2F.....
'''
BITWISE OPERATORS:

1. AND OPERATOR(&) -> THE 2 BITS SHOULD BE ONE SO IT WILL ONE

I/P1  I/P2  O/P
  0     0    0
  0     1    0
  1     0    0
  1     1    1

2. OR OPERATOR(|) -> IF ANY OF THE BIT IS ONE THEN THE O/P WILL BE 1

 I/P1 I/P2 O/P
  0    0    0
  0    1    1
  1    0    1
  1    1    1

3. XOR OPERATOR(^) -> WHEN 2 BITS ARE SAME O/P WILL BE ZERO AND IF THEY ARE DIFFERENT O/P WILL BE 1 , FOR POWER (**) , FOR XOR(^) 

  I/P1  I/P2 O/P
   0     0    0
   1     0    1
   0     1    1 
   1     1    0

# BITWISE OPERATORS PART - 2

# 4. LEFT SHIFT OPERATOR(X<<N) -> Left shift operator shifts the elements towards left side with placing all the zeros on the right

for example x = 5 , 101

x<<1 -> 1010 -->x*2^1 --> 5*2 -> 10 
x<<2 -> 10100 --> x*2^2 --> 5*4 ->20
x<<3 -> 101000 --> x*2^3 --> 5*8 -> 40

by doing the left shift the value will be increased 

# 5 . RIGHT SHIFT OPERATOR (X>>N) -> right shift operator shifts the elements towards right by removing the bits

for example x = 5 , 101 , right shift will be dividing with the power of 2

x>>1 -> 010 -> 5/2^1 ->5//2 ->2
x>>2 -> 001 -> 5/2^2 -> 5//4 -> 1
x>>3 -> 000 -> 5/2^3 ->5//8 -> 0

note : if leading bit is 1 its a negative number, if its zero its a positive number.


# 6. BITWISE NOT(~) -> 
Note : negative numbers are not stored directly as thier starting digits will be 1 it will be huge so they are stored in 2's complement form
reason is because as we discussed to avoid the +ve zero's and -ve zero's.

since if the leading bit is 1 it is treated as the -ve number

ex: x=5 , ~5 => 000...0101 ->> 111..1010 (-6)

6: 00000...0110
1's complement: 11111..1001
2's complement:1's complement+1: 00000..1010

Checking whether the kth bit is set or not:

it can be done using 2 methods: left shift operator or right shift operator:

ex: x = 101 checking whether k=3 is set bit

sol: 101 & with 1<<(k-1)

x:       00101
1<<(k-1):00100
&       :00100 (it will be some value here 4) there fore k bit is set


similarly with the right shift operator:

sol: x>>k-1 & 1 
x=5 , k =3 
x:     00101
x>>k-1:00001
&x    :00001
out   : 1 (if the output is equal to 1 then k bit is set bit)



'''
# checking whether the kth bit is set or not 
def issetbit_lftshft(x,k):

    if(x&(1<<(k-1))):
        print('true kth bit is set')
    else:
        print('false kth bit is not set')

issetbit_lftshft(5,3)

def issetbit_rghtshft(x,k):
    if((x>>(k-1))&1):
        print('true kth bit is set')
    else:
        print('false kth bit is not set')

issetbit_rghtshft(5,3)

# checking the number of set bits in a given number

def naive_check_setbits(n):
    sum=0
    while n:
        if n%2==1:
            sum+=1
        n=n//2
    return sum

print(naive_check_setbits(7))

# optimizing the above code with the bitwise operator

def naive_check_setbits_opt(n):
    sum=0
    while n:
        if n&1==1:
            sum+=1
        n=n//2
    return sum

print(naive_check_setbits_opt(7))
# here in the above codes the TC is O(no of bits)

def check_setbits_brainalgo(n):
    count=0
    while n:
        n=n&(n-1)
        count+=1
    return count

print(check_setbits_brainalgo(7))

#this is more optimized code the time complexity here is 0(number of set bits)

# we can optimized this further more using the preprocessing that is using the look-up table
# for 32 bits case 
class check_set_bits_lookuptable:
    tble=[0] * 256
    def initialize():
        
        for i in range(256):
            check_set_bits_lookuptable.tble[i] = (i&1)+ check_set_bits_lookuptable.tble[i//2]
            # print(i,check_set_bits_lookuptable.tble[i])
    def count_bits(n):
        check_set_bits_lookuptable.initialize()
        return check_set_bits_lookuptable.tble[n&0xff]+check_set_bits_lookuptable.tble[(n>>8)&0xff]+check_set_bits_lookuptable.tble[(n>>16)&0xff]+check_set_bits_lookuptable.tble[(n>>24)&0xff]

# similarly for 64 bits we do it 8 times till 64 bits 



print(check_set_bits_lookuptable.count_bits(2000))

# note:break statement gets you out of the loop but the return statement get you out of the function 

# finding the odd occuring element from the list 

lst = [10,10,20,20,26]
# note: X^0 = X (XOR Operation: if different elements then only o/p = 1)
# using naive for only one time odd

def one_odd_occuring(lst):
    for i in lst:
        count=0
        for j in lst:
            if i == j:
                count+=1
        if(count%2!=0):
            return i
print('using the naive appoach',one_odd_occuring(lst))
# Time complexity : O(n^2) for the naive approach

# using bits 
def find_odd_occuring(lst):
    out=0
    for i in lst:
        out = i^out
    return out
print('the odd occuring element from the lst',find_odd_occuring(lst))
# Time complexity is 0(n) using the bits 

# naive approach
def ispow2(n):
    if n==0:
        return False
    while n!=1:
        if n%2!=0:
            return False
        n=n//2
    return True


# finding occuring n elements that occur odd times
print(ispow2(4))

def ispow2_bitmagic(n):

    if n==0:
        return False
    return n&(n-1)==0

print(ispow2_bitmagic(18))

# check for two odd occuring naive approach

def odd_occuring(lst):
    lt=set()
    for i in lst:
        count=0
        for j in lst:
            if i==j:
                count+=1
        if(count%2!=0):
                
                lt.add(i)
    return lt

lst=[20,20,10,10,10,26]
print(odd_occuring(lst))

def odd_occuring_two_opt(lst):
    xor=0
    res1=0
    res2=0
    for i in lst:
        xor=xor^i
    # here we have finded the xor for ex:[2,2,5,6] -> o/p ->5,6 so we will be getting 3 after this
    #so now considering into 2 groups by considering the first set bit
    set_bit = xor & ~(xor-1)

    for i in lst:
        if(set_bit&i==0):
            res1=res1^i
        else:
            res2=res2^i

    print(res1,res2)


odd_occuring_two_opt([2,2,3,3,3,3,5,6])


# optimized case 
'''
for example s='ab' 

i = 1<<s.len
j=  s.len

i = 000,001,010,011
j = 000,001

logic:i&(1<<j)!=0:

i=0
j=0
000
001
o/p=000

i=0
j=1 (in j we are iterating 1 by j number)

000
010
o/p=000

when i=0 ==> o/p = 000+000 = 000 = ''

i=1
j=0
001
001
o/p=001

i=1
j=1
001
010
o/p=000

when i=1==>o/p =001+000 = 001
print(when o/p!=0 print arr[j]) ==> arr[0] in 1st case => 'a'

i=2
j=0

010
001
000

i=2
j=1

010
010
010

when i=2==>o/p=000+010=010
=> arr[j] -> in 2nd case =>'b'

i=3
j=0

011
001
001

011
010
010

when i=3 ==>o/p= 001+010 => 011

o/p=> 'a'+'b'


'''

def sub_seq(xyz):
    l = len(xyz)
    no_possibilites= 1<<l

    for i in range(no_possibilites):
        for j in range(l):
            if(i&(1<<j)!=0):
                print(xyz[j], end=' ')
        print(' ')

print('subseq')
sub_seq('ab')


'''
** Terminologies used in the tree.

tree : collection of entities->basically independent elements (nodes) linked together to simulate a hirarcy

Trees are non-linear data structures that goes from top to bottom but doesn't go from bottom to top , the top one is root and rest are nodes that stores
data

root : the node that is not having any parent/ top most element in the hirarchy.

node: stores some information and links to another node

parent node: the immidiete predicissor(previous node to that node) of any node 

child node: the immidiete successor(next node) of that node are child node/nodes

grand parent : A(Grandparent to C)--> B --> C

leaf node/ external nodes: the node which is having no child i.e called leaf node

non-leaf nodes/ internal nodes: which are having atleast one child

Edge: link between 2 nodes

path: sequence of consecutive edges from source node to destination node.
 
Ancestor: any predecessor node on the path from root to that node.

A->B->C->D , Ancestor of D is C,B,A

Decendent: any successor node on the path from that node to leaf node

A-->B ->C -> D , Decendent of B is C,D

so with this we can find the common ancestor and common decendent

Sub tree: it contains a node with all its decendents 

sibling: when the node have the same parent they are called sibling nodes

degree : the number of children of that node is called a degree of the node.

degree of tree: maximum degree among all the nodes

depth of a node: the length of path/ no of edges from root to that node.

height of a node: no of edges in the longest path from that node to leaf

The height and depth may or may not be same.

level of the node == depth of the node

level of the tree == height of the tree 

height of the tree == depth of the tree (only in the balanced tree condition because in the hight we will be checking the longest path from node to leaf)

condition:

if n nodes then (n-1) edges should be there. if u add one more edge then it form a cycle in tree we cannot form a cycle, it should be always  Acyclic.

n nodes = (n-1) edges 

Binary tree : where each node can have atmost 2 children referred as left child and right child.

in general memory it is stored as left link --> left child and right link-->right child.

Application example : tree is going to store hierarical data so the best example is file system.
so trees are used to implement file system and in routing protocal also trees are used.
quick search --> binary search tree

In graph theory trees are connected & Acyclic

Dynamic Programming:

Dynamic programming is an algorithm paradigm to solve the given complex problem by breaking into sub problems and memorizing the outcomes of those
subproblem to prevent the repetative computations.

properties: 
1. optimal substructure (if we can formulate the recurrance relation to it)
2. Overlapping subproblem: The recurring problem is considered to have overlapping subprob if it solves the same sub problem again and again.
two methods to store the results:
1.memorization: store the result when the particular sub-problem is solved for the 1st time, top to bottom approach like in general recurrsion code and a lookup table is maintained for storing the computed results of every state, its recursive process  
2.tabulation: we pre compute the solutions in a linear fashion and store in table format, bottom to top approach, it is iterative process
this pattern of overlapping subproblems is called dynamic programming.

'''
# dynamic programming
# type-1: memoization
def fib_memoized(n, mem):
    if n in mem:
        return mem[n]
    print(mem.items())
    mem[n] = fib_memoized(n - 2, mem) + fib_memoized(n - 1, mem)
    return mem[n]

print('using DP0', fib_memoized(10, mem={0:0,1:1} ))





# type-2: tabulation
def fib_tabulation(n):
    lst = [0] * (n+2)
    lst[0]=0
    lst[1]=1
    for i in range(0, n):
        if i<n:
           lst[i+1] = lst[i]+lst[i+1]
           lst[i+2] = lst[i]+lst[i+2]
    return lst[n]
print(fib_tabulation(10))

'''
Graphs

Data structures -> 2 types 1.linear data structures 2. non-linear data structures

1.linear data structures -> 1. static, ex:arrays 2.dynamic --> ex: linkedlist, stack and queue

2.non-linear data structures -> 1.tree 2.graph

Graph: it's a non-linear data structure that have finate number of edges and nodes (vertices)

vertices: where the data us stored eg:A,B,C,D,E

Edges: which connect the vertices i.e lines between alphabets

Applications of graphs: flight network , product recommendations, facebook, google maps etc

lets take fb: lets A,B are 2 users which are nodes/vertices here , if A&B are friends then there will be a edge connecting them.

friends suggestions also works on this concept it checks the users you are frnds with and mutual friends and it will give you the suggestions
algo can be complicated but the basic data structure is graph. always the nodes are having the data and edges are for connectivity

and google maps also uses the graph, when u enter a destination it tries to find the destination in shortest distance and time.

there are various algorithms in graphs to find the shortest distance from A to B/ between 2 edges, and also route planning

Product recommendation(Amazon): if user A buys some products(like combo) and if a user b buys one of these products then there will be more likely
to buy the other products also so to find the mapping there we will be using the graph data structures

graph properties:
1.connected graph/disconnected graph
2.directed/undirected graph
3.weighted graph
4.cyclic/acyclic graph
5.dense/sparse graph
6.simple graph
7.complete graph
8.strongly connected graph

Back tracking --> Brute force approach (brute force is trying all the possible solutions and finding the desired one ):

this brute force is already followed in DP, but in DP we are solving the optimization problems.

back tracking is used when you have multiple solutions and you need all those solutions, it generates state space tree to solve

from all the solutions we will select few based on the constraints in the back tracking for example

3 seats , boy1,boy2 and a girl are there .. no of possible ways is 3! , constraint : girl should not sit in the middle .. find no of ways they can sit.

when the constraint is imposed(possibilty of girl in middle ) then the node is not generating further then we are killing the node,(using the condition)->that we called as bounding function

so we will apply the bounding function and kill the node.

*similar to back tracking we have other approach that uses brute force to solve it is branch and bound, difference is in back tracking DFS is 
used and in branch and bound BFS is used. 


'''



# DP -> memoize -> top to bottom approach , tabulation -> bottom to top approach.

def fib_mem(n,xyz={}):
    if n in xyz:
        return xyz[n]
    if n<=2:
        return 1
    xyz[n] = fib_mem(n-1)+fib_mem(n-2)
    return xyz[n]
print('using',fib_mem(10,{}))

def fib_memoized(n,memo={}):
    if n in memo:
        return memo[n]
    if n<=2:
        return 1
    
    memo[n] = fib_memoized(n-1)+fib_memoized(n-2)
    return memo[n]
print('using DP',fib_memoized(10))

'''
Graphs prep for final exam:
# representation of graph in computer:
representation of graph in computer can be done using 2 methods:
1.Adjacency matrix
2.Adjacency list

1.Adjacency Matrix
here matrix is basically the general mathematical matrix: M*N : M is no of rows and n is number of columns
in the case of the adjacency matrix is basically represented as n*n matrix where n is the number of vertices of the graph

ex : 5*5 Matrix
   1  2  3  4  5
 1[             ]
 2[             ]
 3[             ]
 4[             ]
 5[             ] 

 ex: if for the graph if there any 1 --> 1 connection then will mark as 1, if no loop then zero similarly 1->2 if it is undirected graph then
 both 1->2 and 2->1 will be marked as 1

 space complexity: n^2 in general for n*n

 2.Adjacency list

as name suggest for each vertex we are having the linkedlist, for each vertex one linked list is maintained

for each vertex one linked list would be there

ex: 
vertex 1 --> linked list that contains the adjacent nodes to this node i.e nodes that all connected to vertex 1
vertex 2 --> similar to vertex 1 all other vertex's
vertex 3 -->
vertex 4 -->
vertex 5 -->
total number of vertices will be equal to number of linked list.

for the  dense graph it is better to use adjacency matrix and in the case of sparse graph adjacency list is prefered

for adjacency matrix space complexity is 0(n+2e) in case if it is undirectional ,(+2e because of un directional where one element can be
written as a->b and b->a too) 

Two types of graph traversal techniques: 

*Generally in trees for BFS  we move traverse horizantally of each node and DFS we traverse vertically in the depth of each node but it is different
in the case of the Graphs.

1.BFS (Breadth first search) is also known as the level order traversal

2.DFS (Depth first search)

Important points:

In BFS traversal when you start then you can take any node as the root node , can start traversing the graph from any node, if its given in the 
question to consider a root node of some point, then we should start traversing from that point.

Queue data structure(FIFO) is used in the BFS traversal

steps:

all the adjacent vertcies that are as close to root vertex are traversed first (in any order you can insert)
By moving next consider all the unvisited vertices 


slicing = [start:stop:step] . step is where you step 

ex : l = [10,20,30,40,50], l[0:5:2] = [10,30,50]

l[ :4] = [10,20,30,40]
l[2: ] = [30,40,50] when no stop, stop is considered as the length of the list, and stop element won't be considered for calculation.

list slicing returns a list , tuple slicing returns a tuple  and string slicing returns a string

the difference is incase of when you are slicing a copy same from one list to another , you always get a different list as output but in tuple and string they return same object because they are immutable because python does it in that way

comprehensions in python : list comprehensions, set comprehensions and a dict comprehensions.

list comprehension: new_list = [expression for item in iterable if condition]

else in a for loop is only happened only in python. 

'''

list = [10,20,30,40,50]

print(list[0:5:2])

# using the list comprehensions
new_list = [x*x for x in list[0:4]]
print('using the comprehension the new list is',new_list)
# o/p: [100, 400, 900, 1600]

# using the dictionary comprehensions 

d1 = {x:x*x for x in list[0:]}
print(d1)
# {10: 100, 20: 400, 30: 900, 40: 1600, 50: 2500}

dict = {1:'sandeep', 2:'nandu', 3:'sam'}
# using a dictionary comprehension we get the inverted dictionary 
dict2 = {y:x for (x,y) in dict.items()}
print('reversing the key value pairs using the dictionary comprehension', dict2) 

# list = [10,20,30,40,50]
# list = [12,12,12,12,12]
list = [40,8000,2000,30,10]

# only for the greatest element. 

def largest_element_list(list):

    for i in range(len(list)):
        for j in range(i+1,len(lst)-1):
            if list[i] < list[j] or list[i] == list[j]:
                break
        else:
                return list[i]

    return None

print(largest_element_list(list))


def largest_element_list1(list):
    largest = 0
    for i in list:

        if i>largest:
            largest=i
    return largest

print('using the efficient sol',largest_element_list1(list))

def second_largest_element(list):
     lar1= largest_element_list1(list)
     lar2=0

     for i in list:
         if i!=lar1 and i>lar2:
             lar2=i
     return lar2
print('using the efficient sol',second_largest_element(list))

listsorted = sorted(list)
print('sorted',listsorted)
list.reverse()
xyz=[11,12,12,13]

# returns a reverse iterator and we are converting into lst 
print(list)



# or else using the slicing u can reverse 

lis = [1,2,3,4,5]
revlis= lis[-1::-1]
print(revlis)

def funrev(lis):
    s=0
    e=len(lis)-1

    while s<e:
        lis[s],lis[e] = lis[e],lis[s]
        s=s+1
        e=e-1
    return lis

print('usingmyownfun',funrev(lis))

'''
Recursion: it can be direct recursion (function calls itself) or indirect recursion (where the other function calls this function and again the other function is called from this function x calls y and y calls x)

Applications of the recursion: 
many algo's and problems are based on the recursion:

algo's:
1.Dynamic programming
2.back tracking
3.divide and conquer (most of these algo's uses the recursion ex: quick sort and merge sort)

they are basically recursive in nature.


problem's:

tower of hanoi
DFS based traversals (DFS graph traversal's inorder,preorder and post order traversal)

real world example for DFS: searching for a file in the computer.





'''

def fact_rec(n):
    if n==0:
        return 1
    
    return n*fact_rec(n-1)
n=5
print('using the recurssion the factorial of', n , 'is',fact_rec(n))

n=7
def fib_rec(n):
    if n==0:
        return 0
    if n==1:
        return 1
    
    return fib_rec(n-1)+fib_rec(n-2)
print('using the recurssion the fib of',n,'is',fib_rec(n))

# tail recurssion : a function is said to be tail recursive if the function doesnot do anything after the last recursive call
# --> a function which is a tail recurssive is typically optimized by the modern compilers and its good to have the tail recurssive functions

'''
you can convert any tail recursive function to a non recursive function in py

example of tail recursive function:

quick sort and post order tree traversal

where as the merge sort is not the tail recursive algorithm, that's the reason the quick sort is faster than the merge sort because  quick sort can be quickly optimized

in other lang like c and cpp etc they do tail call elmination automatically thats the reason it is optimized automatically

'''

def rec_sum(n):
    if n==0:
        return 0
    return n%10+rec_sum(n//10) 

print('sum using the recursion',rec_sum(984))

def rec_palin(string,start,end):

    if start>=end:
        return True
    
    return (string[start] == string[end] and rec_palin(string,start+1,end-1))

print(rec_palin('sssssss',0,6))

'''
Binary Search: 

binary search is a optimal sorting algorithm for a sorted array.
if the value is present it will return the index of the element
else it will return the -1

10 20 30 40 50 
0  1  2  3  4
'''
# binary search using the recursion
#  --> both of them have the logn in iterations but the main difference is the iterative solution is going to take 0(1) auxilary space but the recursive solution is going to take O(logn) auxilary space so the iterative is prefered  
def BS(arr,start,end,val):

    # never miss the base case in the recursion
#   base case   
  if start<=end:
    mid = (start+end)//2

    if arr[mid]==val:
        return mid
    elif arr[mid]<val:
        return BS(arr,mid+1,end,val) #recursion call
    else:
        return BS(arr,start,mid-1,val)
  else:
        return -1 # return if the value is not found edge case.

arr=[10,20,30,40,50]   
print(BS(arr,0,len(arr)-1,20))

'''
in linear search it takes O(n) time but with the binary search
it takes O(logn) time but the consition is that the array should be always in the sorted order

time complexity is O(logn) --> anywhere between O and logn for the succesfull search for the unsuccessful search it is theta(logn) --> exactly logn
'''

# first occurance of the element using the binary search 




arr=[10,10,10,10,10]
def first_occurance_bs(arr,val,start,end):

    if start<=end:
        mid = (start+end)//2
        if arr[mid]>val:
            first_occurance_bs(arr,val,start,mid-1)
        elif arr[mid]<val:
            first_occurance_bs(arr,val,mid+1,end)
        else:
            if mid == 0 or arr[mid] != arr[mid-1]:
             return mid
            else:
               return first_occurance_bs(arr,val,start,mid-1)            
    return -1

print('first occurance',first_occurance_bs(arr,10,0,4))

# now the last occurance of the element using the binary search 

arr=[10,10,10,10,10]
def last_occurance_bs(arr,val,start,end):

    if start<=end:
        mid = (start+end)//2
        if arr[mid]>val:
            last_occurance_bs(arr,val,start,mid-1)
        elif arr[mid]<val:
            last_occurance_bs(arr,val,mid+1,end)
        else:
            if mid == len(arr)-1 or arr[mid] != arr[mid+1]:
             return mid
            else:
               return last_occurance_bs(arr,val,mid+1,end)            
    return -1

print('last occurance',last_occurance_bs(arr,10,0,4))

# just for checking the total number of occurances(As the number is in the sorted order) the formula is = last ocuurance - first occurance + 1 

arr=[10,10,10,10,10]
def check_no_of_occurances(arr,val,start,end):
    first_occurance = first_occurance_bs(arr,10,0,4)

    if first_occurance != -1:

        return last_occurance_bs(arr,val,start,end)-first_occurance_bs(arr,val,start,end)+1

print(check_no_of_occurances(arr,10,0,4))


def squareroot(x):
   low = 1
   high = x
   ans=-1
   while low<=high: 

    mid = (low+high)//2
  
    sqr = mid*mid

    if sqr == x:
        return mid
    
    elif sqr>x:
        high = mid-1

    else:
        ans = mid
        low=mid+1
   return ans

print('sqrt using the binary search',squareroot(20))

'''
sorting: python has its own built in methods for the sorting
using sort() and sorted() built in methods

sort => use for only list to sort inplace
sort() : work only for list
this will sort in place(where it sorts the orginal list)

sorted ==> used for all iterable containers to create a new sorted container
sorted():
works for any iterable, like for the popular containers list,string,tuple,dictionary and set they are iterable and many of them are immutable like tuple,string 
doesnot modify the past container, return a list of sorted items by creating a new list.
both of these use tim sort and are stable, tim sort is the hybrid algorithm that uses the merge sort and the insertion sort internally 
TC = O(nlogn), stabilty mean: if you have two keys with same values thier originality order is retained
'''
arr=[50,30,10,40,20]
arr.sort()
print('the sorted array is',arr)
arr=[-50,30,10,40,20]
sortedarr = sorted(arr)
# sorted is basically used for all the iterable containers, it basically returns another sorted arr doesn't modify the original one
print('iterative sorting method',sortedarr)

# sorting the list in the decreasing order 
arr.sort(reverse=True)
print("sorting in decending order",arr)

# lexicographical order
arr = ["anjuuu","sunny","hema","nandu"]
arr.sort()
print('string sorting',arr)

def myfun(s):
    return len(s)

arr.sort(key=myfun)
print('sorting using the func',arr)

# sort takes basically 2 inputs key and reverse --> sort(key=myfun, reverse=True)

class Point:

    def __init__(self,x,y):
        self.x=x
        self.y=y
    # method for user defined objects,it defines the ordering among the objects
    # this in built method is useful when your class has a natural order 
    '''
    def __lt__(self,other):
        return self.x < other.x
    '''
    def __lt__(self,other):
        if self.x==other.x:
            return self.y<other.y
        else:
            return self.x < other.x
    
    #once this func is define no need to define again in sort(key)

l=[Point(1,26),Point(10,1),Point(1,25)]
# print(l)
l.sort()
for i in l:
    print(i.x,i.y)
# So, while __lt__ is not required for sorting, it provides a way to customize the sorting order for instances of your class.
# so according the method that is required you can sort by using sort method
# sorted works for all iterables those are mutable too, returns with sorted items, parameters like reverse are key works same as sort
# string sorted --> we get chars sorted , dict--> keys sorted, if list of list or tuple then first items are compared first for sorting (natural)       
       
'''
stable sort: if both values are matched then it goes/consider the original order how its given

example of stable sorts: bubble sort, insertion sort , merge sort

example of unstable sorts: selection,quick,heap (actually disturb the original order of values that's the reason they are not stable)
'''
'''
Hashing is the technique mainly used for dictionary where you have key value pairs and also sets where you have set of keys
the best thing about hashing is it performs very well on search,insert and delete operations in O(1) on average.
In hasing the duplicates are not allowed if the key already exists then it will overide the key

areas where the hashing is not recommended for:

1.finding the closest value (un order collection)
2.for the sorted data (because they are not in order, unorder collection)

in case of 1 and 2 we use the self balancing binary search tree like AVL or red black
3. prefix searching 
if you want a particular key that matches the prefix it wont work because hashing does the exact key searching.

for case 3 prefix searching trie data structure is best one, for strings this trie data structure is alternate dictionary implementation that also provide the quick prefix search which is not there with hashing,it is used to solve all the algorithms efficiently


Applications of hashing:

hashing is the most used technique and hashtables are most used data structures after the list 
Applications of hasing are:

1.dictionaries
2.data base indexing
3.cryptography
login --> password --> never store in plain text --> generate hash out of it --> store the hash --> when u login again the hash is function is computed if the stored hash matches with this computed one then the user is allowed
4.caches(browser cache: url--> key , data associated becomes value )
5.routing(device (mac address) for an IP address ), getting data from DB etc

direct address tables:
for short range , stores value as value as index , if the element is present value is 1 else value is 0 (first all the values will be 0's).
This DAT handles search,insertion & delete when your keys are small where the array of size memory can be created and we can do all the operations in O(1) time

problems with direct address table approach why do we need another data structure like hashing?

1. it cannot handle large values ex:10 digit mob num
2. floating numbers you cannot use these numbers as index
3. keys can be strings or addreses or combination of strings and numbers that is where hashing comes in.


hashing: the idea is similar to direct address tables but we use hash function in between to convert the large values into smaller and use the small value as the index in the array
and we call this array as the hashtable. 

large values --> (hash function) --> small values.
'''

# how hash function(magical function) work? 

'''
should always map a large key to small key

should generate values from o to m-1

should be fast, O(1) for integers and O(len) for string of length len (as each string should process through each char)

should uniformly distribute large keys to hash table slots

if hash table size is 100.

then chance of 10^10 phone numbers of any 100 phonenumbers can be fit into hash table using the hash function
the hash table size is proportional to n (number of keys going to insert)

collison can be :
2 large numbers can result into same small value this problem is called collision

ex: ph num % x --> 2 large values can result in same small val

*universal hashing --> having set of hash functions pick one of them to hash all of the data, the idea is to make sure that expected time is O(1)

collisions are bound to happen if you do not know the keys in advance 

if we know the keys in advance then we can do perfect hashing

if we do not know keys, then we do one of the following:

1. chaining : make chain of colliding items

2. open addressing --> 
we use the array & if a position is occupied we tried to put the key in the other slot. 


methods: linear probling, quadratic probling, double hashing

chaining: 

ex: hash(key) = key%7

keys = {50,21,58,17,49,56,22,23,25}

arr 

0 21 --> 49 -->56

1 50 --> 15 --> 22

2 58 --> 23

3 17

4 25

5

6

performance of chaining: load factor = n(no of keys to be inserted)/m(no of slots)

as the load factor big or hash table is small then collisions are more, so we want the load factor to be small.

expected chain length for the random set of keys ? 

in the worest case it can be every key will go to the same chain

if the keys are uniformly distributed in the hash table, if n keys and n hash table slots  
then the expected chain length = alpha (load factor-> n/m)\
ex: (uniformly distributed 100/100) alpha = 100/100 = 1 

under the assumption of uniform distribution of keys in the hash table by the hash function
TC of search, insert and del = O(1+alpha) a for the hash function computation and alpha for the chain length

data structures for storing chains:
1. linked list (search,delete and insert is O(len), more disadv like its not cache friendly because it is stored in diff locations & uses extra space for next references
2. dynamic sized arrays(list) --> same search,insert and del is O(len) and the adv is they are cache friendly as they store in contiguoes loc
When data is stored in continuous or contiguous locations, it enhances cache-friendliness because of the way caches work. Caches often use a principle called spatial locality, which means that if a particular piece of data is accessed, it's likely that nearby data will also be accessed soon.
3. self balancing BST( AVL Tree, Red Black Tree)
search,insert and delete --> O(logl), they have one disadv they are not cache friendly 


open addresing:
it is another way of handling collisions, the idea is to use single array only, not to forms any chains of any other data structure 
one basic req for open addressing : no of slots in hash table(hash table size) >= no of keys to be inserted 
advantage if open addressing over chaining: it is cache friendly, so its more likely to have fewer cache misses.

3 ways of implementing the open addresing : linear probing, quardratic probing and double hashing  
linear probing: when their is a collision occur we linearly search for the next empty slot

search: for search we compute the hash function, we go to that index and compare if we find the key we rtturn true or otherwise we linearly search, 
we stop searching when 
1. we either find the key or
2. find the empty slot or 
3. we have traversed throught the whole table

problem when simply making the slot empty with delete:

when you delete don't make it empty because when you make empty the next elements next to empty while they are searched stops at this empty and they are not searched (search fails) so we will make these slots as deleted.
and we say don't stop the search when you see deleted slots, stop the search when you see only empty slots, and insert can insert in a deleted slot

problem with the open addressing(linear probing):

clustering --> with this the operations (search, insert, delete) becomes costly.

ex: when you search some thing at that index it is not there so u have to travel through whole cluster so this will impact the performance of open addressing

how to handle the clustering with the linear probing?
linear probing = hash(key,i) = (h(key)+i)%7

in linear probing we go to next slot
1.quardratic probing : (hash(key) + i^2)%m
in quardratic probing rather than going to next slot we go to i^2 slot
so there is also a problem in quardratic probing there are secondary clusters formed but it is better than primary clustering
there is one more disadvantage with the quardratic probing, even though if there are empty slots it might not found the empty slots, so you need to have the double the table size(load factor alpha is less than 0.5 size and n/m where m(hash table size) is prime number them only it works) for this quardratic probing
so an alternative idea for it is double hashing.

2.double hashing: h1(key)+i*h2(key))%m (your h2 function shouldn't return zero)
ex: h1= (key%7) h2 = 6-(key%6), when ever the collision happen then we use the i, for 1st collision i=1, 2nd collision i=2 and so on ...
in double hashing we use 2 hashfunctions one hash function is the original function and the other hashfunction is  to find the next slot
while we are probing for a free slot, in quardratic probing there is a issue that you need double size to ensure that you find a free slot and also m should be prime
the big advantage of double hashing is if h2 key and m be relatively prime(both don't have a common divisor other than 1) then you don't need double size, if there is a free slot u always get one
and also it avoid clustering, no clustering

dynamic structure insert more and more keys then --> chaining
collision handling has better performance in chaining compared to open addressing but chaining has poor cache performane where as open addressing is cache friendly.
'''
# set in python 

'''
--> distinct elements

--> unordered

--> no indexing

--> union(A union B -> all elements from a and b), intersection(A intersection B common elements from a and b)--> ), set difference, etc are fast the operations like searching are really fast with hashing, thats reason these union,intersection and set diff are fast.

--> uses hashing internally (thats the reason the operations are faster with hashing)

for list search takes O(n) where as for hashing it takes O(1) in case of large data this makes a huge difference
For scenarios where fast search, insertion, and deletion operations are crucial for large datasets, alternative data structures like sets, dictionaries, or specialized data structures optimized for specific use cases might be more appropriate.

If you need fast membership tests, have a large dataset, and can design a good hash function, hashing may be more efficient.
If maintaining order is important, and the dataset is small or linear search operations are acceptable, a list may be more suitable (Cache friendly).

so if you use hashing searching for the item is really going to be fast for intersection , union searching is very imp

that's the reason the union,intersection operations are really fast.

with .update() you can add elements from other collections items/multiple collection items to set

in(x in y) operation is faster in set because it uses hashing(searching) internally rather than list
'''
s1 = {2,4,6,8}

s2 = {3,6,9}

#in the context of set or is union
print(s1 | s2) #union

print(s1.union(s2)) #another way, it doesn't modify set 1 or set 2 both the sets remains as it is

print(s1 & s2 ) #intersection

print(s1.intersection(s2)) #another way


print(s1-s2) #difference

print(s1.difference(s2)) #another way

print(s1 ^ s2 ) #symmetric difference ( union but wont give the common elements in 2 sets)

print(s1.symmetric_difference(s2)) #another way

print(s1.isdisjoint(s2)) # return boolean if common elements present it will return false

print(s1 <= s2) #s1 subset of s2 false because it doesn;t have all elements of s1 in s2

print(s1 < s2) #doesn't allow equal set too , proper subset

print(s1 >= s2) # false it is not super set because it doesn't have all elements of s2 in s1

print(s1 > s2) #s1 proper super set or not, false if even both sets are same

print('superset',s1.issuperset(s2)) # true if all elements of s2 present in s1 

d={}
d['name'] = 'sandeep'
print(d.get('name'))

# count distinct elements 

s = {26,3,2001,26,3,2001}

st = set(s)
print('length of set',len(st))
print(st)

# strings in python - seq of characters, used to store text data like data read from a file
#typically set of characters, characters 'A' to 'Z' are stored values from 65 to 90 and characters 'a' to 'z' are stored from 97 to 122
# ascii consits of 128 characters, unicode: consists for other lang too like chinease/hindi
# raw string --> if we just write r before thhe string it becomes raw string so the escape sequence like /n next line will not also count, and escape characters \


#using the formatted string in python 1. using %, using format , using f-string

firstname = "sandeep"
lastname = "paruchuri"

s="welcome %s %s " %(firstname,lastname)
print(s)

s= "welcome {0} {1}".format(firstname,lastname)
print(s)

s = f"welcome {firstname} {lastname}"
print(s)

print(f"upper case of {firstname} is {firstname.upper()}")

# in python characters are stored using the uni codes where as in other lang like java or cpp  it is ASCII codes , the difference is ASCII just supports 128 characters and extended ASCII supports 256 characters where as uni code support large range of char (other lang too), ASCII values are same as unicode values but in unicode there are large range of char
# using ord('a') u can get the unicode value in py
# in string comparison first char of one string is compared with first char of other string and if they are equal it compares till end & so on when one string finishes and other string has extra char then the string with extra char is treated as largest ex: "abcd">"abc", "ZAB">"ABC", "z">"abc".
# mainly it checks the first char of the string with the other string's first char then it makes the decision.

'''
#string operations --> check if it's substring, it prints boolean value

'''

s1 = "geeksforgeeks"

s2 = "geeks"


print(s2 in s1 )
print(s2 not in s1 )

print(f"index of {s2} in {s1} is ", s1.index(s2))
print('reverse index',s1.rindex(s2))
print(s1.index(s2,1,13))

# imp --> these will raise error(value error) if the substring is not present 

str = "GeeksforGeekspythoncourse"

print(str.startswith("Geeks"))
print(str.endswith("course"))
print(str.startswith("Geeks",1))
print(str.startswith("Geeks",8,len(str)))

#split function : by default split method splits with spaces
str="geeks for geeks "
print(str.split())
str="geeks, for, geeks"
print(str.split(", "))
str="geeksforgeeks"
print('using join'," ".join(str))
s="__gfg__"
print(s.strip("_"))
# lstrip - left shift and rstrip - right strip
print(s.lstrip("_"))
print(s.rstrip("_"))

# similar to index function we have find but the main difference is with index function if the substring is not found it throws value error but with the find it wont

s1 = "geeksforgeeks"
s2 = "geeks"

print('Using find',s1.find(s2,1,len(s1)))

num=0
 
# 0 - 48 , 1-49 , 2-50 , 3-51 so to get exact value when string to int 
print(ord('1')- ord('0')) #then you can append to a number when appending multiply with 10 so the digits also counts
 
def asciiToSentence(string, length) :
	
	num = 0; 
	for i in range(length) :

		# Append the current digit 
		num = num * 10 + (ord(string[i]) -
						ord('0')); 

		# If num is within the required range 
		if (num >= 32 and num <= 122) : 

			# Convert num to char 
			ch = chr(num); 
			print(ch, end = "");  # print the char in the same line with out moving to next line

			# Reset num to 0 
			num = 0; 
# 'A' to 'Z' are stored values from 65 to 90 and characters 'a' to 'z' are stored from 97 to 122
# ord to convert the char to ascii and chr to convert the number to a char 

str='sandeep'
# reversing a string 
print(str[:-2:-1]) #start stop step
print(str[::-1])
rev=""
for i in str:
    rev = i+rev

print("rev of str",rev)

#check if the string is rotated or not 

'''
logic : you can one way either clock wise or anti clock in both ways while checking you can obtain all the check of n elements
ABCD 

BCDA  DABC

CDAB  CDAB

DABC  BCDA

every counter clock wise rotation is equal to a clock wise rotation
'''

# naive 

def checkrotated(str,str1):
    for i in range(0,len(str)):

     str = str+str[i]
    #  print(str)
     Ans= str[i+1:len(str)]
    #  print(Ans)
     if Ans == str1:
         return True
    return False

print(checkrotated("ABCD","ABCD"))


def efficient_checkrotated(str,str1):
    if len(str)!=len(str1):
        return False
    
    ans = str+str
    return ans.find(str1)!=-1 #using find it gives the index, if not found -1 
print(efficient_checkrotated("ABCD","ABCD"))

def check_palin(str):
    low = 0
    high = len(str)-1
    while low<high:
      if str[low]!=str[high]:
        return False
      low=low+1
      high=high-1
    return True

print(check_palin("abaaa"))

def efficienct_palin(str):
    if str == str[::-1] : 
        return True
    return False
print(efficienct_palin("aaaa"))


# substring vs subsequence --> substring means the contigouescharacters, subsequence means where we can remove the char from the middle 

'''
the total number of subsequences of length are going to be 2^n

All subsequences of ABC are
"",A,B,C,AB,BC,AC,ABC
'''

def sub_seq(str):
    for i in range(1<<len(str)):
        curr_sub_seq=''
        for j in range(len(str)):
            if ((1<<j)& i)!=0:
                curr_sub_seq = curr_sub_seq+str[j]
        print(curr_sub_seq)
sub_seq('ab')

# efficient way 

# check if all the characters of the substring  are present in the  string in the seq order and the length matches

def checksub_seq(str,substr):
    i,j=0
    while (i<len(str) & j<len(substr)):
        if (str[i]==str[j]):
            j=j+1
        i=i+1
    if len(i) == len(j):
        return True
    else:
        return False
    
# check subsequence using recursion 
def eff_checksub_seq(st,subst,i,j):
    if j==0:
        return True
    
    if i==0:
        return False
    
    if st[i-1] == subst[j-1]:
        return eff_checksub_seq(st,subst,i-1,j-1)
    else:
        return eff_checksub_seq(st,subst,i-1,j)
    
print('using recurssion', eff_checksub_seq('abcd','ab',4,2))

# check anagram , anagram --> ex: listen, silent (all the no of characters should match)


def check_anagram(str,str1):

    if(len(str)==len(str1)): 

        for i in range(0,len(str)):
            x = str[i]
            if str.count(x) != str1.count(x):
                return False
        return True
    return False
    
    
print(check_anagram('listen','silent'))



def check_anagram_efficientsol(str,str1):
    if (len(str)!=len(str1)):
        return False
    
    count = [0]*256
    
    for i in range(0,len(str)):
        count[ord(str[i])]  +=1
        count[ord(str1[i])] -=1
    
    for x in count:
        if x!=0:
            return False
        else:
            return True
        
print('anagram efficient sol', check_anagram_efficientsol('ssilent','listen'))

def left_mostnonrepeating(s):
    count = [-1]*256
    res=0
    for i in range(len(s)-1,-1,-1):
        if(count[ord(s[i])] == -1):
            count[ord(s[i])] += 1
            res=i
        else:
            count[ord(s[i])] += 1

    return res

print('leftmostoccuringchar',left_mostnonrepeating("geeksforgeeks"))



def revwords_string(str):

    arr = str.split(" ")
    ans=''

    for i in arr:
        for j in range(len(i)-1,-1,-1):
                ans=ans+i[j]
        if i!=len(arr):
            ans+=' '     
        

    return ans

print(revwords_string("geeks for geeks"))

st = "geeks for geeks"
xy = st[::-1]

print(xy)

# reverse  function works for arrays, as string are mutable we use slicing [start stop step]

# s = "".join(s) --> when s is list , using join we convert to string , it join between 2 chars

#ord --> used to convert char to int ascii/unicode , where as char convert int to char 

# [start:stop:step] -- slicing of strings


# ---> LinkedList 

'''
problems with the list

the operations can be costly like when you insert the element in the middle you have to move all the elements to the next position as they have to be contigueous
ex: when you have 1000 elements in the array when you insert an element in the 2nd position then you have to move 998 elements to next so as the size grow this will be very costly operation
now the same applies for the deletion as well, deletion in the middle

so when you are taking about the stack insertions and deletions happens in the same end , so we can use the last end of the array and implement the stack
but about the queue and dequeue efficient implementation is not easy with array's because insertion happen in one end deletion happens in other end(when removing element in the start all the elements will also moves its position so its not effcient)
In dequeue insertion and deletion both happen in both the ends, its possible to do with the arrays in 0(1) but the implementation is complex but with linkedlist the implementation is super easy.

Time complexity in arr/list:

Insertion at the end(Append): O(1)
Insertion at the Beginning: O(n)
Insertion at the specific Index(middle): O(n)

Implementing round robin scheduling using array it going to be difficult removing at the start and some times remove at the start and inserting in the end(depends on the token time)

LinkedList: linked list --> ordered data structure has a pre defined order but the difference between linkedlist and array's are in contiguoes locations and in case of linkedlist they are not in contiguoes locations

head->(data,ref of next node)-->(data,ref of next node)-->(data,ref of next node)-->(data,none)
the references basically takes you to the next node, the ref of the last node is none that's how you know you reached the last node 

there are adv and disadv of dropping the contiguoes memory requirement

1.lists are cache friendly because of thier contiguoes memory and they are randomly accessible 

2. linkedlist: they are stored in diff locations and also they are not friendly and not randomly accessible, if you want some random 3rd item you have to traverse to complete linkedlist using head

Advantages of linkedlist 

1.suppose you have ref to a particular node and if you want to delete next of it you can delete in constant time
2.and also you can insert in constant time so in the case of linkedlist the insertions and deletions becomes fast.(Even insertions in the beginning are also faster in O(1))

LL: ADV --> faster insertions and deletion: Disadv --> not cache friendly, they are not accessible randomly

'''
class node:
    def __init__(self,k):
        self.key=k
        self.next = None
temp1 = node(10)
head = temp1
temp2 = node(20)
temp3 = node(30)
temp1.next = temp2  # or head.next = node(20)
temp2.next = temp3  # or head.next.next = node(30)
print(temp1.key)

# Applications of the linkedlist 
'''
insertions,deletions and at the middle are theta(1) 
round robin implementation
merging 2 sorted linkedlists is faster than arrays
implementation of simple memory manager 
easier implementation of queue and dequeue

'''
# Traversing a linkedlist  in python 

def printlist(head):
    curr=head
    while curr is not None:
        print(curr.key, end=" ")
        curr = curr.next
    print(" ")

#using the above driver code as class node
        

printlist(temp1)

# searching for a value using linkedlist  
def search(head,x):
    pos=1
    curr=head
    while curr is not None:
        if curr.key == x:
            return pos
        pos+=1
        curr=curr.next
    return -1

print('the pos of x in LL is', search(temp1,30))

def insertingatbegin(head,key):
    h=node(key)
    h.next=head
    printlist(h)
  

insertingatbegin(temp1,1)

def insertingatend(head,key):
    curr=head
    while curr.next!=None:
        curr=curr.next
    curr.next=node(key)
    
    printlist(head)


insertingatend(temp1,100)

# 10 20 30 40 50  
# need to insert at 4
# inserting data at a certain position

def insertatpos(head,pos,data):
    temp=node(data)
    if pos==1:
        temp.next=head
        return temp
    curr=head
    for i in range(pos-2):
        curr=curr.next
        if curr==None:
            return head
    temp.next = curr.next
    curr.next = temp
    printlist(temp1)
    return head
    

insertatpos(temp1,4,50)


# in LL (10 20 30 50 100) --> LL's don't have indexes we will be just traversing through entire list



def deletefirstnodell(head):

    if head==None:
        return None
    head = head.next
    printlist(head)

deletefirstnodell(temp1)


def deletelastnodell(head):
    if head == None:
        return None
    if head.next == None:
        return None
    curr = head
    while curr.next.next!=None:
        curr = curr.next 
    curr.next = None
    printlist(head)

deletefirstnodell(temp1)
deletelastnodell(temp1)


    
# key and data both are the interchangeable terms
# 10 20 30 40 50

def delnodewithpointer(pointer,head):
    temp = pointer.next
    pointer.key = temp.key
    pointer.next = temp.next
    printlist(head)

deletefirstnodell(temp1)
deletelastnodell(temp1)
delnodewithpointer(temp2,temp1)

# 10 20 30 40
def sortingll(head,val):
    temp = node(val)
    if head == None:
        return None
    elif val<head.key:
        temp.next=head
    else:
        curr = head
        while curr.next!=None and curr.key<x:
            curr=curr.next
        temp.next=curr.next
        curr.next=temp
    printlist(head)
sortingll(temp1,26)



def middleoflinkedlist(head):
    curr=head
    count=0
    if head == None:
        return None
    while curr:
        curr=curr.next
        count+=1
    curr=head
    for i in range(count//2):
        curr=curr.next
    return curr.key

print(middleoflinkedlist(temp1))

# 10 20 30 40 50 logic slow and fast pointer (Using the 2 pointer approach)
# --> slow pointer and fast pointer 
def midoflinkedlistoptimized(head):

    if head == None:
        return head
    
    slow=head
    fast=head
    
    while fast!=None and fast.next!=None:
        slow=slow.next
        fast=fast.next.next
    return slow.key

print(midoflinkedlistoptimized(temp1))

def nthnodefromendofll(head,n):

    if head == None:
        return None
    len=0
    curr=head
    
    while curr:
        len+=1
        curr=curr.next
    if len<n:
        return
    curr=head
    for i in range(len-(n+1)):
        curr=curr.next
    return curr.key

print(nthnodefromendofll(temp1,0))

# 10 20 30  
def nthnodefromendoflloptimized(head,n):

    if head==None:
        return None
    first=head
    for i in range(n):
        if first.next == None:
            return
        first=first.next
    second=head
    while first.next!=None:
        second=second.next
        first=first.next
    return second.key

print(nthnodefromendoflloptimized(temp1,0))

# 10 20 20 30 40 
def remdupsortedll(head):
    curr=head
    while curr.next!=None and curr!=None:
        if curr.data == curr.next.data:
            curr.next = curr.next.next
        else:
            curr=curr.next
printlist(temp1)

# 10 20 30 40 50

def reverselinkedlist(head):

    stack = []
    curr=head
    while curr!=None:
        stack.append(curr.key)
        curr=curr.next
    curr=head
    while curr!=None:
        curr.key=stack.pop()
        curr=curr.next
    return head

reverselinkedlist(temp1)
printlist(temp1)


# Using the recurrsion

'''
2 ways of the recursive approach : 1. reversing the list after head.next, then you link the reverse list back to the first node so the first node becomes the last nodew 

20 30 40 50 60 --> 20, rev(30 40 50 60), 20<-30<-40<-50<-60

2. first change the first link then make the recursive call
'''
# ex: 10 20 30 : 10->head, 20->tail, 30->rest head
''' 
10 20 30 40 : 10 head, rh = rev(20)
20 head, rh=rev(30)  => rest head= 40 , 20->(40->30) --> (40->30) ->20
30 head, rh=rev(40) ==> rest head = 40 , 40->30

rest_head = reverseList(head.next)
    head.next.next=head
    head.next=None

10 (20 30)

10 20 30  

h=10 rh=rev(20-30), rh = 30-20-none. return:30-20-none

h=20 rh=rev(30), rh=30, rt = 30,   

30->20->none
     | 
    10


'''



def reversell_rec(head):
    if head == None:
        return None
    if head.next == None:
        return head
    
    rest_head = reversell_rec(head.next)
    rest_tail = head.next
    rest_tail.next=head
    head.next=None

    return rest_head
    

# 10 20 30 40 50 
def reverselinkedlistoptimized(head):

    curr=head
    prev=None
    while curr!=None:
        next=curr.next
        curr.next=prev
        prev=curr
        curr=next
    print('revlist')
    printlist(prev)
    return prev

# rev = reverselinkedlistoptimized(temp1)
# print(rev)
reverselinkedlistoptimized(temp1)
printlist(temp1)































    







    









    
     













    

    

    
    




    















        




    

    








        
 




    


        

















        











        












    





    

    

    

    

      




















    
    
        







    











    






    


        




    
    













    

    












    







    










