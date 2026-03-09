from collections import deque
import random 
# adding a dequeue of size 5
buffer = deque(range(10), maxlen=5)
print('Buffer initialized as:', buffer)
# Append 10 to the right of the buffer
buffer.append(10)
print('Buffer after appending:', buffer)
print('Random sample of 3 elements from the buffer:', random.sample(buffer, 3))