# 1. singly linked list
# 2. it contains n number of values
# 3. Q: Find the middle element


temp1, temp2 = head, head

while (temp2 != NULL and temp2.next != NULL):
    temp1 = temp1.next
    if temp2.next == NULL:
        break
    temp2 = temp2.next.next
return temp1

def isArmstrong(i):
    sum = 0
    j=i
    while i>0:
        a = i%10
        sum += a**3
        i = i//10
    return j==a  

for i in range(100, 999):
    if isArmstrong(i):
        print(i)
