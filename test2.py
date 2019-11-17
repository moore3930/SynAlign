
a = 1
b = 2

print([a, b])
def func():
    global a, b
    tmp = a
    a = b
    b = tmp

func()
print([a, b])
