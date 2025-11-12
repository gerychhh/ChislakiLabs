import time


def timeCheck(func):
    def wrapper(*args, **kwargs):
        timeStart = time.thread_time()
        func(*args, **kwargs)
        result = time.thread_time() - timeStart
        print(result)
        return result
    return wrapper

@timeCheck
def test(a, b):
    for i in range(a, b):
        print(i)



a = int(input("a: "))
b = int(input("b: "))


test(a,b)
