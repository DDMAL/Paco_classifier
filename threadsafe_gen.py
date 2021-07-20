import threading
#Credit: https://anandology.com/blog/using-iterators-and-generators/
class threadsafe_gen:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()