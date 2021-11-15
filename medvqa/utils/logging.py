class CountPrinter:
    def __init__(self):
        self.count = 1
    def __call__(self, *args):
        print(f'{self.count}) ', end='')
        print(*args)
        self.count += 1