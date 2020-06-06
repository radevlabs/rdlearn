import time
import numpy as np

class Timer:
    def __init__(self):
        self.years = 0
        self.months = 0
        self.days = 0
        self.hours = 0
        self.minutes = 0
        self.seconds = 0.

    def start(self):
        self.__start = time.time()
        return self

    def end(self):
        self.seconds = time.time() - self.__start
        self.seconds = float(self.seconds)
        self.minutes = self.seconds // 60
        self.hours = self.minutes // 60
        self.days = self.hours // 24
        self.months = self.days // 30
        self.years = self.months // 12

        self.seconds -= (self.minutes * 60)
        self.minutes -= (self.hours * 60)
        self.hours -= (self.days * 24)
        self.days -= (self.months * 30)
        self.months -= (self.years * 12)

        self.years = int(self.years)
        self.months = int(self.months)
        self.days = int(self.days)
        self.hours = int(self.hours)
        self.minutes = int(self.minutes)
        return self

    def summary(self, f=2, comma='.'):
        seconds = np.around(self.seconds, f) if f > 0 else int(self.seconds)
        t = [self.years, self.months, self.days, self.hours, self.minutes, seconds]
        s = ['year(s)', 'month(s)', 'day(s)', 'jam', 'menit', 'detik']
        smmr = ''
        for c in range(len(t)):
            if t[c] != 0:
                smmr += f'{t[c]} {s[c]} '
        return smmr[:-1].replace('.', comma)