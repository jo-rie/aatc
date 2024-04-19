from string import Template
import time


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    """Format the timedelta tdelta according to the format fmt"""
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


class Timer(object):
    def __init__(self, name: str = None):
        """Create a new Timer.
        :param name: Name of the timer
        :return: Timer object"""
        self.name = name

    def __enter__(self):
        """Start the timer"""
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        """Print the elapsed time"""
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))
