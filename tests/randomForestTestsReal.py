from testCookies import testCookies
from testHomePage import testHomePage
from testRandomForestHealthy import testRandomForestHealthy
from testRandomForestRandom import testRandomForestRandom
from testRandomForestUnhealthy import testRandomForestUnhealthy

def randomForestTestsReal(url = "https://mdmgcapstone.herokuapp.com/"):

    testCookies(url)
    testRandomForestHealthy(url)
    testRandomForestRandom(url)
    testRandomForestUnhealthy(url)

if __name__ == "__main__":
    randomForestTestsReal()