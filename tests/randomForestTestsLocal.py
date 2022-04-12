from testCookies import testCookies
from testHomePage import testHomePage
from testRandomForestHealthy import testRandomForestHealthy
from testRandomForestRandom import testRandomForestRandom
from testRandomForestUnhealthy import testRandomForestUnhealthy

def randomForestTestsLocal():

    testCookies()
    testRandomForestHealthy()
    testRandomForestRandom()
    testRandomForestUnhealthy()

if __name__ == "__main__":
    randomForestTestsLocal()