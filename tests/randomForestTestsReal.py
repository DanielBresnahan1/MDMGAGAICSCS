##
# @file testRandomForestTestsReal.py
#
# @brief Runs all tests concerning the random forest code, directed at the real application.
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.
from testCookies import testCookies
from testHomePage import testHomePage
from testRandomForestHealthy import testRandomForestHealthy
from testRandomForestRandom import testRandomForestRandom
from testRandomForestUnhealthy import testRandomForestUnhealthy

def randomForestTestsReal(url = "https://mdmgcapstone.herokuapp.com/"):
    """! Tests random forest functionality
        This test will simply run all tests concerned with the random forest functionality, but will also specify that they be run on the real application.
        @param url: The url which the test will attempt to reach. Defaults to the application for the real url.
    """
    testCookies(url)
    testRandomForestHealthy(url)
    testRandomForestRandom(url)
    testRandomForestUnhealthy(url)

if __name__ == "__main__":
    randomForestTestsReal()