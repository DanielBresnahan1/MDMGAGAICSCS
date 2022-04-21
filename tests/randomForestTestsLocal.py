##
# @file testRandomForestTestsLocal.py
#
# @brief Runs all tests concerning the random forest code, directed at a locally hosted version of the application.
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.

from testCookies import testCookies
from testHomePage import testHomePage
from testRandomForestHealthy import testRandomForestHealthy
from testRandomForestRandom import testRandomForestRandom
from testRandomForestUnhealthy import testRandomForestUnhealthy

def randomForestTestsLocal():
    """! Tests random forest functionality
        This test will simply run all tests concerned with the random forest functionality.
    """

    testCookies()
    testRandomForestHealthy()
    testRandomForestRandom()
    testRandomForestUnhealthy()

if __name__ == "__main__":
    randomForestTestsLocal()