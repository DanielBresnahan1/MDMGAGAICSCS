##
# @file allTestsReal.py
#
# @brief Runs all tests for the entire site, directed at the real application.
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.
from randomForestTestsReal import randomForestTestsReal
from testHomePage import testHomePage
from tests.testManVsMachine import testManVsMachine

##
# Specifies the url at which the real application is deployed.
url = "https://mdmgcapstone.herokuapp.com/"


testHomePage(url)
randomForestTestsReal(url)
testManVsMachine(url)