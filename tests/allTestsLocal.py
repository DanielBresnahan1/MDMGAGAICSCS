##
# @mainpage AG-AI Test Documention
#
# @section description_main Description
# Documentation for the tests for the AG-AI application
##
# @file allTestsLocal.py
#
# @brief Runs all tests for the entire site, directed at a locally run version of the application.
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.


from randomForestTestsLocal import randomForestTestsLocal
from testHomePage import testHomePage
from tests.testManVsMachine import testManVsMachine


testHomePage()
randomForestTestsLocal()
testManVsMachine()