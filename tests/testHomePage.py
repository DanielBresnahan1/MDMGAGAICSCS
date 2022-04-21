##
# @file testHomePage.py
#
# @brief Tests that the home page is functional
#
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def testHomePage(url = "http://127.0.0.1:5000/"):
    """! Tests home page functionality
        This test will boot a selenium driver, access the application at the specified url, and ensure that the necessary aspects of the homepage exist and are functional.
        It will do this by checking that the necessary buttons exist and are enabled.
    @param url: The url which the test will attempt to reach. Defaults to a local url.
    """

    driver = webdriver.Firefox()
    driver.get(url)
    #Get the two critical buttons on the page
    forest = driver.find_element_by_id("forest")
    versus = driver.find_element_by_id("versus")

    #Assert that they are clickable
    assert versus.is_enabled
    assert forest.is_enabled
    print("Test Home Page - Success")

if __name__ == "__main__":
    testHomePage()