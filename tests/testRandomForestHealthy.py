##
# @file testRandomForestHealthy.py
#
# @brief Tests random forest functionality with only healthy inputs
#
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from sympy import false
from selenium.common.exceptions import NoSuchElementException


def testRandomForestHealthy(url = "http://127.0.0.1:5000/"):
    """! Tests Random Forest functionality.
        This test will boot a selenium driver, access the application at the specified url, and run through the random forest code selecting only healthy.
        It will then double check that there are ten healthy images, zero unhealthy images, and that the confidence of the model is 100%.
    @param url: The url which the test will attempt to reach. Defaults to a local url.
    """
    
    driver = webdriver.Firefox()
    driver.get(url)

    forest = driver.find_element_by_id("forest")
    forest.click()

    choice0 = driver.find_element_by_id("choice-0")
    choice1 = driver.find_element_by_id("choice-1")

        #Assert that they are clickable
    assert choice0.is_enabled
    assert choice1.is_enabled
    choice0.click()

    #Get through all 9
    for i in range(0, 9):
        choice0 = driver.find_element_by_id("choice-0")
        choice1 = driver.find_element_by_id("choice-1")

        assert choice0.is_enabled
        assert choice1.is_enabled
        choice0.click()

    confidence = driver.find_element_by_id("confidence")
    himages = driver.find_elements_by_id("himage")
    uimages = driver.find_elements_by_id("uimage")
        

    #Test Results

    assert confidence.text == "Your model is 100.00% confident!"
    assert len(himages) == 10
    assert len(uimages) == 0

    #Make sure to reset cookies
    restart = driver.find_element_by_id("restart")
    restart.click()

    print("Test Random Forest Healthy - Success")

if __name__ == "__main__":
    testRandomForestHealthy()    