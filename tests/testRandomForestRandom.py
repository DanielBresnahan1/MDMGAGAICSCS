##
# @file testRandomForestRandom.py
#
# @brief Tests random forest functionality with a random mix of healthy and unhealthy inputs
#
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.
import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from sympy import false
from selenium.common.exceptions import NoSuchElementException

def testRandomForestRandom(url = "http://127.0.0.1:5000/"):    
    
    """! Tests Random Forest functionality.
        This test will boot a selenium driver, access the application at the specified url, and run through the random forest code selecting healthy and unhealthy randomly.
        It will then double check that there are the correct number of healthy and unhealthy images.
    @param url: The url which the test will attempt to reach. Defaults to a local url.
    """
    times0 = 0
    times1 = 0
    def choice(choice0, choice1, times0, times1):
        if random.randrange(0,2) == 1:
            times1 += 1
            choice1.click()
            return (times0, times1)
        else:
            times0 += 1
            choice0.click()
            return (times0, times1)

    driver = webdriver.Firefox()
    driver.get(url)

    forest = driver.find_element_by_id("forest")
    forest.click()

    choice0 = driver.find_element_by_id("choice-0")
    choice1 = driver.find_element_by_id("choice-1")
    submit = driver.find_element_by_id("submit")

        #Assert that they are clickable
    assert choice0.is_enabled
    assert choice1.is_enabled
    data = choice(choice0, choice1, times0, times1)
    times0 = data[0]
    times1 = data[1]
    assert submit.is_enabled
    submit.click()

    #Get through all 9
    for i in range(0, 9):
        choice0 = driver.find_element_by_id("choice-0")
        choice1 = driver.find_element_by_id("choice-1")
        submit = driver.find_element_by_id("submit")

        assert choice0.is_enabled
        assert choice1.is_enabled
        assert submit.is_enabled
        data = choice(choice0, choice1, times0, times1)
        times0 = data[0]
        times1 = data[1]
        submit.click()

    healthy = driver.find_element_by_id("healthy")
    unhealthy = driver.find_element_by_id("unhealthy")
    himages = driver.find_elements_by_id("himage")
    uimages = driver.find_elements_by_id("uimage")
        

    #Test Results

    assert healthy.text == "Healthy(User): " + str(times0)
    assert unhealthy.text == "Unhealthy(User): " + str(times1)

    assert len(himages) == times0
    assert len(uimages) == times1

    #Make sure to reset cookies
    restart = driver.find_element_by_id("restart")
    restart.click()

    print("Test Random Forest Random - Success")

if __name__ == "__main__":
    testRandomForestRandom()

