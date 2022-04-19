##
# @file testRandomForestUnhealthy.py
#
# @brief Tests random forest functionality with only unhealthy inputs
#
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def testRandomForestUnhealthy(url = "http://127.0.0.1:5000/"):

    """! Tests Random Forest functionality.
        This test will boot a selenium driver, access the application at the specified url, and run through the random forest code selecting only unhealthy.
        It will then double check that there are ten unhealthy images, zero healthy images, and that the confidence of the model is 100%.
    @param url: The url which the test will attempt to reach. Defaults to a local url.
    """

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
    choice1.click()
    assert submit.is_enabled
    submit.click()

    for i in range(0, 9):
        choice0 = driver.find_element_by_id("choice-0")
        choice1 = driver.find_element_by_id("choice-1")
        submit = driver.find_element_by_id("submit")

        #Assert that they are clickable
        assert choice0.is_enabled
        assert choice1.is_enabled
        assert submit.is_enabled
        submit.click()

    healthy = driver.find_element_by_id("healthy")
    unhealthy = driver.find_element_by_id("unhealthy")
    confidence = driver.find_element_by_id("confidence")
    himages = driver.find_elements_by_id("himage")
    uimages = driver.find_elements_by_id("uimage")

    assert confidence.text == "Confidence: 100.00%"
    assert healthy.text == "Healthy(User): 0"
    assert unhealthy.text == "Unhealthy(User): 10"
    assert len(himages) == 0
    assert len(uimages) == 10

    #Make sure to reset cookies
    restart = driver.find_element_by_id("restart")
    restart.click()

    print("Test Random Forest Unhealthy - Success")

if __name__ == "__main__":
    testRandomForestUnhealthy()