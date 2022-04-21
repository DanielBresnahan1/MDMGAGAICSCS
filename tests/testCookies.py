##
# @file testCookies.py
#
# @brief Tests that the required cookies are working. 
#
# @section author_sensors Author(s)
# - Created by Gabe Drew on 04/06/2022.

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def testCookies(url = "http://127.0.0.1:5000/"):
    """! Tests cookies functionality
        This test will boot a selenium driver, access the application at the specified url, and ensure that the necessary aspects of the cookies are functional.
        It will do this by going through the entire random forest section, returning to the beginning, and running thorugh the random forest section a second time.
        It will check that the second time it is only presented with five images, and that the ten specified images from before also appear.
    @param url: The url which the test will attempt to reach. Defaults to a local url.
    """
    
    driver = webdriver.Firefox()
    driver.get(url)

    forest = driver.find_element_by_id("forest")
    forest.click()

    choice0 = driver.find_element_by_id("choice-0")
    choice1 = driver.find_element_by_id("choice-1")
    submit = driver.find_element_by_id("submit")


    choice0.click()
    submit.click()

    #Get through all 9
    for i in range(0, 9):

        submit = driver.find_element_by_id("submit")

        submit.click()



    driver.get("http://127.0.0.1:5000/")
    forest = driver.find_element_by_id("forest")
    forest.click()


    choice0 = driver.find_element_by_id("choice-0")
    choice1 = driver.find_element_by_id("choice-1")
    submit = driver.find_element_by_id("submit")


    choice0.click()
    submit.click()

    #Get through all 9
    for i in range(0, 4):

        submit = driver.find_element_by_id("submit")

        submit.click()

    healthy = driver.find_element_by_id("healthy")
    unhealthy = driver.find_element_by_id("unhealthy")
    confidence = driver.find_element_by_id("confidence")

    #Test Results

    assert confidence.text == "Confidence: 100.00%"
    assert healthy.text == "Healthy(User): 15"
    assert unhealthy.text == "Unhealthy(User): 0"

    #Make sure to reset cookies
    restart = driver.find_element_by_id("restart")
    restart.click()


    print("Test Cookies - Success")

if __name__ == "__main__":
    testCookies()