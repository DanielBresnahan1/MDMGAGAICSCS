from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def testHomePage(url = "http://127.0.0.1:5000/"):

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