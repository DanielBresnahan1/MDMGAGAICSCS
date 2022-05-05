##
# @file testManVsMachine.py
#
# @brief Tests that the core functions of man vs machine are working
#
# @section author_sensors Author(s)
# - Created by Gabe Drew on 05/05/2022.

import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def testManVsMachine(url = "http://127.0.0.1:5000/"):
    """! Tests cookies functionality
        This test will boot a selenium driver, access the application at the specified url, and ensure that the necessary aspects of man vs machine are functional.
        
    @param url: The url which the test will attempt to reach. Defaults to a local url.
    """
    
    times0 = 0
    times1 = 0
    def choice(choice0, choice1, times0, times1):
        if random.randrange(0,2) == 1:
            times1 += 1
            return (times0, times1, choice1)
        else:
            times0 += 1
            return (times0, times1, choice0)

    def isPresent(element, attribute, searched):
        value = element.get_attribute(attribute)
        if(value):
            return value == searched
        else:
            False
    
    driver = webdriver.Firefox()
    driver.get(url)

    versus = driver.find_element_by_id("versus")
    versus.click()
    
    mvm = driver.find_element_by_id("MvM")
    mvm.click()




    choice0 = driver.find_element_by_id("healthy")
    choice1 = driver.find_element_by_id("unhealthy")


        #Assert that they are clickable
    assert choice0.is_enabled
    assert choice1.is_enabled
    data = choice(choice0, choice1, times0, times1)
    times0 = data[0]
    times1 = data[1]
    data[2].click()

    #Get through all 9
    for i in range(0, 9):
        choice0 = driver.find_element_by_id("healthy")
        choice1 = driver.find_element_by_id("unhealthy")

        assert choice0.is_enabled
        assert choice1.is_enabled
        data = choice(choice0, choice1, times0, times1)
        times0 = data[0]
        times1 = data[1]
        data[2].click()

    himages = driver.find_elements_by_id("himage")
    uimages = driver.find_elements_by_id("uimage")
    aihimages = driver.find_elements_by_id("aihimage")
    aiuimages = driver.find_elements_by_id("aiuimage")
    utp = 0
    ufp = 0
    utn = 0
    ufn = 0
    atp = 0
    afp = 0
    atn = 0
    afn = 0

    for image in himages:
        if isPresent(image, "class", "card border-success"):
            utp += 1
        else:
            ufp += 1

    for image in uimages:
        if isPresent(image, "class", "card border-success"):
            utn += 1
        else:
            ufn += 1

    for image in aihimages:
        if isPresent(image, "class", "card border-success"):
            atp += 1
        else:
            afp += 1

    for image in aiuimages:
        if isPresent(image, "class", "card border-success"):
            atn += 1
        else:
            afn += 1
    userSum = utp + ufp + ufn + utn
    aiSum = atp + afp + afn + atn
    userSumFalse = ufp + ufn
    aiSumFalse = afp + afn
    userAcc = 1 - (userSumFalse / userSum)
    aiAcc = 1 - (aiSumFalse / aiSum)

        

    #Test Results
    assert len(himages) == times0
    assert len(uimages) == times1
    print(utp)
    print(driver.find_element_by_id("utp").text)
    assert utp == int(driver.find_element_by_id("utp").text)
    assert ufp == int(driver.find_element_by_id("ufp").text)
    assert utn == int(driver.find_element_by_id("utn").text)
    assert ufn == int(driver.find_element_by_id("ufn").text)
    assert atp == int(driver.find_element_by_id("atp").text)
    assert afp == int(driver.find_element_by_id("afp").text)
    assert atn == int(driver.find_element_by_id("atn").text)
    assert afn == int(driver.find_element_by_id("afn").text)
    assert userAcc == float(driver.find_element_by_id("uacc").text)
    assert aiAcc == float(driver.find_element_by_id("aacc").text)

    

    #Make sure to reset cookies
    restart = driver.find_element_by_id("restart")
    restart.click()

    print("Test Man Vs Machine - Success")

if __name__ == "__main__":
    testManVsMachine()

