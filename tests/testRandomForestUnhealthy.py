from selenium import webdriver
from selenium.webdriver.common.keys import Keys


driver = webdriver.Firefox()
driver.get("http://127.0.0.1:5000/")

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

print("Success")