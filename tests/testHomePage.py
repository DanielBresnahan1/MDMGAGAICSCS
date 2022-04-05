from selenium import webdriver
from selenium.webdriver.common.keys import Keys


driver = webdriver.Firefox()
driver.get("http://127.0.0.1:5000/")
#Get the two critical buttons on the page
forest = driver.find_element_by_id("forest")
versus = driver.find_element_by_id("versus")

#Assert that they are clickable
assert versus.is_enabled
assert forest.is_enabled
print("Success")