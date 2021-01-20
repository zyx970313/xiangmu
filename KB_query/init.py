import time
from selenium import webdriver







if __name__ == '__main__':
    url = "http://localhost:3030/manage.html"
    browser = webdriver.Chrome()
    browser.get(url=url)
    time.sleep(1)
    button = browser.find_element_by_css_selector("[class='btn btn-sm btn-primary']")
    button.click()
    time.sleep(1)
    name = browser.find_element_by_name('dbName')
    name.send_keys('faults4')
    for i in browser.find_elements_by_xpath("//*/input[@type='radio']"):
        i.click()
    button2 = browser.find_element_by_css_selector("[class='btn btn-sm btn-primary action commit simple']")
    button2.click()
    time.sleep(1)
    browser.close()



