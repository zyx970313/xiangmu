import time
from selenium import webdriver
from selenium.webdriver import Chrome, ChromeOptions



if __name__ == '__main__':
    url = "http://localhost:3030/dataset.html?tab=upload&ds=/faults4.html"
    opt = ChromeOptions()  # 创建Chrome参数对象
    opt.headless = True  # 把Chrome设置成可视化无界面模式
    browser = Chrome(options=opt)  # 创建Chrome无界面对象
    # 访问图片上传的网页地址
    browser.get(url=url)
    time.sleep(3)
    # 点击图片上传按钮，打开文件选择窗口
    browser.find_element_by_name("files[]").send_keys(r"D:\pycode\library\d2rq\d2rq-0.8.1\faults_test1.nt")
    button = browser.find_element_by_css_selector("[class='btn btn-primary start action-upload-all']")
    button.click()
    browser.close()
