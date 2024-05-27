from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import data_helper
import country_config
import logging
import time
import random



options = Options()
options.add_argument('--headless=new')
#options.add_argument('--enable-chrome-browser-cloud-management')
custom_user_agent = "Mozilla/5.0 (Linux; Android 11; 100011886A Build/RP1A.200720.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.69 Safari/537.36"
options.add_argument(f'user-agent={custom_user_agent}')

logging.basicConfig(level=logging.INFO,  # Adjust as needed
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='scraping_try3.log',  # Logs will be saved to this file
                    filemode='a')  # Append mode

def getResponse(url):
    driver = webdriver.Chrome(
    options=options)

    driver.get(url)

    response = driver.page_source

    driver.quit()

    return response



def getHTML(response):
   return BeautifulSoup(response, "html.parser")



def getSubLocationUrls(country_url):
    response = getResponse(country_url)
    soup = getHTML(response)
    sub_locations = soup.select("ul.facets.locations li.child a")
    sub_location_urls = [a['href'] for a in sub_locations]
    return sub_location_urls

def calculate_ads_per_sublocation(total_sub_locations, max_num):
    if total_sub_locations == 0:
        logging.warning("Warning: No sublocations have been detected.")
        return 0
    ads_per_sub_location = max_num // total_sub_locations
    logging.info(f"Ads per sublocation: {ads_per_sub_location}")
    return max(1, ads_per_sub_location)  



def crawlData(country_code):
    logging.info(f"Starting scrape for {country_code}")
    base_url = country_config.country_base_urls[country_code]
    search_url = country_config.country_search_urls[country_code]

    first_page = f"{base_url}{search_url}1"

    try:
        sub_location_urls = getSubLocationUrls(first_page)
    except Exception as e:
        logging.error(f"ERROR. Could not load sublocations. \n Exception: {e}")
    ads_per_sublocation = calculate_ads_per_sublocation(len(sub_location_urls), max_num=12000)


    for sub_url in sub_location_urls:
        
        page_num = 1  # Start from the first page
        ads_count = 0
        data = data_helper.data_frame


        while True:
            full_url = f"{base_url}{sub_url}?p={page_num}"
            list_jobs = None
            try:
                response = getResponse(full_url)
                soup = getHTML(response)
                list_jobs = soup.select_one("ul.jobs").select("li")
                logging.info(f"Fetching page {page_num} for location: {sub_url} in {country_code}")
            except Exception as e:
                logging.error(f"ERROR. Could not get a response from the website. \n Exception: {e}")


            if not list_jobs:
                break

            urls = []

            for job in list_jobs:
                article = job.select_one("article")
                if(article is not None):
                    url = article['data-url']
                    if("/jobad/" in url):
                        urls.append(f"{base_url}{url}")

            

            while urls:
                current_url = urls.pop()
                try:
                    logging.info(f"Fetching link {current_url}")
                    res = getResponse(current_url)
                    s = getHTML(res)
                    logging.info(f"Fetching complete {current_url}")

                    data["url"].append(current_url)
                    data["title"].append(s.select_one("div.container").select_one("h1").text.strip())
                    data["description"].append(s.select_one("section.content").text.strip())
                    data["date"].append("")  # Adjust based on availability
                    data["company"].append(s.select_one("p.company").text.strip())
                    data["country"].append(country_code)  # Add country code to each job listing

                    # Default values
                    data["location"].append("")
                    data["contractType"].append("")
                    data["contractTerm"].append("")
                    data["salary"].append("")

                    for tag in s.select_one("ul.details").find_all("li"):
                        icon = tag.select_one("svg.icon").select_one("use")["xlink:href"]
                        text = tag.get_text(strip=True)

                        if "location" in icon:
                            data["location"][-1] = text
                        elif "money" in icon:
                            data["salary"][-1] = text
                        elif "contract" in icon:
                            data["contractType"][-1] = text
                        elif "duration" in icon:
                            data["contractTerm"][-1] = text
                    logging.info(f"Successfully processed {current_url}")
                    time.sleep(random.uniform(1, 3))  # Sleep for a random time between 1 and 5 seconds
                except Exception as e:
                    logging.error(f"Error processing {current_url}: {e}")

            # Save and clear data periodically or after each country is processed
            try:
                data_helper.saveData(data)
                ads_count += len(data)  # Increment the counter for each ad processed

            except Exception as e:
                logging.error(f"Error saving page {page_num}  for location: {sub_url} in {country_code} : {e}")


            # Reset data for the next batch/country
            data = {key: [] for key in data}

            if ads_count > ads_per_sublocation:
                logging.info (f"Reached ad limit for {sub_url} in {country_code}.")
                break

            page_num += 1
            if page_num > 100:
                logging.info(f"Reached page limit for: {sub_url} in {country_code}.")
                break  # Check again in case the limit was reached during the page processing

    logging.info(f"Completed scrape for {country_code}")

def main():
    data_helper.initialize_database()

    for country_code in country_config.country_base_urls.keys():
        crawlData(country_code=country_code)

if __name__ == "__main__":
   main()


