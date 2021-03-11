import re
import sys
import json
import time
import scrapy
import datetime
import numpy as np
import pandas as pd
from dateutil import tz
from scrapy import signals
import chromedriver_binary
from selenium import webdriver
from scrapy.selector import Selector
from selenium.webdriver.firefox.options import Options

from collections import OrderedDict
from ..items import BetItem, MatchItem, MatchItemURL

# General utilities
weekdays = [ "monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday", "today", "tomorrow", "future" ]

def convert_odds(odds):

    nums = [ float(x.split("/")[0]) for x in odds ]
    dens = [ float(x.split("/")[1]) for x in odds ]
    odds = [ frac_to_decimal(nums[i], dens[i]) for i in range(len(nums)) ]

    return odds


def frac_to_decimal(num, den):
    '''
    Function to convert fractional odds to decimal odds.

    Parameters
    ----------
    num: float.
        Numerator of the fractional odds.
    den: float.
        Denominator of the fractional odds.

    Returns
    -------
    dec: float.
        Decimal odds.
    '''

    dec = np.round(num / den + 1, decimals=2)

    return dec


# Parsing spiders
class WHSpider(scrapy.Spider):
    '''
    Class to parse William Hill for odds.
    '''

    name = 'whspider'
    bookie = 'William Hill'
    allowed_domains = ['https://sports.williamhill.it/betting/it-it']
    start_urls = ['https://sports.williamhill.it']
    base_url = start_urls[0]

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed,
                                signal=signals.spider_closed)
        crawler.signals.connect(spider.spider_opened,
                                signal=signals.spider_opened)

        return spider

    def spider_opened(self, spider):
        options = Options()
        options.headless = True
        self.browser = webdriver.Firefox(options=options)

    def spider_closed(self, spider):
        self.browser.quit()

    def parse(self, response):
        '''
        Function to parse start_urls.

        Parameters
        ----------
        response: scrapy.Request object.
            Request data obtained from start_urls.

        Returns
        -------
        iterator: scrapy.Request object.
            Iterator over the matches_urls.
        '''

        urls = response.xpath('//a[@class="c-list__item"]/@href').extract()
        urls = list(set([ self.base_url + url for url in urls
                          if "betting" in url ]))

        # Here we manually select football, but both matches and competitions
        # can be appended to any sport to parse odds
        urls = [ url for url in urls if url.split("/")[-1] == "calcio" ]
        parse_urls = []
        for url in urls:
            parse_urls.append(url + '/matches')

            # Competitions not necessary atm since we parse day by day
            # parse_urls.append(url + '/competitions')

        for url in parse_urls:
            if url.split("/")[-1] == "matches":
                yield scrapy.Request(url, callback=self.parse_days,
                                     dont_filter=True)

            # elif url.split("/")[0] == "competitions":
            #     pass

    def parse_days(self, response):
        '''
        Function to parse a matches page to obtain a list of urls with the
        daily matches.

        Parameters
        ----------
        response: scrapy.Request object.
            Request data obtained from start_urls.

        Returns
        -------
        iterator: scrapy.Request object.
            Iterator over the daily matches urls.
        '''

        # Here we need the web browser to load dynamic content
        self.browser.get(response.url)
        time.sleep(3)
        page = Selector(text=self.browser.page_source)

        daypth = '//div[contains(@data-test-id, "Carousel")]/a/@href'
        urls = [ self.base_url + x for x in page.xpath(daypth).extract()
                 if x.split("/")[-2] in weekdays ]

        for url in urls:
            yield scrapy.Request(url, callback=self.parse_matches,
                                 dont_filter=True)

    def parse_matches(self, response):
        '''
        Function to parse a daily matches page to obtain a list of urls with
        the market for each event.

        Parameters
        ----------
        response: scrapy.Request object.
            Request data obtained from start_urls.

        Returns
        -------
        iterator: scrapy.Request object.
            Iterator over the event markets.
        '''

        # Here we need the web browser to load dynamic content
        self.browser.get(response.url)
        time.sleep(3)
        page = Selector(text=self.browser.page_source)

        pth = '//div[@class="sp-o-market__title"]/a/@href'
        urls = [ self.base_url + x for x in page.xpath(pth).extract() ]

        for url in urls:
            yield scrapy.Request(url, callback=self.parse_event_markets,
                                 dont_filter=True)

    def parse_competition(self, response):
        '''
        Function to parse a competition page to obtain a list of urls with
        the market for each event.

        Parameters
        ----------
        response: scrapy.Request object.
            Request data obtained from start_urls.

        Returns
        -------
        iterator: scrapy.Request object.
            Iterator over the event markets.
        '''

        # Here we need the web browser to load dynamic content
        self.browser.get(response.url)
        time.sleep(3)
        page = Selector(text=self.browser.page_source)
        pth = '//main[@class="sp-o-market__title"]/a/@href'
        urls = [ self.base_url + x for x in page.xpath(pth).extract() ]

        for url in urls:
            yield scrapy.Request(url, callback=self.parse_event_markets,
                                 dont_filter=True)

    def parse_event_markets(self, response):
        '''
        Function to parse an event page to obtain a list of markets and odds.

        Parameters
        ----------
        response: scrapy.Request object.
            Request data obtained from start_urls.

        Returns
        -------
        data: dict.
            Dictionary with event metadata and odds.
        '''

        # Metadata
        pth = '//span[@class="pageTitle__page-title___22oul"]/h2/text()'
        event = response.xpath(pth).extract_first()
        tournament = response.xpath('//span[@class="css-19shaaq"]/a/span/text()').extract()[-2]
        date = response.xpath('//span[@class="event-start-date"]/@data-startdate').extract_first()
        dateobj = datetime.datetime.fromisoformat(date)
        localdate = dateobj.astimezone(tz.tzlocal())
        day = localdate.strftime('%d-%m-%Y')
        t = localdate.strftime('%H:%M')

        # Betting data
        subitem = {}
        markets = response.xpath('//section[@class="event-container scrollable"]')
        for market in markets:
            name = market.xpath('.//header/h2/text()').extract_first()
            outcomes = market.xpath('.//p/span/text()').extract()
            fracodds = market.xpath('.//button/span[@class="betbutton__odds"]/text()').extract()
            odds = convert_odds(fracodds)
            subitem[name] = dict(zip(outcomes, odds))

        for market, outcomes in subitem.items():
            for outcome, odds in outcomes.items():

                bet = BetItem()

                bet['day']        = day
                bet['time']       = t
                bet['event']      = event
                bet['tournament'] = tournament
                bet['bookie']     = self.bookie
                bet['market']     = market
                bet['outcome']    = outcome
                bet['odds']       = odds
                bet['url']        = response.url

                yield bet

# Test to check whether multiple spiders can be dealt with
# Apparently scrapy can run them simultaneously, but maybe
# it is complicated to deal with data writing. We will see
# afterwards
class BFSpider(scrapy.Spider):
    '''
    Class to parse BetFair Exchange for odds.
    '''

    name = 'bfspider'
    bookie = 'Betfair'
    allowed_domains = ['https://www.betfair.it/exchange/plus']
    start_urls = ['https://www.betfair.it/exchange/plus']
    base_url = start_urls[0]

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed,
                                signal=signals.spider_closed)
        crawler.signals.connect(spider.spider_opened,
                                signal=signals.spider_opened)

        return spider

    def spider_opened(self, spider):
        options = Options()
        options.headless = True
        self.browser = webdriver.Firefox(options=options)

    def spider_closed(self, spider):
        self.browser.quit()

    def parse(self, response):
        '''
        Run on the main page to get sports to parse.
        '''

        # Here we need the web browser to load dynamic content
        self.browser.get(response.url)
        time.sleep(3)
        page = Selector(text=self.browser.page_source)

        urls = page.xpath('//a[@data-link-type="SPORT"]/@href').extract()
        urls = [ url for url in urls if "calcio" in url ]
        urls = [ self.base_url + x for x in urls ]
        print(urls)


class SerieASpider(scrapy.Spider):
    '''
    Class to parse Serie A historical data.
    '''

    name = 'serieaspider'
    allowed_domains = ['http://www.legaseriea.it']
    start_urls = ['http://www.legaseriea.it/it/serie-a/archivio']
    base_url = allowed_domains[0]
    base_arch = start_urls[0]

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed,
                                signal=signals.spider_closed)
        crawler.signals.connect(spider.spider_opened,
                                signal=signals.spider_opened)

        return spider

    def spider_opened(self, spider):
        options = Options()
        options.headless = True
        self.browser = webdriver.Firefox(options=options)

    def spider_closed(self, spider):
        self.browser.quit()

    def parse(self, response):
        '''
        Run on the main page to get seasons.
        '''

        # Seasons
        seasons = response.xpath('//select[@name="stagione"]/option/text()').extract()
        seasons = seasons[:10]
        urls = [ self.base_arch + '/' + x for x in seasons ]

        for url in urls:
            yield scrapy.Request(url, callback=self.parse_season,
                                 dont_filter=True)

    def parse_season(self, response):
        '''
        Run on season page to get rounds.
        '''

        # Rounds
        urls1 = response.xpath('//li[@class="box_Ngiornata_andata"]/a/@href').extract()
        urls2 = response.xpath('//li[@class="box_Ngiornata_ritorno"]/a/@href').extract()
        urls = urls1 + urls2
        urls = [ self.base_url + x for x in urls ]

        for url in urls:
            yield scrapy.Request(url, callback=self.parse_round,
                                 dont_filter=True)

    def parse_round(self, response):
        '''
        Run on round page to parse match reports
        '''

        # Reports
        urls = response.xpath('//div[@class="link-matchreport"]/a/@href').extract()
        urls = [ self.base_url + x for x in urls if "report" in x ]

        for url in urls:
            item = MatchItemURL(url=url)
            yield item


class CurrentSerieASpider(scrapy.Spider):
    '''
    Class to parse Serie A current data.
    '''

    name = 'currentserieaspider'
    allowed_domains = ['http://www.legaseriea.it']
    start_urls = ['http://www.legaseriea.it/it/serie-a/calendario-e-risultati']
    base_url = allowed_domains[0]
    base_arch = start_urls[0]

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed,
                                signal=signals.spider_closed)
        crawler.signals.connect(spider.spider_opened,
                                signal=signals.spider_opened)

        return spider

    def spider_opened(self, spider):
        options = Options()
        options.headless = True
        self.browser = webdriver.Firefox(options=options)

    def spider_closed(self, spider):
        self.browser.quit()

    def parse(self, response):
        '''
        Run on season page to get rounds.
        '''

        # Rounds
        urls1 = response.xpath('//li[@class="box_Ngiornata_andata"]/a/@href').extract()
        urls2 = response.xpath('//li[@class="box_Ngiornata_ritorno"]/a/@href').extract()
        urls = urls1 + urls2
        urls = [ self.base_url + x for x in urls ]

        for url in urls:
            yield scrapy.Request(url, callback=self.parse_round,
                                 dont_filter=True)

    def parse_round(self, response):
        '''
        Run on round page to parse match reports
        '''

        # Reports
        urls = response.xpath('//div[@class="link-matchreport"]/a/@href').extract()
        urls = [ self.base_url + x for x in urls if "report" in x ]

        for url in urls:
            item = MatchItemURL(url=url)
            yield item


class SerieAMatchSpider(scrapy.Spider):
    '''
    Class to parse Serie A historical data.
    '''

    name = 'serieamatchspider'
    allowed_domains = ['http://www.legaseriea.it']
    base_url = allowed_domains[0]

    custom_settings = {
            "FEED_EXPORT_FIELDS" : [
                'Date',
                'Round',
                'Home_Team',
                'Away_Team',
                'Home_Goals',
                'Away_Goals',
                'Result',
                'Home_Possesso_palla',
                'Home_Parate',
                'Home_Rigori',
                'Home_Tiri_Totali',
                'Home_Tiri_in_porta',
                'Home_Tiri_in_porta_da_punizione',
                'Home_Tiri_fuori',
                'Home_Tiri_fuori_da_punizione',
                'Home_Tiri_in_porta_da_area',
                'Home_Tiri_in_porta_su_azione_da_palla_inattiva',
                'Home_Tiri_fuori_su_azione_da_palla_inattiva',
                'Home_Falli_commessi',
                'Home_Falli_subiti',
                'Home_Pali',
                'Home_Occasioni_da_gol',
                'Home_Assist_Totali',
                'Home_Fuorigioco',
                'Home_Corner',
                'Home_Passaggi_riusciti',
                'Home_Accuratezza_passaggi',
                'Home_Ammonizioni',
                'Home_Doppie_ammonizioni',
                'Home_Espulsioni',
                'Home_Passaggi_chiave',
                'Home_Recuperi',
                'Home_Ripartenze_da_Recupero',
                'Home_Attacchi_centrali',
                'Home_Attacchi_a_destra',
                'Home_Attacchi_a_sinistra',
                'Away_Possesso_palla',
                'Away_Parate',
                'Away_Rigori',
                'Away_Tiri_Totali',
                'Away_Tiri_in_porta',
                'Away_Tiri_in_porta_da_punizione',
                'Away_Tiri_fuori',
                'Away_Tiri_fuori_da_punizione',
                'Away_Tiri_in_porta_da_area',
                'Away_Tiri_in_porta_su_azione_da_palla_inattiva',
                'Away_Tiri_fuori_su_azione_da_palla_inattiva',
                'Away_Falli_commessi',
                'Away_Falli_subiti',
                'Away_Pali',
                'Away_Occasioni_da_gol',
                'Away_Assist_Totali',
                'Away_Fuorigioco',
                'Away_Corner',
                'Away_Passaggi_riusciti',
                'Away_Accuratezza_passaggi',
                'Away_Ammonizioni',
                'Away_Doppie_ammonizioni',
                'Away_Espulsioni',
                'Away_Passaggi_chiave',
                'Away_Recuperi',
                'Away_Ripartenze_da_Recupero',
                'Away_Attacchi_centrali',
                'Away_Attacchi_a_destra',
                'Away_Attacchi_a_sinistra'
                ]
            }

    def __init__(self, filename=None):
        if filename is not None:
            with open(filename) as f:
                self.start_urls = list(map(str.strip, f.readlines()[1:]))

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed,
                                signal=signals.spider_closed)
        crawler.signals.connect(spider.spider_opened,
                                signal=signals.spider_opened)

        return spider

    def spider_opened(self, spider):
        options = Options()
        options.headless = True
        self.browser = webdriver.Firefox(options=options)

    def spider_closed(self, spider):
        self.browser.quit()

    def parse(self, response):
        '''
        Run on report page to parse information
        '''

        # Match Report
        date = response.xpath('//div[@class="report-data"]/span/text()').extract_first()
        date = date.split("-")[0].strip()

        rnd = response.xpath('//h2/text()').extract_first().split("|")[-1].strip()
        rnd = re.findall('\d+', rnd)[0]

        home = response.xpath('//h3[@class="report-squadra squadra-a"]/span/text()').extract_first()
        away = response.xpath('//h3[@class="report-squadra squadra-b"]/span/text()').extract_first()

        home_goals = response.xpath('//div[@class="squadra-risultato squadra-a"]/text()').extract_first()
        away_goals = response.xpath('//div[@class="squadra-risultato squadra-b"]/text()').extract_first()

        home_goals = int(home_goals)
        away_goals = int(away_goals)

        if home_goals > away_goals:
            result = "1"
        elif home_goals < away_goals:
            result = "2"
        else:
            result = "X"

        stat_names = response.xpath('//div[@class="valoretitolo"]/text()').extract()
        h_stat_names = [ "Home_" + x.replace(" ", "_") for x in stat_names ]
        a_stat_names = [ "Away_" + x.replace(" ", "_") for x in stat_names ]
        home_stats = response.xpath('//div[@class="valoresx"]/text()').extract()
        away_stats = response.xpath('//div[@class="valoredx"]/text()').extract()
        home_dict = dict(zip(h_stat_names, home_stats))
        away_dict = dict(zip(a_stat_names, away_stats))
        data_dict = {
                "Date" : date,
                "Round" : rnd,
                "Home_Team" : home,
                "Away_Team" : away,
                "Home_Goals" : home_goals,
                "Away_Goals" : away_goals,
                "Result" : result,
                }

        data_dict.update(home_dict)
        data_dict.update(away_dict)
        del data_dict['Home_Passaggi_nella_3/4_avversaria']
        del data_dict['Away_Passaggi_nella_3/4_avversaria']

        match = MatchItem(data_dict)

        yield match
