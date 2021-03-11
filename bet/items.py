# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from collections import OrderedDict


class BetItem(scrapy.Item):
    # define the fields for your item here like:
    day = scrapy.Field()
    time = scrapy.Field()
    event = scrapy.Field()
    tournament = scrapy.Field()
    bookie = scrapy.Field()
    market = scrapy.Field()
    outcome = scrapy.Field()
    odds = scrapy.Field()
    url = scrapy.Field()

class MatchItemURL(scrapy.Item):
    # define the fields for your item here like:
    url = scrapy.Field()

class MatchItem(scrapy.Item):

    # def __init__(self, *args, **kwargs):
    #     self._values = OrderedDict()
    #     if kwargs:
    #         for k, v in kwargs.items():
    #             self[k] = v

    # define the fields for your item here like:
    Date = scrapy.Field()
    Round = scrapy.Field()
    Home_Team = scrapy.Field()
    Away_Team = scrapy.Field()
    Home_Goals = scrapy.Field()
    Away_Goals = scrapy.Field()
    Result = scrapy.Field()
    Home_Possesso_palla = scrapy.Field()
    Home_Parate = scrapy.Field()
    Home_Rigori = scrapy.Field()
    Home_Tiri_Totali = scrapy.Field()
    Home_Tiri_in_porta = scrapy.Field()
    Home_Tiri_in_porta_da_punizione = scrapy.Field()
    Home_Tiri_fuori = scrapy.Field()
    Home_Tiri_fuori_da_punizione = scrapy.Field()
    Home_Tiri_in_porta_da_area = scrapy.Field()
    Home_Tiri_in_porta_su_azione_da_palla_inattiva = scrapy.Field()
    Home_Tiri_fuori_su_azione_da_palla_inattiva = scrapy.Field()
    Home_Lanci_lunghi = scrapy.Field()
    Home_Cross = scrapy.Field()
    Home_Falli_commessi = scrapy.Field()
    Home_Falli_subiti = scrapy.Field()
    Home_Pali = scrapy.Field()
    Home_Occasioni_da_gol = scrapy.Field()
    Home_Assist_Totali = scrapy.Field()
    Home_Fuorigioco = scrapy.Field()
    Home_Corner = scrapy.Field()
    Home_Passaggi_riusciti = scrapy.Field()
    Home_Accuratezza_passaggi = scrapy.Field()
    Home_Ammonizioni = scrapy.Field()
    Home_Doppie_ammonizioni = scrapy.Field()
    Home_Espulsioni = scrapy.Field()
    Home_Passaggi_chiave = scrapy.Field()
    Home_Recuperi = scrapy.Field()
    Home_Ripartenze_da_Recupero = scrapy.Field()
    Home_Attacchi_centrali = scrapy.Field()
    Home_Attacchi_a_destra = scrapy.Field()
    Home_Attacchi_a_sinistra = scrapy.Field()
    Away_Possesso_palla = scrapy.Field()
    Away_Parate = scrapy.Field()
    Away_Rigori = scrapy.Field()
    Away_Tiri_Totali = scrapy.Field()
    Away_Tiri_in_porta = scrapy.Field()
    Away_Tiri_in_porta_da_punizione = scrapy.Field()
    Away_Tiri_fuori = scrapy.Field()
    Away_Tiri_fuori_da_punizione = scrapy.Field()
    Away_Tiri_in_porta_da_area = scrapy.Field()
    Away_Tiri_in_porta_su_azione_da_palla_inattiva = scrapy.Field()
    Away_Tiri_fuori_su_azione_da_palla_inattiva = scrapy.Field()
    Away_Lanci_lunghi = scrapy.Field()
    Away_Cross = scrapy.Field()
    Away_Falli_commessi = scrapy.Field()
    Away_Falli_subiti = scrapy.Field()
    Away_Pali = scrapy.Field()
    Away_Occasioni_da_gol = scrapy.Field()
    Away_Assist_Totali = scrapy.Field()
    Away_Fuorigioco = scrapy.Field()
    Away_Corner = scrapy.Field()
    Away_Passaggi_riusciti = scrapy.Field()
    Away_Accuratezza_passaggi = scrapy.Field()
    Away_Ammonizioni = scrapy.Field()
    Away_Doppie_ammonizioni = scrapy.Field()
    Away_Espulsioni = scrapy.Field()
    Away_Passaggi_chiave = scrapy.Field()
    Away_Recuperi = scrapy.Field()
    Away_Ripartenze_da_Recupero = scrapy.Field()
    Away_Attacchi_centrali = scrapy.Field()
    Away_Attacchi_a_destra = scrapy.Field()
    Away_Attacchi_a_sinistra = scrapy.Field()
