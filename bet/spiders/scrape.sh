#!/bin/bash

# split --lines=64 urls.csv
# for i in `ls x*`; do
#     scrapy crawl serieamatchspider -a filename="${i}" -o matches.csv -t csv
# done

split --lines=64 current_urls.1.csv
for i in `ls x*`; do
    scrapy crawl serieamatchspider -a filename="${i}" -o current_matches.1.csv -t csv
done

rm x*
