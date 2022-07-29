"""Extract demo set from Wiki dumps."""
from wiki import wiki_dump_api

if __name__ == '__main__':
    wiki_dump_api.extract_demo_dump({
        "Tirana", "Andorra la Vella", "Yerevan", "Vienna", "Baku", "Minsk", "Brussels", "Sarajevo", "Sofia", "Zagreb",        "Nicosia", "Prague", "Copenhagen", "Tallinn", "Helsinki", "Paris", "Tbilisi", "Berlin", "Athens", "Budapest",
        "Reykjavik", "Dublin", "Rome", "Nur-Sultan", "Pristina", "Riga", "Vaduz", "Vilnius", "Luxembourg(city)",
        "Valletta", "Chisinau", "Monaco", "Podgorica", "Amsterdam", "Skopje", "Oslo", "Warsaw", "Lisbon", "Bucharest",
        "Moscow", "San Marino", "Belgrade", "Bratislava", "Ljubljana", "Madrid", "Stockholm", "Bern", "Ankara", "Kyiv",
        "Kiev", "London", "Vatican City"
    })
