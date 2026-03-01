"""
generate_mock_data.py
---------------------
Generates realistic synthetic hotel data for the NLP-Reco demo.

Outputs (written to data/):
  - hotel_info.csv    : 50 mock Turkish hotels
  - hotel_desc.csv    : ~200 Turkish reviews (3-4 per hotel)
  - session_data.csv  : 500 synthetic booking sessions

Run from the project root:
    python scripts/generate_mock_data.py

No external dependencies beyond pandas and numpy.
"""

import os
import random
import string
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITIES = ["İstanbul", "Antalya", "Bodrum", "İzmir", "Kapadokya", "Marmaris"]

SUBTOWNS = {
    "İstanbul": ["Beşiktaş", "Kadıköy", "Taksim", "Üsküdar", "Şişli"],
    "Antalya": ["Lara", "Konyaaltı", "Belek", "Side", "Alanya"],
    "Bodrum": ["Gümbet", "Türkbükü", "Yalıkavak", "Ortakent", "Bitez"],
    "İzmir": ["Alsancak", "Çeşme", "Alaçatı", "Urla", "Foça"],
    "Kapadokya": ["Göreme", "Ürgüp", "Avanos", "Uçhisar", "Mustafapaşa"],
    "Marmaris": ["Içmeler", "Armutalan", "Datça", "Bozburun", "Selimiye"],
}

FACILITY_TYPES = ["Otel", "Butik Otel", "Resort", "Apart Otel", "Villa"]

HOTEL_NAMES_TEMPLATES = [
    "{city} Grand Hotel",
    "{city} Palace",
    "{subtown} Boutique",
    "{city} Resort & Spa",
    "Hotel {adjective} {city}",
    "{subtown} Apart",
    "The {adjective} {city}",
    "{city} Suites",
]

ADJECTIVES = ["Royal", "Blue", "Golden", "Azure", "Majestic", "Serene", "Luxury", "Classic", "Elite", "Premium"]

# ---------------------------------------------------------------------------
# Turkish review phrase pools
# ---------------------------------------------------------------------------

BEACH_POOL_PHRASES = [
    "Havuz alanı çok geniş ve temizdi.",
    "Denize sıfır konumu sayesinde plaj erişimi mükemmeldi.",
    "Özel plaj alanı oldukça bakımlıydı.",
    "Sonsuzluk havuzu muhteşem bir manzara sunuyordu.",
    "Çocuk havuzu ve kaydıraklar aileler için idealdi.",
    "Deniz suyu tertemizdi, <br/>plaj şezlonları çok rahattı.",
    "Otel plajı sakin ve özeldi, kalabalık değildi.",
]

FOOD_PHRASES = [
    "Açık büfe kahvaltı çok zengin ve tazeydi.",
    "Akşam yemekleri Türk mutfağının en güzel örneklerini sunuyordu.",
    "Restoran personeli oldukça ilgiliydi.",
    "A la carte menüsündeki deniz ürünleri tazeydi.",
    "Sabah kahvaltıları için sunulan peynir ve zeytin çeşitleri harikaydı.",
    "Yemek kalitesi fiyatıyla orantılıydı.",
    "Barbekü akşamları otelin en keyifli anıydı.",
]

LOCATION_PHRASES = [
    "Konum merkeze yaklaşık 2 km mesafedeydi.",
    "Şehir merkezine yürüme mesafesindeydi.",
    "Havalimanına olan uzaklık mantıklıydı.",
    "Çevre düzenlemesi ve bahçesi son derece güzeldi.",
    "Tarihi alanlara ve müzelere çok yakındı.",
    "Otelden sahile yürüyerek kolayca ulaşılıyordu.",
    "Ulaşım imkânları yeterliydi, servis düzenliydi.",
]

ROOM_PHRASES = [
    "Odalar geniş ve ferahtı, temizliğe önem verilmişti.",
    "Balkon manzarası nefes kesiciydi.",
    "Yatak konforlu ve yastıklar kaliteliydi.",
    "Klima sistemi sorunsuz çalışıyordu.",
    "Banyoda jakuzi bulunması büyük artıydı.",
    "Oda servisi hızlı ve eksiksizdi.",
    "Ses yalıtımı iyiydi, komşu seslerinden rahatsız olmadık.",
]

SPA_PHRASES = [
    "Spa ve wellness merkezi profesyonelce yönetiliyordu.",
    "Hamam ve sauna tesisleri mükemmel durumdaydı.",
    "Masaj hizmetleri gerçekten rahatlatıcıydı.",
    "Fitness merkezi modern ekipmanlarla donatılmıştı.",
    "Türk hamamı deneyimi unutulmazdı.",
]

SERVICE_PHRASES = [
    "Resepsiyon ekibi 7/24 ilgiliydi ve her sorumuzu çözüme kavuşturdu.",
    "Check-in ve check-out işlemleri hızlı ve sorunsuzdur.",
    "Personel güler yüzlü ve yardımseverdi.",
    "Türkçe konuşan personel bulmak çok kolaylaştırdı iletişimi.",
    "Şikâyetlerimiz anında çözüldü, yönetim hassastı.",
    "Concierge hizmeti tur önerileri için çok yardımcıydı.",
]

PRICE_PHRASES = [
    "Fiyat-performans dengesi oldukça iyiydi.",
    "Verilen ücret için beklentilerin üzerinde hizmet aldık.",
    "Erken rezervasyon indirimi sayesinde çok uygun fiyata kaldık.",
    "Ekstra ücretler şeffaf bir şekilde bildirildi.",
    "Her şey dahil konsepti parasının hakkını veriyordu.",
]

NEGATIVE_PHRASES = [
    "Wi-Fi bağlantısı zaman zaman kesinti yaşattı.",
    "Otopark alanı biraz küçüktü.",
    "Bazı ortak alanlarda yenileme ihtiyacı hissediliyordu.",
]

# phrases to exercise HTML cleanup
HTML_VARIANTS = [
    "Harika bir tatil geçirdik.<br/>Kesinlikle tavsiye ederim.",
    "Oda temizliği mükemmeldi.<br/>Personel çok ilgiliydi.",
    "Plaj erişimi kolaydı.<br/>Deniz suyu temizdi.",
]

ALL_PHRASE_POOLS = [
    BEACH_POOL_PHRASES,
    FOOD_PHRASES,
    LOCATION_PHRASES,
    ROOM_PHRASES,
    SPA_PHRASES,
    SERVICE_PHRASES,
    PRICE_PHRASES,
]


_used_names: set = set()

def _make_hotel_name(city: str, subtown: str) -> str:
    for _ in range(20):  # try up to 20 combinations before adding a suffix
        template = random.choice(HOTEL_NAMES_TEMPLATES)
        name = template.format(city=city, subtown=subtown, adjective=random.choice(ADJECTIVES))
        if name not in _used_names:
            _used_names.add(name)
            return name
    # Fallback: append a disambiguating number
    base = f"{subtown} Hotel"
    i = 2
    while f"{base} {i}" in _used_names:
        i += 1
    unique_name = f"{base} {i}"
    _used_names.add(unique_name)
    return unique_name


def _random_rating() -> float:
    return round(random.uniform(6.0, 10.0), 1)


# ---------------------------------------------------------------------------
# Generate hotel_info.csv
# ---------------------------------------------------------------------------

def generate_hotel_info(n_hotels: int = 50) -> pd.DataFrame:
    rows = []
    hotel_id = 1001
    hotels_per_city = n_hotels // len(CITIES)

    for city in CITIES:
        subtowns = SUBTOWNS[city]
        for _ in range(hotels_per_city):
            subtown = random.choice(subtowns)
            rows.append({
                "hotel_id": hotel_id,
                "hotel_name": _make_hotel_name(city, subtown),
                "hotel_city": city,
                "hotel_subtown": subtown if random.random() > 0.05 else None,  # ~5% missing
                "facility_type": random.choice(FACILITY_TYPES) if random.random() > 0.03 else None,
                "hotel_avg_rating_general": _random_rating(),
                "hotel_avg_rating_food": _random_rating(),
                "hotel_avg_rating_cleaning": _random_rating(),
                "hotel_avg_rating_location": _random_rating(),
                "hotel_avg_rating_service": _random_rating(),
                "hotel_avg_rating_wifi": _random_rating(),
                "hotel_avg_rating_condition": _random_rating(),
                "hotel_avg_rating_price": _random_rating(),
                "hotel_comment_count": random.randint(20, 2000),
                "hotel_image_count": random.randint(5, 300),
            })
            hotel_id += 1

    # Fill remainder if n_hotels not perfectly divisible
    city = CITIES[0]
    while len(rows) < n_hotels:
        subtown = random.choice(SUBTOWNS[city])
        rows.append({
            "hotel_id": hotel_id,
            "hotel_name": _make_hotel_name(city, subtown),
            "hotel_city": city,
            "hotel_subtown": subtown,
            "facility_type": random.choice(FACILITY_TYPES),
            "hotel_avg_rating_general": _random_rating(),
            "hotel_avg_rating_food": _random_rating(),
            "hotel_avg_rating_cleaning": _random_rating(),
            "hotel_avg_rating_location": _random_rating(),
            "hotel_avg_rating_service": _random_rating(),
            "hotel_avg_rating_wifi": _random_rating(),
            "hotel_avg_rating_condition": _random_rating(),
            "hotel_avg_rating_price": _random_rating(),
            "hotel_comment_count": random.randint(20, 2000),
            "hotel_image_count": random.randint(5, 300),
        })
        hotel_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generate hotel_desc.csv
# ---------------------------------------------------------------------------

def _build_review(city: str) -> str:
    """Compose a Turkish review from several phrase pools."""
    parts = []
    # always pick 2-3 random pools
    chosen_pools = random.sample(ALL_PHRASE_POOLS, k=random.randint(2, 4))
    for pool in chosen_pools:
        parts.append(random.choice(pool))

    # occasionally add an HTML variant
    if random.random() < 0.25:
        parts.append(random.choice(HTML_VARIANTS))

    # occasionally add a "misafir" or "km" reference to exercise cleanup
    if random.random() < 0.2:
        parts.append(f"Toplam {random.randint(1, 10)} km mesafede birçok misafir dostu restoran bulunuyor.")

    # city-specific colour
    city_flavour = {
        "İstanbul": "İstanbul'un tarihi dokusu otelin yakınında hissediliyordu.",
        "Antalya": "Antalya'nın muhteşem sahil şeridine bakan bir konumda yer alıyor.",
        "Bodrum": "Bodrum'un yat limanı manzarası balkondan izlenebiliyordu.",
        "İzmir": "İzmir'in canlı yaşamı otele yansımış gibi hissettirdi.",
        "Kapadokya": "Kapadokya'nın peri bacaları manzarası odadan görünüyordu.",
        "Marmaris": "Marmaris körfezinin berrak suları otelin hemen önündeydi.",
    }
    parts.append(city_flavour.get(city, ""))

    return " ".join(p for p in parts if p)


def generate_hotel_desc(hotel_info: pd.DataFrame, reviews_per_hotel: int = 4) -> pd.DataFrame:
    rows = []
    for _, hotel in hotel_info.iterrows():
        n = random.randint(reviews_per_hotel - 1, reviews_per_hotel)
        for _ in range(n):
            rows.append({
                "hotel_id": hotel["hotel_id"],
                "description_text": _build_review(hotel["hotel_city"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generate session_data.csv
# ---------------------------------------------------------------------------

def generate_session_data(hotel_ids, n_sessions: int = 500) -> pd.DataFrame:
    base_date = date(2024, 1, 1)
    rows = []
    funnel_id = 90001

    for _ in range(n_sessions):
        check_in_offset = random.randint(0, 365)
        check_in = base_date + timedelta(days=check_in_offset)
        stay_nights = random.randint(1, 14)
        check_out = check_in + timedelta(days=stay_nights)

        rows.append({
            "funnel_id": funnel_id,
            "adult_count": random.randint(1, 4),
            "child_count": random.randint(0, 3),
            "check_in_date": check_in.isoformat(),
            "check_out_date": check_out.isoformat(),
            "hotel_id": random.choice(hotel_ids),
        })
        funnel_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating hotel_info.csv (50 hotels)...")
    hotel_info = generate_hotel_info(n_hotels=50)
    hotel_info.to_csv(os.path.join(out_dir, "hotel_info.csv"), index=False)
    print(f"  Written: {len(hotel_info)} rows")

    print("Generating hotel_desc.csv (~200 reviews)...")
    hotel_desc = generate_hotel_desc(hotel_info, reviews_per_hotel=4)
    hotel_desc.to_csv(os.path.join(out_dir, "hotel_desc.csv"), index=False)
    print(f"  Written: {len(hotel_desc)} rows")

    print("Generating session_data.csv (500 sessions)...")
    session_data = generate_session_data(hotel_info["hotel_id"].tolist(), n_sessions=500)
    session_data.to_csv(os.path.join(out_dir, "session_data.csv"), index=False)
    print(f"  Written: {len(session_data)} rows")

    print("\nDone. Files written to data/")


if __name__ == "__main__":
    main()
