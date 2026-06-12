from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import recommender

app = FastAPI(
    title="Global Bite Recommendation API",
    description="Service API Rekomendasi Kuliner Lintas Kota berbasis Machine Learning untuk Capstone Project Global Bite.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Global Bite AI API is running! Access /docs for swagger documentation.",
        "model_loaded": recommender.df_unique_menu is not None
    }

@app.get("/api/recommend/cross-city")
def get_cross_city_recommendation(
    liked: List[str] = Query(..., description="Daftar makanan yang disukai user di kota asal, contoh: ['Nasi Goreng', 'Bakso']"),
    city: str = Query(..., description="Nama kota tujuan traveling, contoh: 'Tokyo'"),
    n: int = Query(5, description="Jumlah makanan rekomendasi yang dikembalikan")
):
    """
    Rekomendasi Lintas Kota (Cross-City):
    Mencari kuliner lokal di kota tujuan yang rasanya mirip dengan makanan-makanan kesukaan user di kota asal.
    """
    res = recommender.recommend_cross_city(liked, city, top_n=n)
    if isinstance(res, dict) and "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.get("/api/recommend/by-tags")
def get_tag_based_recommendation(
    tags: List[str] = Query(..., description="Daftar preferensi rasa user, contoh: ['spicy', 'savory']"),
    city: str = Query(..., description="Nama kota tujuan, contoh: 'Bangkok'"),
    n: int = Query(5, description="Jumlah makanan rekomendasi yang dikembalikan")
):
    """
    Rekomendasi Berdasarkan Rasa (Tag-Based):
    Mencari kuliner lokal di kota tertentu yang sesuai dengan preferensi rasa pilihan user (cocok untuk onboarding user baru).
    """
    res = recommender.recommend_by_tags(tags, city, top_n=n)
    if isinstance(res, dict) and "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.get("/api/recommend/top")
def get_top_popular_menus(
    city: str = Query(..., description="Nama kota, contoh: 'Seoul'"),
    n: int = Query(5, description="Jumlah makanan terpopuler yang dikembalikan")
):
    """
    Rekomendasi Populer Lokal (Fallback):
    Mengambil makanan terpopuler di kota tujuan (berdasarkan ulasan dan rating) jika user belum memiliki profil rasa.
    """
    res = recommender.get_top_by_city(city, top_n=n)
    if isinstance(res, dict) and "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.get("/api/recommend/similar")
def get_similar_menu_recommendations(
    menu: str = Query(..., description="Nama menu asal, contoh: 'Nasi Goreng'"),
    n: int = Query(5, description="Jumlah makanan serupa yang dikembalikan")
):
    """
    Rekomendasi Menu Serupa secara Global (Content-Based):
    Mencari menu makanan lain yang mirip secara rasa dan konten dengan menu masukan.
    """
    res = recommender.get_similar_menus(menu, top_n=n)
    if isinstance(res, dict) and "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.get("/api/cities")
def get_all_cities():
    """
    Daftar Kota & Negara:
    Mengembalikan semua nama kota dan negara yang tersedia di dalam database kuliner model.
    """
    cities = recommender.get_available_cities()
    if not cities:
        raise HTTPException(status_code=500, detail="Database model kosong atau tidak ter-load.")
    return {
        "total_cities": len(cities),
        "cities": cities
    }

@app.get("/api/tags")
def get_all_tags():
    """
    Daftar 23 Preference Tags:
    Mengembalikan daftar semua tag preferensi rasa makanan yang didukung sistem.
    """
    if not recommender.VALID_TAGS:
        return {
            "total_tags": len(recommender.TAG_KEYWORDS.keys()),
            "tags": sorted(list(recommender.TAG_KEYWORDS.keys()))
        }
    return {
        "total_tags": len(recommender.VALID_TAGS),
        "tags": recommender.VALID_TAGS
    }

@app.get("/api/menu/detail")
def get_menu_merchant_detail(
    menu: str = Query(..., description="Nama menu, contoh: 'Yakisoba'"),
    city: str = Query(..., description="Nama kota, contoh: 'Tokyo'")
):
    """
    Detail Menu & Restoran:
    Mengambil deskripsi lengkap menu serta daftar semua restoran (merchants) yang menjualnya di kota tersebut.
    """
    res = recommender.get_menu_detail(menu, city)
    if isinstance(res, dict) and "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res

@app.get("/api/menus")
def get_all_menus():
    menus = recommender.get_all_menus()

    return {
        "total_menus": len(menus),
        "menus": menus
    }

@app.get("/api/menus/search")
def search_menus(
    q: str = Query(
        ...,
        description="Keyword pencarian menu"
    )
):
    menus = recommender.search_menus(q)

    return {
        "total_results": len(menus),
        "menus": menus
    }

@app.get("/api/merchants")
def get_all_merchants():
    merchants = recommender.get_all_merchants()

    return {
        "total_merchants": len(merchants),
        "merchants": merchants
    }

@app.get("/api/countries")
def get_all_countries():

    countries = recommender.get_available_countries()

    return {
        "total_countries": len(countries),
        "countries": countries
    }